#!/usr/bin/env python3
"""S63: reference-anchored, prefix-local preference learning.

Why this replaces "cached reverse KL"
-------------------------------------
The revised stage-2 registration (S60) asked for reverse KL on the student's
sampled completion tokens with a **cached scalar teacher log-probability** per
token. That cache cannot implement reverse KL. At a prefix ``h``,

    KL(pi_theta(.|h) || p_T(.|h))
        = sum_y pi_theta(y|h) [log pi_theta(y|h) - log p_T(y|h)]

needs ``p_T`` over the whole vocabulary (or at least the support pi_theta puts
mass on). A cached ``log p_T(a_t | h_t)`` for the one sampled action ``a_t`` is
one term of that sum: it is not normalised, it says nothing about the other
``|V|-1`` actions, and summing it over a trajectory is a likelihood, not a KL.
Calling that difference a reverse KL would be a false claim about the
objective.

What is implemented instead
---------------------------
A DPO/IPO-style preference objective on pairs that share the **student prefix**:

* ``chosen``  = a verified-correct teacher continuation from that prefix;
* ``rejected`` = the student's own failed continuation from the same prefix;
* the comparison is anchored on the frozen post-trained reference ``pi_0``.

For a pair ``(h, a+, a-)`` the per-pair margin is

    m = beta * [ (log pi_theta(a+|h) - log pi_theta(a-|h))
                 - (log pi_0(a+|h)     - log pi_0(a-|h)) ]

and the loss is ``-log sigmoid(m)``. Only per-action log-probabilities are
needed, so a scalar cache of the frozen reference is sufficient and exact.

Token-length normalisation (stated explicitly)
----------------------------------------------
By default each sequence log-probability is the **mean log-probability per
continuation token** (``normalization="mean"``), for the student and for the
frozen reference alike. A long failed continuation therefore cannot win the
pair merely by having more tokens. ``normalization="sum"`` is available for a
diagnostic but is not the registered objective; whichever is used is written
into the run receipt. The continuation token counts are always reported.

Teacher-guided recovery for p = 0
---------------------------------
The PACED weight is ``w(p) = p(1-p)``, which is exactly zero for a problem the
student never solves (``p=0``) and for one it always solves (``p=1``). The
``p=0`` case still has a verified teacher solution available. The recovery arm
adds those pairs with the **matched mean positive weight** of the on-policy
pairs, so it is neither dropped nor allowed to dominate. The recovery pair's
rejected side is the student's failed continuation from the same prefix.

LoRA and merge
--------------
The adapter is rank-64 LoRA on every linear layer (``alpha=32``, the usual
``alpha/rank`` scaling), with zero-initialised ``B`` and Kaiming-uniform ``A``.
``merge_lora`` folds ``scale * B @ A`` into each base weight and removes the
adapter modules. The deployed model is then a full-weight text tower: no
adapter parameter is a separate deployed identity, so the unique active union
after the merge is the base model's 752,393,024 elements.

The module is CPU-testable with tiny tensors and a tiny model; the GPU run is a
separate, receipted job.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Registered geometry.
REGISTERED_LORA_RANK = 64
REGISTERED_LORA_ALPHA = 32
REGISTERED_ACTIVE_UNION = 752_393_024
REGISTERED_NORMALIZATION = "mean"
#: Stated merge tolerance for float32 tiny-model parity.
MERGE_ATOL = 1e-5
MERGE_RTOL = 1e-4

NORMALIZATIONS = ("mean", "sum")

#: How a run consumes the pair set. `shuffled` reproduces every run before S70
#: exactly (the same `random.Random(seed).shuffle` stream); `weighted` draws a
#: difficulty-proportional permutation without replacement inside each epoch.
PAIR_ORDERS = ("shuffled", "weighted")


# ---------------------------------------------------------------------------
# pairs


@dataclass(frozen=True)
class PreferencePair:
    """One prefix, a chosen teacher continuation and a rejected student one."""

    problem_id: str
    prompt_ids: tuple[int, ...]
    chosen_ids: tuple[int, ...]
    rejected_ids: tuple[int, ...]
    weight: float = 1.0
    chosen_reference_logprobs: tuple[float, ...] | None = None
    rejected_reference_logprobs: tuple[float, ...] | None = None
    provenance: str = "on-policy"

    def __post_init__(self):
        if not self.chosen_ids:
            raise ValueError("a preference pair needs a chosen continuation")
        if not self.rejected_ids:
            raise ValueError("a preference pair needs a rejected continuation")
        if self.weight < 0.0:
            raise ValueError("a preference weight must be non-negative")


def assert_shared_prefix(prompt_a, prompt_b) -> None:
    """Fail closed unless both continuations are conditioned on the same prefix."""
    if tuple(prompt_a) != tuple(prompt_b):
        raise ValueError("the pair does not share the student prefix")


def make_pair(problem_id: str, prompt_ids, chosen_ids, rejected_ids,
              weight: float = 1.0) -> PreferencePair:
    return PreferencePair(
        problem_id=str(problem_id),
        prompt_ids=tuple(int(x) for x in prompt_ids),
        chosen_ids=tuple(int(x) for x in chosen_ids),
        rejected_ids=tuple(int(x) for x in rejected_ids),
        weight=float(weight))


def difficulty_weight(p: float) -> float:
    """PACED weight w(p) = p(1-p), clamped to the probability simplex."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    return p * (1.0 - p)


def recovery_weight(positive_weights) -> float:
    """The matched weight for a p=0 recovery pair.

    Returns the mean of the positive on-policy weights. A problem with p=0 has
    w=0, so without this rule its verified teacher solution would be dropped
    exactly where the student needs it most. If every on-policy weight is zero,
    the recovery weight is 1.0 and the receipt says so.
    """
    values = [float(weight) for weight in positive_weights if weight > 0.0]
    if not values:
        return 1.0
    return sum(values) / len(values)


def recovery_pair(problem_id: str, prompt_ids, teacher_ids, failed_ids,
                  p: float, on_policy_weights) -> PreferencePair:
    """Build the teacher-guided pair for a problem the student never solved."""
    if p != 0.0:
        raise ValueError("the recovery arm is only for problems with p == 0")
    if not teacher_ids:
        raise ValueError("a recovery pair needs a verified teacher solution")
    return PreferencePair(
        problem_id=str(problem_id),
        prompt_ids=tuple(int(x) for x in prompt_ids),
        chosen_ids=tuple(int(x) for x in teacher_ids),
        rejected_ids=tuple(int(x) for x in failed_ids),
        weight=recovery_weight(on_policy_weights),
        provenance="teacher-recovery")


def build_recovery_arm(ledger_entries, teacher_by_id, failed_by_id
                       ) -> tuple[list[PreferencePair], dict]:
    """Build the recovery pairs from the per-problem ledger.

    `ledger_entries` is the rollout ledger (each row has `id`, `p` and the
    attempt counters). `teacher_by_id` maps a problem id to a verified teacher
    solution token list and `failed_by_id` to the student's failed continuation
    token list. Only problems with p == 0 are candidates; a missing teacher or
    failed continuation is counted, never invented. The weight is the mean
    positive p(1-p) of the on-policy problems.
    """
    entries = list(ledger_entries)
    positive = [float(entry.get("weight_p1mp", 0.0)) for entry in entries]
    pairs: list[PreferencePair] = []
    missing_teacher = []
    missing_failed = []
    for entry in entries:
        if float(entry.get("p", 0.0)) != 0.0:
            continue
        problem_id = str(entry.get("id"))
        teacher = teacher_by_id.get(problem_id)
        failed = failed_by_id.get(problem_id)
        if not teacher:
            missing_teacher.append(problem_id)
            continue
        if not failed:
            missing_failed.append(problem_id)
            continue
        prompt = entry.get("prompt_ids")
        if not prompt:
            raise ValueError(
                f"ledger row {problem_id!r} has no prompt_ids; the recovery "
                "pair needs the shared prefix")
        pairs.append(recovery_pair(problem_id, prompt, teacher, failed, 0.0,
                                   positive))
    zero = sum(1 for entry in entries if float(entry.get("p", 0.0)) == 0.0)
    return pairs, {
        "zero_p_problems": zero,
        "recovery_pairs": len(pairs),
        "missing_teacher": missing_teacher,
        "missing_failed_continuation": missing_failed,
        "matched_weight": recovery_weight(positive),
        "basis": "p=0 problems receive the mean positive p(1-p) weight; "
                 "missing data is counted, not fabricated",
    }


# ---------------------------------------------------------------------------
# the difficulty weight, made real


def corpus_mean_weight(pairs) -> float:
    """The mean weight over the positive-weight pairs, computed once.

    This is the fixed denominator of the corpus-normalised objective. It is a
    property of the pair file, not of a batch, so it is identical for every
    step and can be written into the run receipt before training starts.
    """
    values = [float(pair.weight) for pair in pairs if float(pair.weight) > 0.0]
    if not values:
        raise ValueError("the corpus mean weight needs a positive-weight pair")
    return sum(values) / len(values)


def weighted_permutation(weights, rng: random.Random) -> list[int]:
    """A deterministic weighted permutation without replacement.

    Efraimidis-Spirakis keys in log form: draw ``u_i`` uniformly and sort by
    ``log(1 - u_i) / w_i`` descending. The first element is drawn with
    probability ``w_i / sum(w)``, and the whole permutation is a sample without
    replacement proportional to ``w``, so over many epochs each pair is visited
    in proportion to its weight. Exactly one raw draw is consumed per index, in
    ascending index order, so an independent implementation that does the same
    reproduces the order element for element.
    """
    keyed: list[tuple[float, int]] = []
    for index, weight in enumerate(weights):
        value = float(weight)
        if not value > 0.0:
            raise ValueError(
                "a weighted permutation needs strictly positive weights; the "
                "launcher drops weight <= 0 before ordering")
        # ``1 - u`` keeps the argument of log in (0, 1], so a draw of exactly
        # 1.0 is the best key instead of a -inf sentinel.
        keyed.append((math.log(1.0 - rng.random()) / value, index))
    keyed.sort(key=lambda item: (-item[0], item[1]))
    return [index for _, index in keyed]


def pair_stream(pairs, *, seed: int, order: str = "shuffled"):
    """Yield pair indices forever, one epoch at a time.

    ``shuffled`` is the pre-S70 rule and is bit-identical to it: one
    ``random.Random(seed)``, one ``shuffle`` per epoch, so an old run is
    reproducible from its seed. ``weighted`` draws a fresh weighted permutation
    per epoch from the same generator, so the two modes differ only in how each
    epoch is ordered.
    """
    if order not in PAIR_ORDERS:
        raise ValueError(f"unknown pair order {order!r}")
    if not pairs:
        raise ValueError("a pair stream needs at least one pair")
    rng = random.Random(seed)
    weights = [float(pair.weight) for pair in pairs]
    epoch = list(range(len(pairs)))
    while True:
        if order == "shuffled":
            # In place, exactly like the pre-S70 launcher: the second epoch
            # shuffles the first epoch's permutation, not the identity list.
            rng.shuffle(epoch)
        else:
            epoch = weighted_permutation(weights, rng)
        yield from epoch


# ---------------------------------------------------------------------------
# log-probabilities and the loss


def normalise_logprob(total: torch.Tensor, tokens: int,
                      normalization: str = REGISTERED_NORMALIZATION):
    """Apply the stated token-length normalisation."""
    if normalization not in NORMALIZATIONS:
        raise ValueError(f"unknown normalisation {normalization!r}")
    if tokens <= 0:
        raise ValueError("a continuation must have at least one token")
    if normalization == "mean":
        return total / float(tokens)
    return total


def sequence_logprob(logprobs: torch.Tensor, normalization: str
                     = REGISTERED_NORMALIZATION) -> torch.Tensor:
    """Sum or mean of a per-token log-probability vector."""
    total = logprobs.reshape(-1).sum()
    return normalise_logprob(total, int(logprobs.numel()), normalization)


def _gather_continuation(logits: torch.Tensor, start: int,
                         continuation_ids: torch.Tensor) -> torch.Tensor:
    """Per-token log-probabilities of a continuation conditioned on a prefix.

    ``logits`` is ``[batch, seq, vocab]`` and `start` is the first position
    that predicts a continuation token. When the caller asked the model for
    only the continuation positions, ``start`` is 0 and ``seq`` equals the
    continuation length.
    """
    count = int(continuation_ids.numel())
    start = int(start)
    if start < 0 or start + count > logits.shape[1]:
        raise ValueError(
            "logits do not cover the continuation: start "
            f"{start}, continuation {count}, seq {logits.shape[1]}")
    selected = logits[:, start:start + count, :].float()
    log_probs = F.log_softmax(selected, dim=-1)
    return log_probs.gather(
        2, continuation_ids.reshape(1, count, 1)).reshape(-1)


def continuation_logprobs(model, prompt_ids, continuation_ids,
                          device: str | None = None,
                          normalization: str = REGISTERED_NORMALIZATION
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(normalised log-prob, per-token log-probabilities)``."""
    import torch as _torch

    if device is None:
        try:
            device = next(model.parameters()).device
        except (AttributeError, StopIteration):
            device = None
    prompt = _torch.as_tensor(list(prompt_ids), dtype=_torch.long,
                              device=device).reshape(1, -1)
    continuation = _torch.as_tensor(list(continuation_ids), dtype=_torch.long,
                                    device=device).reshape(1, -1)
    inputs = _torch.cat([prompt, continuation[:, :-1]], dim=1)
    # Only the continuation positions' logits are consumed. Asking the LM head
    # for exactly those positions is exact (every other position's logits are
    # multiplied by zero in the loss) and keeps a 1.3k-token row inside the
    # device: materialising the full [1, seq, 248k] tensor was the S65 smoke's
    # CUDA OOM.
    keep = _torch.arange(prompt.shape[1] - 1,
                         prompt.shape[1] - 1 + continuation.shape[1],
                         device=inputs.device)
    logits = model(input_ids=inputs, use_cache=False,
                   logits_to_keep=keep).logits
    per_token = _gather_continuation(logits, 0, continuation.reshape(-1))
    return sequence_logprob(per_token, normalization), per_token


@torch.no_grad()
def cache_reference_logprobs(reference_model, pair: PreferencePair,
                             device: str | None = None,
                             normalization: str = REGISTERED_NORMALIZATION
                             ) -> PreferencePair:
    """Fill both reference caches from the frozen reference model pi_0.

    This is the only place a reference model is run. No teacher weights are
    needed at training time: the cached per-action values are sufficient for
    the preference loss.
    """
    _, chosen = continuation_logprobs(reference_model, pair.prompt_ids,
                                      pair.chosen_ids, device, normalization)
    _, rejected = continuation_logprobs(reference_model, pair.prompt_ids,
                                        pair.rejected_ids, device,
                                        normalization)
    return replace(pair,
                   chosen_reference_logprobs=tuple(
                       float(value) for value in chosen.tolist()),
                   rejected_reference_logprobs=tuple(
                       float(value) for value in rejected.tolist()))


def reference_logprob(cached, normalization: str
                      = REGISTERED_NORMALIZATION) -> torch.Tensor:
    """Normalise a cached per-token reference log-probability vector."""
    if cached is None:
        raise ValueError(
            "the pair has no cached frozen-reference log-probabilities; cache "
            "them with cache_reference_logprobs (a scalar teacher cache is not "
            "a reverse KL and is not accepted here)")
    values = torch.as_tensor(list(cached), dtype=torch.float32)
    return sequence_logprob(values, normalization)


def preference_loss(chosen_logprob: torch.Tensor,
                    rejected_logprob: torch.Tensor,
                    chosen_reference_logprob: torch.Tensor,
                    rejected_reference_logprob: torch.Tensor,
                    beta: float = 0.1,
                    weights: torch.Tensor | None = None) -> torch.Tensor:
    """``-log sigmoid(beta * [(s+ - s-) - (r+ - r-)])``, optionally weighted.

    The arguments may be scalars or one-dimensional tensors. The default
    sequence log-probabilities are token-length-normalised means, so the
    subtraction is between per-token averages.
    """
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    margin = beta * ((chosen_logprob - rejected_logprob)
                     - (chosen_reference_logprob - rejected_reference_logprob))
    losses = -F.logsigmoid(margin)
    if weights is None:
        return losses.mean()
    weights = weights.reshape(-1).float()
    losses = losses.reshape(-1)
    if losses.numel() != weights.numel():
        raise ValueError("loss and weight tensors differ in length")
    return (losses * weights).sum() / weights.sum().clamp_min(1e-12)


def pair_loss(model, pair: PreferencePair, beta: float = 0.1,
              device: str | None = None,
              normalization: str = REGISTERED_NORMALIZATION
              ) -> tuple[torch.Tensor, dict]:
    """Student preference loss for one cached pair."""
    s_chosen, _ = continuation_logprobs(model, pair.prompt_ids, pair.chosen_ids,
                                        device, normalization)
    s_rejected, _ = continuation_logprobs(model, pair.prompt_ids,
                                          pair.rejected_ids, device,
                                          normalization)
    r_chosen = reference_logprob(pair.chosen_reference_logprobs, normalization)
    r_rejected = reference_logprob(pair.rejected_reference_logprobs,
                                   normalization)
    # The cache is built on CPU; move it to the student's device before the
    # subtraction. The values themselves are unchanged.
    r_chosen = r_chosen.to(s_chosen.device)
    r_rejected = r_rejected.to(s_rejected.device)
    loss = preference_loss(s_chosen, s_rejected, r_chosen, r_rejected, beta)
    margin = beta * ((s_chosen - s_rejected) - (r_chosen - r_rejected))
    return loss, {
        "margin": float(margin.detach()),
        "student_chosen": float(s_chosen.detach()),
        "student_rejected": float(s_rejected.detach()),
        "reference_chosen": float(r_chosen),
        "reference_rejected": float(r_rejected),
        "chosen_tokens": len(pair.chosen_ids),
        "rejected_tokens": len(pair.rejected_ids),
        "normalization": normalization,
    }


def preference_batch_loss(model, pairs, beta: float = 0.1,
                          device: str | None = None,
                          normalization: str = REGISTERED_NORMALIZATION,
                          weight_normalizer: float | None = None
                          ) -> tuple[torch.Tensor, dict]:
    """Weighted batch loss over pairs, with per-pair diagnostics.

    ``weight_normalizer=None`` keeps the pre-S70 denominator, the batch's own
    weight sum. At ``batch_pairs = 1`` that denominator is the single pair's
    weight, so the weight cancels exactly and the objective is unweighted: the
    registered ``w = p(1-p)`` is inert. Supplying the corpus mean weight (see
    :func:`corpus_mean_weight`) divides by a constant instead, so a pair with
    ``w < mean_w`` contributes proportionally less and a pair with ``w > mean_w``
    proportionally more. The supplied normaliser is never replaced by the
    batch's own sum.
    """
    losses, weights, diagnostics = [], [], []
    for pair in pairs:
        loss, detail = pair_loss(model, pair, beta, device, normalization)
        losses.append(loss)
        weights.append(float(pair.weight))
        diagnostics.append({"problem_id": pair.problem_id,
                            "weight": float(pair.weight), **detail})
    weight_tensor = torch.tensor(weights, dtype=torch.float32,
                                 device=losses[0].device)
    stacked = torch.stack(losses)
    weight_sum = float(weight_tensor.sum())
    if weight_normalizer is None:
        denominator = weight_tensor.sum().clamp_min(1e-12)
        normalizer = None
        basis = ("batch weight sum (the S64 behaviour; at batch_pairs = 1 the "
                 "weight cancels and the objective is unweighted)")
    else:
        normalizer = float(weight_normalizer)
        if not normalizer > 0.0:
            raise ValueError("weight_normalizer must be positive when supplied")
        denominator = normalizer
        basis = "corpus mean weight over the positive-weight pairs"
    scaled = stacked * weight_tensor
    loss = scaled.sum() / denominator
    for detail, weight in zip(diagnostics, weights):
        detail["weight_scale"] = (weight / normalizer if normalizer
                                  else weight / weight_sum)
    return loss, {"pairs": len(pairs),
                  "weight_sum": weight_sum,
                  "weight_normalizer": normalizer,
                  "weight_normalizer_basis": basis,
                  "mean_loss": float(loss.detach()),
                  "diagnostics": diagnostics,
                  "normalization": normalization}


# ---------------------------------------------------------------------------
# all-linear LoRA and the merge path


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable rank-``r`` update.

    ``h -> W h + (alpha / r) * B (A h)``. ``B`` starts at zero, so the adapter
    is the identity at initialisation. ``merge`` folds the update into ``W``.
    """

    def __init__(self, base: nn.Linear, rank: int = REGISTERED_LORA_RANK,
                 alpha: float = REGISTERED_LORA_ALPHA):
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be >= 1")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        # Allocate on the base layer's device: `install_lora` runs after the
        # model is already placed, so a CPU-allocated adapter would make the
        # first forward fail with a cross-device matmul.
        device = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(
            self.rank, base.in_features, dtype=torch.float32, device=device))
        self.lora_B = nn.Parameter(torch.zeros(
            base.out_features, self.rank, dtype=torch.float32,
            device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        # Compute in the activation's dtype (bf16 for the real student) while
        # the master A/B stay float32 for the optimizer. Upcasting every
        # activation to float32 retained a full-sequence float32 copy at each
        # of the 186 adapted layers, which is what filled the S65 smoke's card.
        # For a float32 model this is exactly the previous computation.
        dtype = x.dtype
        update = F.linear(F.linear(x, self.lora_A.to(dtype)),
                          self.lora_B.to(dtype))
        return base_out + (self.scaling * update).to(base_out.dtype)

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        return self.base.weight + self.scaling * (self.lora_B @ self.lora_A)

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        self.base.weight.copy_(self.merged_weight().to(self.base.weight.dtype))
        return self.base


def default_linear_target(name: str, module: nn.Module) -> bool:
    """Every linear layer except the output head."""
    return isinstance(module, nn.Linear) and not name.endswith("lm_head")


def install_lora(model: nn.Module, rank: int = REGISTERED_LORA_RANK,
                 alpha: float = REGISTERED_LORA_ALPHA, target=None,
                 seed: int | None = None) -> list:
    """Replace every targeted ``nn.Linear`` in-place with ``LoRALinear``.

    ``seed`` is applied to ``random``, ``numpy`` (when importable) and
    ``torch`` immediately before the randomly initialised ``A`` factors are
    drawn, so the initial adapter state is reproducible and can be hashed.
    Pair ordering uses its own seed and is not affected by this global seed.
    """
    if seed is not None:
        seed = int(seed)
        random.seed(seed)
        try:
            import numpy as _numpy
            _numpy.random.seed(seed % (2 ** 32))
        except Exception:  # numpy is optional for the adapter init
            pass
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    target = target or default_linear_target
    installed = []

    def walk(parent: nn.Module, prefix: str) -> None:
        for name, child in list(parent.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and target(full, child):
                wrapper = LoRALinear(child, rank, alpha)
                setattr(parent, name, wrapper)
                installed.append(wrapper)
            elif not isinstance(child, LoRALinear):
                walk(child, full)

    walk(model, "")
    return installed


def lora_parameters(model: nn.Module) -> list:
    parameters = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            parameters.extend([module.lora_A, module.lora_B])
    return parameters


def adapter_element_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in lora_parameters(model))


def adapter_state_sha256(model: nn.Module) -> str:
    """Canonical digest of every LoRA factor, in ``named_modules`` order.

    Hashes the name, dtype, shape and bytes of both factors of every adapter,
    so the receipt can prove the initial state a run started from rather than
    asserting that a seed was set somewhere.
    """
    digest = hashlib.sha256()
    found = 0
    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        for part in ("lora_A", "lora_B"):
            tensor = getattr(module, part).detach().to("cpu").contiguous()
            digest.update(
                f"{name}.{part}|{tensor.dtype}|{tuple(tensor.shape)}|"
                .encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        found += 1
    if not found:
        raise ValueError("the model has no LoRA modules to hash")
    return digest.hexdigest()


def merge_lora(model: nn.Module) -> dict:
    """Fold every adapter into its base weight and remove the wrappers."""
    merged = 0

    def walk(parent: nn.Module) -> None:
        nonlocal merged
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                setattr(parent, name, child.merge())
                merged += 1
            else:
                walk(child)

    trainable_before = sum(parameter.numel()
                           for parameter in lora_parameters(model))
    walk(model)
    remaining = sum(1 for module in model.modules()
                    if isinstance(module, LoRALinear))
    if remaining:
        raise RuntimeError(f"{remaining} LoRA modules survived the merge")
    return {"merged_layers": merged,
            "adapter_elements_removed": trainable_before,
            "remaining_lora_modules": remaining,
            "basis": "scale * B @ A added to each base weight; the adapter "
                     "parameters cease to be separate deployed identities"}


def merged_state_from_model(model: nn.Module,
                            exclude_names=()) -> tuple[dict, dict]:
    """Export the merged full-arm state on CPU without copying the model.

    The previous path deep-copied the text tower on its own device and then
    moved the copy to CPU, which doubled GPU residency for the whole merge.
    Here each base weight and its rank-``r`` factors are moved to CPU one
    layer at a time and the update ``scale * B @ A`` is folded there; the
    training model is never copied and never mutated.

    Keys are returned in the unmounted model's namespace
    (``...q_proj.weight``), exactly what ``merge_lora`` produces, and the
    ``lora_A``/``lora_B`` parameters are not exported. ``exclude_names`` drops
    frozen shared parameters (the tied embedding) from the state without
    changing the element accounting.
    """
    import torch

    excluded = set(exclude_names)
    state: dict = {}
    merged = 0
    adapter_elements = 0
    wrapped = {name for name, module in model.named_modules()
               if isinstance(module, LoRALinear)}
    for name, parameter in model.named_parameters():
        if name.endswith(".lora_A") or name.endswith(".lora_B"):
            adapter_elements += int(parameter.numel())
            continue
        if ".base." in name:
            module_name, suffix = name.split(".base.", 1)
            module = dict(model.named_modules()).get(module_name)
            if not isinstance(module, LoRALinear):
                raise RuntimeError(
                    f"{name}: base parameter without a LoRALinear wrapper")
            if suffix == "weight":
                base_cpu = parameter.detach().to("cpu").clone()
                factors_a = module.lora_A.detach().to("cpu").float()
                factors_b = module.lora_B.detach().to("cpu").float()
                update = factors_b @ factors_a
                merged_weight = (base_cpu.float()
                                 + module.scaling * update).to(base_cpu.dtype)
                state[f"{module_name}.{suffix}"] = merged_weight
                merged += 1
                wrapped.discard(module_name)
            elif name not in excluded:
                state[f"{module_name}.{suffix}"] = (
                    parameter.detach().to("cpu").clone())
            continue
        if name in excluded:
            continue
        state[name] = parameter.detach().to("cpu").clone()
    if wrapped:
        raise RuntimeError(
            f"{len(wrapped)} LoRA modules were not exported "
            f"(first {sorted(wrapped)[:3]})")
    report = {"merged_layers": merged,
              "adapter_elements_removed": adapter_elements,
              "remaining_lora_modules": 0,
              "basis": "scale * B @ A folded into CPU copies of each base "
                       "weight; the training model is not copied and the "
                       "adapter is not exported as a deployed identity"}
    return state, report


def active_union_after_merge(base_elements: int,
                             adapter_elements: int) -> int:
    """The unique active union after the adapter is folded into the base.

    The adapter does not add a deployed identity: its elements are written into
    weights the base already reads. The union is therefore the base count, not
    ``base + adapter``.
    """
    if base_elements <= 0:
        raise ValueError("base_elements must be positive")
    if adapter_elements < 0:
        raise ValueError("adapter_elements must be non-negative")
    return int(base_elements)


def unique_active_elements(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


# ---------------------------------------------------------------------------
# matched SFT control


def sft_control_loss(model, pair: PreferencePair, device: str | None = None,
                     normalization: str = REGISTERED_NORMALIZATION
                     ) -> torch.Tensor:
    """Completion CE on the chosen continuation only (no rejected/reference)."""
    logprob, _ = continuation_logprobs(model, pair.prompt_ids, pair.chosen_ids,
                                        device, normalization)
    return -logprob


def sft_batch_loss(model, pairs, device: str | None = None, *,
                   weights=None, weight_normalizer: float | None = None,
                   normalization: str = REGISTERED_NORMALIZATION
                   ) -> tuple[torch.Tensor, dict]:
    """Batch completion CE, optionally difficulty-weighted.

    ``weights=None`` is the unweighted matched-SFT control (mean CE over the
    batch). With per-pair ``weights`` and a positive ``weight_normalizer`` the
    loss is ``sum(w_i * CE_i) / mean_w`` -- the same weighting mechanism the
    preference objective uses -- so the two controls differ in the objective,
    not in whether the difficulty weight is realized.
    """
    if not pairs:
        raise ValueError("the SFT batch loss needs at least one pair")
    stacked = torch.stack([sft_control_loss(model, pair, device, normalization)
                           for pair in pairs])
    chosen_tokens = sum(len(pair.chosen_ids) for pair in pairs)
    if weights is None:
        return stacked.mean(), {
            "pairs": len(pairs),
            "chosen_tokens": chosen_tokens,
            "weighting": "uniform",
            "mean_loss": float(stacked.detach().mean()),
        }
    weight_tensor = torch.tensor([float(weight) for weight in weights],
                                 dtype=torch.float32, device=stacked.device)
    if weight_tensor.numel() != stacked.numel():
        raise ValueError("loss and weight tensors differ in length")
    if weight_normalizer is None or not float(weight_normalizer) > 0.0:
        raise ValueError("difficulty-weighted SFT needs a positive "
                         "weight_normalizer")
    loss = (stacked * weight_tensor).sum() / float(weight_normalizer)
    return loss, {
        "pairs": len(pairs),
        "chosen_tokens": chosen_tokens,
        "weighting": "difficulty",
        "weight_normalizer": float(weight_normalizer),
        "mean_loss": float(loss.detach()),
    }


def preference_processed_tokens(pairs) -> int:
    """Forward token positions used by one preference pass over the pairs.

    Chosen and rejected are two separate forwards, each conditioned on the same
    prefix, so the prompt is paid twice.
    """
    return sum(2 * len(pair.prompt_ids)
               + len(pair.chosen_ids) + len(pair.rejected_ids)
               for pair in pairs)


def sft_processed_tokens(pairs) -> int:
    return sum(len(pair.prompt_ids) + len(pair.chosen_ids) for pair in pairs)


def run_matched_sft_control(model, pairs, budget_tokens: int | None = None,
                            lr: float = 1e-4, device: str | None = None,
                            max_steps: int = 10_000) -> dict:
    """Run completion SFT on the chosen continuations, same examples/budget.

    The default budget is the number of forward token positions the preference
    objective uses (``preference_processed_tokens``), so the control sees the
    same examples and the same processed-token budget. It may pass over the
    chosen examples several times; the report states the realised processed and
    supervised token counts. Fail closed if there is nothing trainable.
    """
    if not pairs:
        raise ValueError("the matched SFT control needs at least one pair")
    trainable = [parameter for parameter in model.parameters()
                 if parameter.requires_grad]
    if not trainable:
        raise ValueError("the matched SFT control needs trainable parameters")
    budget = (preference_processed_tokens(pairs)
              if budget_tokens is None else int(budget_tokens))
    if budget <= 0:
        raise ValueError("the matched SFT control needs a positive budget")
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    processed = 0
    supervised = 0
    steps = 0
    last_loss = None
    while processed < budget and steps < max_steps:
        pair = pairs[steps % len(pairs)]
        optimizer.zero_grad(set_to_none=True)
        loss = sft_control_loss(model, pair, device)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())
        processed += len(pair.prompt_ids) + len(pair.chosen_ids)
        supervised += len(pair.chosen_ids)
        steps += 1
    return {
        "kind": "matched_sft_control",
        "steps": steps,
        "budget_processed_tokens": budget,
        "processed_tokens": processed,
        "supervised_chosen_tokens": supervised,
        "final_loss": last_loss,
        "lr": lr,
        "basis": "completion CE on the chosen continuations of the same pairs; "
                 "budget is the preference forward-token count (prompt counted "
                 "twice there, once here)",
    }


def run_preference(model, pairs, steps: int, lr: float = 1e-4,
                   beta: float = 0.1, device: str | None = None,
                   normalization: str = REGISTERED_NORMALIZATION) -> dict:
    """A CPU/GPU-runnable preference loop over cached pairs."""
    if not pairs:
        raise ValueError("the preference loop needs at least one pair")
    trainable = [parameter for parameter in model.parameters()
                 if parameter.requires_grad]
    if not trainable:
        raise ValueError("the preference loop needs trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    losses = []
    for step in range(steps):
        batch = [pairs[step % len(pairs)]]
        optimizer.zero_grad(set_to_none=True)
        loss, detail = preference_batch_loss(model, batch, beta, device,
                                             normalization)
        loss.backward()
        optimizer.step()
        losses.append(detail["mean_loss"])
    return {
        "kind": "preference",
        "steps": steps,
        "beta": beta,
        "lr": lr,
        "normalization": normalization,
        "first_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "losses": losses,
    }


# ---------------------------------------------------------------------------
# realized-update telemetry


def grad_norm(parameters) -> float:
    """L2 norm of the parameter gradients (zero if there is no gradient)."""
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().double().pow(2).sum())
    return math.sqrt(total)


def parameter_delta_l2(parameters, snapshot) -> float:
    """L2 norm of ``parameters - snapshot``, accumulated in float64."""
    if len(snapshot) != len(list(parameters)):
        raise ValueError("the snapshot does not match the parameter list")
    total = 0.0
    for parameter, before in zip(parameters, snapshot):
        delta = (parameter.detach().double() - before.double())
        total += float(delta.pow(2).sum())
    return math.sqrt(total)


class UpdateTelemetry:
    """Per-step evidence that the difficulty weight changes realized updates.

    A loss-value change is not evidence: AdamW's per-parameter normalisation
    cancels a constant rescale of the gradient, and clipping can cancel the
    rest. This recorder captures, for every optimizer step, the pre-clip and
    post-clip gradient norms and the L2 norm of the parameter delta the step
    actually produced, tagged with the batch's weight. A weighted run whose
    ``update_l2`` and parameter trajectory match an unweighted run on the same
    pairs is inert whatever its loss curve says.

    It copies the trainable parameters once per step, so it is opt-in: the
    caller passes a trace path only when the evidence is wanted. The trace has
    no timestamps, so the same seed reproduces it byte for byte.
    """

    SCHEMA = "selvedge/s70-update-trace/v1"

    def __init__(self, *, grad_clip: float | None = None,
                 lr: float | None = None) -> None:
        self.grad_clip = grad_clip
        self.lr = lr
        self.rows: list[dict] = []
        self._snapshot: list = []
        self._current: list = []

    def begin(self, parameters) -> None:
        self._snapshot = [parameter.detach().clone() for parameter in parameters]

    def row(self, *, step: int, arm: str, problem_ids, weights, loss: float,
            mean_weight: float | None = None,
            weight_normalizer: float | None = None,
            pre_clip_grad_norm: float = 0.0,
            post_clip_grad_norm: float = 0.0) -> dict:
        if not self._current or not self._snapshot:
            raise RuntimeError("call bind() before recording a row")
        update = parameter_delta_l2(self._current, self._snapshot)
        row = {
            "step": int(step),
            "arm": arm,
            "problem_ids": list(problem_ids),
            "weights": [float(weight) for weight in weights],
            "weight": (float(weights[0]) if len(weights) == 1 else None),
            "weight_normalized": (
                float(weights[0]) / float(weight_normalizer)
                if len(weights) == 1 and weight_normalizer else None),
            "loss": float(loss),
            "pre_clip_grad_norm": float(pre_clip_grad_norm),
            "post_clip_grad_norm": float(post_clip_grad_norm),
            "update_l2": update,
            "clipping_active": bool(
                self.grad_clip and pre_clip_grad_norm > float(self.grad_clip)),
        }
        self.rows.append(row)
        self.begin(self._current)
        return row

    def bind(self, parameters) -> None:
        """Fix the parameter list the rows are measured against."""
        self._current = list(parameters)
        self.begin(self._current)

    def summary(self) -> dict:
        if not self.rows:
            return {"steps": 0, "realized_mean_weight": None,
                    "clipping_active_steps": 0, "mean_update_l2": None}
        weights = [float(row["weights"][0]) for row in self.rows
                   if len(row["weights"]) == 1]
        updates = [row["update_l2"] for row in self.rows]
        return {
            "steps": len(self.rows),
            "realized_mean_weight": (sum(weights) / len(weights)
                                     if weights else None),
            "realized_weight_min": min(weights) if weights else None,
            "realized_weight_max": max(weights) if weights else None,
            "clipping_active_steps": sum(1 for row in self.rows
                                         if row["clipping_active"]),
            "mean_update_l2": sum(updates) / len(updates),
            "unique_pairs": len({tuple(row["problem_ids"])
                                 for row in self.rows}),
        }

    def payload(self) -> dict:
        return {
            "schema": self.SCHEMA,
            "kind": "realized-update-trace",
            "grad_clip": self.grad_clip,
            "lr": self.lr,
            "rows": self.rows,
            "summary": self.summary(),
            "basis": (
                "pre_clip_grad_norm is clip_grad_norm_'s return value; "
                "post_clip_grad_norm is measured after clipping; update_l2 is "
                "the L2 norm of the parameter delta the optimizer step "
                "actually applied; clipping_active compares the pre-clip norm "
                "with grad_clip"),
        }

    def write(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".",
                    exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(self.payload(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, path)
        return path


__all__ = [
    "REGISTERED_LORA_RANK", "REGISTERED_LORA_ALPHA",
    "REGISTERED_ACTIVE_UNION", "REGISTERED_NORMALIZATION",
    "MERGE_ATOL", "MERGE_RTOL", "NORMALIZATIONS", "PAIR_ORDERS",
    "PreferencePair",
    "assert_shared_prefix", "make_pair", "difficulty_weight",
    "corpus_mean_weight", "weighted_permutation", "pair_stream",
    "recovery_weight", "recovery_pair", "build_recovery_arm",
    "normalise_logprob",
    "sequence_logprob", "continuation_logprobs", "cache_reference_logprobs",
    "reference_logprob", "preference_loss", "pair_loss",
    "preference_batch_loss", "LoRALinear", "default_linear_target",
    "install_lora", "lora_parameters", "adapter_element_count", "merge_lora",
    "active_union_after_merge", "unique_active_elements", "sft_control_loss",
    "preference_processed_tokens", "sft_processed_tokens",
    "run_matched_sft_control", "run_preference",
    "UpdateTelemetry", "grad_norm", "parameter_delta_l2",
]
