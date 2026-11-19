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

import math
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
                          normalization: str = REGISTERED_NORMALIZATION
                          ) -> tuple[torch.Tensor, dict]:
    """Weighted batch loss over pairs, with per-pair diagnostics."""
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
    loss = (stacked * weight_tensor).sum() / weight_tensor.sum().clamp_min(1e-12)
    return loss, {"pairs": len(pairs),
                  "weight_sum": float(weight_tensor.sum()),
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
                 alpha: float = REGISTERED_LORA_ALPHA, target=None) -> list:
    """Replace every targeted ``nn.Linear`` in-place with ``LoRALinear``."""
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


__all__ = [
    "REGISTERED_LORA_RANK", "REGISTERED_LORA_ALPHA",
    "REGISTERED_ACTIVE_UNION", "REGISTERED_NORMALIZATION",
    "MERGE_ATOL", "MERGE_RTOL", "NORMALIZATIONS", "PreferencePair",
    "assert_shared_prefix", "make_pair", "difficulty_weight",
    "recovery_weight", "recovery_pair", "build_recovery_arm",
    "normalise_logprob",
    "sequence_logprob", "continuation_logprobs", "cache_reference_logprobs",
    "reference_logprob", "preference_loss", "pair_loss",
    "preference_batch_loss", "LoRALinear", "default_linear_target",
    "install_lora", "lora_parameters", "adapter_element_count", "merge_lora",
    "active_union_after_merge", "unique_active_elements", "sft_control_loss",
    "preference_processed_tokens", "sft_processed_tokens",
    "run_matched_sft_control", "run_preference",
]
