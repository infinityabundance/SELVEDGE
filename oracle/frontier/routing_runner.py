#!/usr/bin/env python3
"""S52: P19.9 counterfactual pre-read router runner (CPU core).

This module is the executable half of P19.9. `oracle/frontier/utility_routing.py`
defines the router and the label contract; this module runs the three stages the
S50 queue entry requires:

1. **paired rollouts.** For one registered prompt, generate `K` completions with
   the base model and `K` with the same model under the external-memory read
   (temperature and top-p registered), verify every completion separately with
   the S45 exact-answer extractor, and form the per-prompt

       Delta_outcome = P(correct | memory) - P(correct | base)

   Plus the per-prompt `Delta_CE = CE_base - CE_memory` and, when the generator
   reports it, the per-token CE difference.
2. **pre-read training.** Features come from the causal hidden state and the
   registered metadata fractions through `PreReadUtilityRouter.features`, which
   is the allow-list. Targets are the paired outcome delta (primary) and the
   counterfactual CE delta (secondary). Prompts are partitioned `fit` / `calib`
   / `holdout`; no prompt text may appear in two splits.
3. **masked evaluation.** Selection is audited at runtime: unselected tokens
   receive exactly zero contribution and the store is called with exactly the
   selected indices. Read cost is the measured gather ledger plus
   `physical_read_accounting` (unique rows, tier bytes), never the
   read-magnitude surrogate.

The GPU entry point is `scripts/s52_p19_9_router.py`; this module imports no
model framework, so every function below is CPU-testable with fake tensors,
fake generators and a fake store.

The routed read itself is **conv-free by definition**: the registered corrective
memory runs an 8-channel causal convolution over retrieved values, so a token's
read depends on the reads of the preceding `kernel-1` tokens. A routed token may
have unselected predecessors, whose values were never fetched, so the same
convolution cannot be evaluated exactly from a sparse gather. The routed
mechanism therefore uses the same 8-row table gather, key gate and value
projection without that convolution, and the paired memory side is the *same*
operator with every token selected. The comparison is internally consistent;
it is not reported as a routed version of the corrective arm.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import torch

from . import rft
from . import memory as memory_module
from . import utility_routing as routing

PROMPT_SET_SCHEMA = "selvedge/p19.9-prompt-set/v1"
ROLLOUT_RECEIPT_SCHEMA = "selvedge/p19.9-rollouts/v1"
ROUTER_RECEIPT_SCHEMA = "selvedge/p19.9-router/v1"
SCREEN_RECEIPT_SCHEMA = "selvedge/p19.9-screen/v1"

SPLITS = ("fit", "calib", "holdout")
MODES = ("greedy", "sampled")
DATASETS = ("gsm8k", "math")
SOURCES = ("counterfactual-outcome", "counterfactual-token-loss")
ROUTER_ARMS = ("utility-outcome", "utility-ce", "no-memory-gate",
               "matched-dense")
ATTRIBUTION_CONTROL = "gated-dense"
CORRECT_VERDICTS = ("correct", "wrong", "ambiguous", "unparseable")

#: Registered rollout spec. The prompt-set file pins the same numbers and the
#: loader refuses a mismatch: a prompt set produced for one sampling plan cannot
#: be reused with another without a new registration.
DEFAULT_SPEC = {
    "k": 4,
    "temperature": 0.8,
    "top_p": 0.95,
    "mode": "sampled",
    "max_new_tokens": 384,
    "seed": 0,
}
#: Registered router training spec.
DEFAULT_TRAINING = {
    "lr": 1e-3,
    "epochs": 60,
    "batch": 256,
    "weight_decay": 0.0,
    "threshold": 0.5,
    "seed": 0,
}
#: Registered read-cost unit. One store request is the 8 addressed table rows
#: for one token; the table is fp32 in the registered memory geometry.
VALUE_DIM = 256
READS_PER_TOKEN = 8
TABLE_BYTES_PER_ROW = VALUE_DIM * 4
ROW_BYTES_BY_TIER = {"table": READS_PER_TOKEN * TABLE_BYTES_PER_ROW}


# ---------------------------------------------------------------------------
# rollout specification and prompt set


@dataclass(frozen=True)
class RolloutSpec:
    k: int
    temperature: float
    top_p: float
    mode: str
    max_new_tokens: int
    seed: int = 0

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if not isinstance(self.k, int) or self.k < 1:
            raise ValueError("k must be a positive int")
        if not isinstance(self.max_new_tokens, int) or self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive int")
        if self.mode == "greedy":
            if float(self.temperature) != 0.0 or float(self.top_p) != 1.0:
                raise ValueError(
                    "greedy mode is registered at temperature 0.0 and top_p 1.0")
        else:
            if not 0.0 < float(self.temperature) <= 2.0:
                raise ValueError("sampled temperature must be in (0, 2]")
            if not 0.0 < float(self.top_p) <= 1.0:
                raise ValueError("sampled top_p must be in (0, 1]")

    def mapping(self) -> dict:
        return {
            "k": self.k, "temperature": float(self.temperature),
            "top_p": float(self.top_p), "mode": self.mode,
            "max_new_tokens": int(self.max_new_tokens),
            "seed": int(self.seed),
        }


@dataclass(frozen=True)
class PromptItem:
    prompt_id: str
    dataset: str
    prompt: str
    answer: str
    split: str


def _normalise_prompt(text: str) -> str:
    return " ".join(str(text).split()).lower()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prompt_set(path: str, spec: RolloutSpec) -> tuple[list[PromptItem], dict]:
    """Load and validate a registered calibration prompt set.

    Refuses: a schema mismatch, a sampling block that differs from `spec`, a
    duplicate prompt id, an unknown split or dataset, an empty prompt or answer,
    and the same prompt text in two splits. The last check is on the normalised
    text, so two ids for the same question cannot leak across the split.
    """
    if not os.path.isfile(path):
        raise SystemExit(f"p19.9-runner: missing prompt set {path}")
    payload = json.load(open(path, encoding="utf-8"))
    if payload.get("schema") != PROMPT_SET_SCHEMA:
        raise SystemExit(
            f"p19.9-runner: {path} schema {payload.get('schema')!r} != "
            f"{PROMPT_SET_SCHEMA!r}")
    registered = payload.get("sampling") or {}
    expected = spec.mapping()
    if registered != expected:
        raise SystemExit(
            "p19.9-runner: prompt set sampling does not match the runner spec: "
            f"registered {registered} != runner {expected}")
    items: list[PromptItem] = []
    seen_ids: set[str] = set()
    text_split: dict[str, str] = {}
    counts = {split: 0 for split in SPLITS}
    for index, row in enumerate(payload.get("prompts") or []):
        where = f"prompts[{index}]"
        prompt_id = row.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise SystemExit(f"p19.9-runner: {where} needs a prompt_id")
        if prompt_id in seen_ids:
            raise SystemExit(f"p19.9-runner: duplicate prompt_id {prompt_id!r}")
        seen_ids.add(prompt_id)
        split = row.get("split")
        if split not in SPLITS:
            raise SystemExit(f"p19.9-runner: {where} split {split!r} not in {SPLITS}")
        dataset = row.get("dataset")
        if dataset not in DATASETS:
            raise SystemExit(
                f"p19.9-runner: {where} dataset {dataset!r} not in {DATASETS}")
        prompt = row.get("prompt")
        answer = row.get("answer")
        if not isinstance(prompt, str) or not prompt.strip():
            raise SystemExit(f"p19.9-runner: {where} has an empty prompt")
        if not isinstance(answer, str) or not answer.strip():
            raise SystemExit(f"p19.9-runner: {where} has an empty answer")
        key = _normalise_prompt(prompt)
        other = text_split.get(key)
        if other is not None and other != split:
            raise SystemExit(
                f"p19.9-runner: prompt text {prompt_id!r} appears in both "
                f"{other!r} and {split!r} splits")
        text_split[key] = split
        counts[split] += 1
        items.append(PromptItem(prompt_id=prompt_id, dataset=dataset,
                                prompt=prompt, answer=answer, split=split))
    if not items:
        raise SystemExit("p19.9-runner: prompt set is empty")
    for split, count in counts.items():
        if count < 1:
            raise SystemExit(f"p19.9-runner: no prompts in split {split!r}")
    meta = {
        "path": path, "sha256": sha256_file(path),
        "schema": payload.get("schema"),
        "source": payload.get("source"),
        "sampling": registered,
        "counts": counts,
        "items": len(items),
    }
    return items, meta


def stable_seed(seed: int, prompt_id: str, sample_index: int) -> int:
    """Deterministic 31-bit sample seed from (run seed, prompt, sample)."""
    digest = hashlib.sha256(
        f"{int(seed)}:{prompt_id}:{int(sample_index)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def sampling_plan(spec: RolloutSpec, prompt_id: str) -> list[dict]:
    """The K draws for one prompt, in a fixed order.

    Greedy draws are repeated `K` times so the paired probability is still a
    frequency and determinism is checkable: every repeat must decode the same
    completion. Sampled draws share no seed across prompts or samples.
    """
    plan = []
    for index in range(spec.k):
        plan.append({
            "sample_index": index,
            "seed": stable_seed(spec.seed, prompt_id, index),
            "greedy": spec.mode == "greedy",
            "temperature": 0.0 if spec.mode == "greedy" else float(spec.temperature),
            "top_p": 1.0 if spec.mode == "greedy" else float(spec.top_p),
        })
    return plan


# ---------------------------------------------------------------------------
# paired rollouts


def run_paired_rollouts(items: Sequence[PromptItem],
                        generate: Callable[[str, PromptItem, dict], Mapping],
                        verify: Callable[[PromptItem, str], Mapping],
                        spec: RolloutSpec) -> list[dict]:
    """Generate and verify paired base / base+memory rollouts.

    `generate(condition, item, sample)` must return at least
    `{"completion": str}`; it may also return `ce` (mean next-token CE of the
    completion) and `token_ce` (per-token CE list). `verify(item, completion)`
    must return `{"verdict": one of CORRECT_VERDICTS}`. Each side is verified
    separately: the two calls are distinct even when the completions match.
    """
    results = []
    for item in items:
        plan = sampling_plan(spec, item.prompt_id)
        pairs = []
        base_ces: list[float] = []
        memory_ces: list[float] = []
        token_deltas: list[list[float]] = []
        completions = {"base": [], "memory": []}
        verdicts = {"base": [], "memory": []}
        for sample in plan:
            sides = {}
            generated_sides = {}
            for condition in ("base", "memory"):
                generated = generate(condition, item, sample)
                generated_sides[condition] = generated
                completion = generated.get("completion")
                if not isinstance(completion, str):
                    raise ValueError(
                        f"generator returned no completion for "
                        f"{item.prompt_id!r}/{condition}")
                verdict = verify(item, completion)
                value = verdict.get("verdict")
                if value not in CORRECT_VERDICTS:
                    raise ValueError(
                        f"verifier returned verdict {value!r} for "
                        f"{item.prompt_id!r}/{condition}")
                sides[condition] = {
                    "verified": value == "correct",
                    "verdict": value,
                    "completion": completion,
                }
                completions[condition].append(completion)
                verdicts[condition].append(value)
                ce = generated.get("ce")
                if ce is None:
                    continue
                if condition == "base":
                    base_ces.append(float(ce))
                else:
                    memory_ces.append(float(ce))
            base_token = generated_sides["base"].get("token_ce")
            memory_token = generated_sides["memory"].get("token_ce")
            if base_token is not None and memory_token is not None:
                if len(base_token) != len(memory_token):
                    raise ValueError(
                        f"prompt {item.prompt_id!r}: base and memory token_ce "
                        "lengths differ")
                token_deltas.append([float(b) - float(m) for b, m
                                     in zip(base_token, memory_token)])
            if spec.mode == "greedy":
                for condition in ("base", "memory"):
                    if len(set(completions[condition])) != 1:
                        raise ValueError(
                            "greedy rollouts are not deterministic for "
                            f"{item.prompt_id!r}/{condition}")
            pairs.append({
                "rollout_id": f"{item.prompt_id}:{sample['sample_index']}",
                "base": {"verified": sides["base"]["verified"]},
                "memory": {"verified": sides["memory"]["verified"]},
            })
        outcome = routing.outcome_delta(pairs)
        ce_delta = None
        if base_ces and memory_ces and len(base_ces) == len(memory_ces):
            ce_delta = (sum(base_ces) / len(base_ces)
                        - sum(memory_ces) / len(memory_ces))
        results.append({
            "prompt_id": item.prompt_id,
            "split": item.split,
            "dataset": item.dataset,
            "prompt": item.prompt,
            "answer": item.answer,
            "sampling": spec.mapping(),
            "paired_rollouts": pairs,
            "outcome": outcome,
            "delta_outcome": outcome["delta"],
            "base_ce": (sum(base_ces) / len(base_ces)) if base_ces else None,
            "memory_ce": (sum(memory_ces) / len(memory_ces)) if memory_ces else None,
            "delta_ce": ce_delta,
            "token_ce_deltas": token_deltas,
            "completions": completions,
            "verdicts": verdicts,
            "verification": "each side verified separately with frontier.rft.verify",
        })
    return results


def rollout_receipt(results: Sequence[dict], spec: RolloutSpec,
                    prompt_meta: Mapping, extra: Mapping | None = None) -> dict:
    """One tracked receipt for a rollout stage (completions included)."""
    payload = {
        "schema": ROLLOUT_RECEIPT_SCHEMA,
        "sampling": spec.mapping(),
        "prompt_set": dict(prompt_meta),
        "prompts": list(results),
        "prompt_count": len(results),
        "split_counts": {
            split: sum(1 for row in results if row["split"] == split)
            for split in SPLITS},
        "verification": "frontier.rft.verify per completion",
        "delta_outcome": "P(correct|memory) - P(correct|base)",
        "delta_ce": "CE_base - CE_memory on the completion, when reported",
    }
    if extra:
        payload.update(extra)
    return payload


def require_rollouts(path: str) -> dict:
    """Refusal path: a runner without paired rollouts must not proceed."""
    if not path:
        raise SystemExit(
            "p19.9-runner: rollouts are required; pass --rollouts from the "
            "paired-rollout stage")
    if not os.path.isfile(path):
        raise SystemExit(f"p19.9-runner: missing rollout receipt {path}")
    payload = json.load(open(path, encoding="utf-8"))
    if payload.get("schema") != ROLLOUT_RECEIPT_SCHEMA:
        raise SystemExit(
            f"p19.9-runner: {path} is not a {ROLLOUT_RECEIPT_SCHEMA} receipt")
    rows = payload.get("prompts") or []
    if not rows:
        raise SystemExit("p19.9-runner: rollout receipt has no prompts")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        where = f"prompts[{index}]"
        prompt_id = row.get("prompt_id")
        if not prompt_id:
            raise SystemExit(f"p19.9-runner: {where} has no prompt_id")
        if prompt_id in seen:
            raise SystemExit(f"p19.9-runner: duplicate prompt {prompt_id!r}")
        seen.add(prompt_id)
        if row.get("split") not in SPLITS:
            raise SystemExit(f"p19.9-runner: {where} has no registered split")
        pairs = row.get("paired_rollouts")
        if not pairs:
            raise SystemExit(
                f"p19.9-runner: {where} has no paired rollouts; the outcome "
                "route cannot be trained without them")
        for pair_index, pair in enumerate(pairs):
            for side in ("base", "memory"):
                value = (pair.get(side) or {}).get("verified")
                if not isinstance(value, bool):
                    raise SystemExit(
                        f"p19.9-runner: {where}.paired_rollouts[{pair_index}]."
                        f"{side}.verified is not a bool")
    return payload


# ---------------------------------------------------------------------------
# labels and features


def label_records(results: Sequence[dict], source: str) -> list[dict]:
    """Convert rollout results into `labels_from_records` inputs.

    The outcome source carries the paired rollouts. The token-loss source carries
    the mean CE of each side; it refuses a prompt whose generator did not report
    CE, because a zero fill would be a fabricated label.
    """
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}")
    records = []
    for row in results:
        record = {
            "prompt_id": row["prompt_id"],
            "split": row["split"],
            "features": {},
            "token_count": 1,
        }
        if source == "counterfactual-outcome":
            record["paired_rollouts"] = row["paired_rollouts"]
        else:
            if row.get("base_ce") is None or row.get("memory_ce") is None:
                raise ValueError(
                    f"prompt {row['prompt_id']!r} has no CE on one or both "
                    "sides; the token-loss route cannot be trained on it")
            record["base_loss"] = float(row["base_ce"])
            record["memory_loss"] = float(row["memory_ce"])
        records.append(record)
    return records


def labels_by_split(results: Sequence[dict], source: str) -> dict[str, list[dict]]:
    """Validate every split through the registered label function.

    `labels_from_records` knows `fit` and `holdout`; the calibration split is
    validated under the same rules and its label is then relabelled `calib`.
    The prompt-id sets of the three splits must be pairwise disjoint.
    """
    records = label_records(results, source)
    grouped = {split: [row for row in records if row["split"] == split]
               for split in SPLITS}
    labels = {}
    for split in ("fit", "holdout"):
        labels[split] = routing.labels_from_records(grouped[split], source)
    # `labels_from_records` registers two splits; calibration is validated under
    # the same rules with the `fit` label and then relabelled.
    calib_records = [dict(row, split="fit") for row in grouped["calib"]]
    labels["calib"] = [
        dict(label, split="calib")
        for label in routing.labels_from_records(calib_records, source)]
    ids = {split: {label["prompt_id"] for label in rows}
           for split, rows in labels.items()}
    for left, right in (("fit", "calib"), ("fit", "holdout"),
                        ("calib", "holdout")):
        overlap = ids[left] & ids[right]
        if overlap:
            raise ValueError(
                f"prompt ids appear in both {left} and {right}: {sorted(overlap)}")
    return labels


def metadata_for_prompt(prompt_id: str, tokens: int,
                        prefix_tokens: int = 0) -> dict[str, torch.Tensor]:
    """Registered compact metadata for the router at these positions.

    `position_fraction` is the absolute position in the sequence divided by the
    context length, and `budget_fraction` is 1.0; both are in [0, 1] by
    construction and neither depends on the memory values.
    """
    if tokens < 1:
        raise ValueError("tokens must be positive")
    positions = torch.arange(prefix_tokens, prefix_tokens + tokens,
                             dtype=torch.float32)
    denominator = float(max(prefix_tokens + tokens, 1))
    fraction = (positions / denominator).clamp(0.0, 1.0)
    return {
        "position_fraction": fraction,
        "budget_fraction": torch.ones(tokens, dtype=torch.float32),
    }


class RoutedMemoryArm(torch.nn.Module):
    """External-memory read behind the pre-read router, conv-free by design.

    The address, key gate and value projection are the registered memory
    geometry. The only change is that the 8-channel causal convolution over
    retrieved values is absent: a token's convolution would need the rows of the
    preceding `kernel-1` tokens, and a routed token may have predecessors that
    were never fetched, so the convolution cannot be evaluated exactly from a
    sparse gather. With `router=None` this arm is the dense condition (every
    token read); with a router it is the routed condition (only selected
    tokens read, zero contribution elsewhere).
    """

    def __init__(self, core, router=None, threshold: float = 0.5,
                 cost: float = 0.0, route: bool = True, zero_read: bool = False,
                 ledger: "StoreLedger | None" = None, site_index: int = 0):
        super().__init__()
        self.core = core
        if router is not None and not isinstance(
                router, routing.TwoStageUtilityRouter):
            wrapped = routing.TwoStageUtilityRouter(
                hidden=int(core.hidden), tau=float(router.tau),
                use_post_read=False)
            wrapped.pre.load_state_dict(router.state_dict())
            router = wrapped
        self.router = router
        self.threshold = float(threshold)
        self.cost = float(cost)
        self.route = bool(route)
        self.zero_read = bool(zero_read)
        self.site_index = int(site_index)
        self.ledger = ledger if ledger is not None else StoreLedger()
        self.table_bytes = int(core.value_dim) * 4
        self.last_execution = None
        # install_memory reads these two attributes off the arm
        self.context = core.context
        self.sites = core.sites

    def reset_state(self) -> None:
        self.core.reset_state()
        self.last_execution = None

    def _token_read(self, hidden_states: torch.Tensor,
                    indices: torch.Tensor) -> torch.Tensor:
        """Gather only the addressed rows of the selected tokens."""
        batch, length, hidden_dim = hidden_states.shape
        tokens = self.core.context.prefix()
        if tokens.shape[-1] < length:
            raise RuntimeError(
                f"token context has {tokens.shape[-1]} positions, layer needs "
                f"{length}")
        addresses = memory_module.support(
            tokens, self.core.sites[self.site_index], self.core.rows)[:, -length:]
        flat_addresses = addresses.reshape(-1, memory_module.HEADS)
        selected = flat_addresses[indices]
        table = self.core.tables[self.site_index]
        rows = self.ledger.gather(table, selected.reshape(-1))
        values = rows.reshape(selected.shape[0], memory_module.HEADS,
                              self.core.value_dim).float()
        flat_hidden = hidden_states.reshape(-1, hidden_dim)[indices].float()
        key = self.core.key_proj[self.site_index](flat_hidden)
        score = (values * key[:, None, :]).sum(-1) / (self.core.value_dim ** 0.5)
        gate = torch.sigmoid(score)
        projected = self.core.value_proj[self.site_index](values)
        return (gate[..., None] * projected).sum(dim=1)

    def forward(self, hidden_states: torch.Tensor, site_index: int = 0
                ) -> torch.Tensor:
        batch, length, hidden_dim = hidden_states.shape
        total = int(self.core.context.prefix().shape[-1])
        metadata = metadata_for_prompt("", length,
                                       prefix_tokens=max(total - length, 0))
        if self.router is None:
            indices = torch.arange(batch * length, device=hidden_states.device)
            read = self._token_read(hidden_states, indices)
            return read.reshape(batch, length, hidden_dim).to(hidden_states.dtype)
        if self.zero_read:
            def read_path(indices):
                return hidden_states.new_zeros((int(indices.numel()), hidden_dim))
        else:
            def read_path(indices):
                return self._token_read(hidden_states, indices)
        execution = self.router.execute(
            hidden_states, metadata, read_path, cost=self.cost,
            threshold=self.threshold,
            row_bytes_by_tier={"table": self.table_bytes})
        self.last_execution = execution
        return execution.contribution.to(hidden_states.dtype)


@dataclass
class RouterExample:
    prompt_id: str
    split: str
    hidden: torch.Tensor          # [T, H] causal hidden state
    metadata: dict                # registered fractions
    features: torch.Tensor        # [T, F], produced by the allow-list
    targets: torch.Tensor         # [T]
    weights: torch.Tensor         # [T], sums to 1 per prompt
    row_ids: torch.Tensor | None = None


def assemble_examples(router: routing.PreReadUtilityRouter,
                      hidden_by_prompt: Mapping[str, torch.Tensor],
                      metadata_by_prompt: Mapping[str, Mapping],
                      labels: Mapping[str, Sequence[dict]],
                      source: str) -> list[RouterExample]:
    """Build router training rows from causal hidden states and labels.

    Features are produced only by `router.features`, which enforces the
    allow-list: `r_t`, `read_norm`, `gate_t`, the target, the answer and every
    loss key raise. The outcome label is prompt-level and is broadcast to the
    prompt's tokens with weight `1/T`, so prompts weigh equally regardless of
    length. The token-loss source may carry `token_labels`; when it does, they
    are used per token and must match the token count.
    """
    examples: list[RouterExample] = []
    for split, rows in labels.items():
        for label in rows:
            prompt_id = label["prompt_id"]
            if prompt_id not in hidden_by_prompt:
                raise ValueError(
                    f"no causal hidden state for prompt {prompt_id!r}")
            hidden = hidden_by_prompt[prompt_id]
            metadata = metadata_by_prompt[prompt_id]
            features = router.features(hidden, metadata)
            tokens = int(features.shape[0])
            if tokens < 1:
                raise ValueError(f"prompt {prompt_id!r} has no tokens")
            token_labels = label.get("token_labels")
            if token_labels is not None:
                if len(token_labels) != tokens:
                    raise ValueError(
                        f"prompt {prompt_id!r}: {len(token_labels)} token "
                        f"labels for {tokens} tokens")
                targets = torch.as_tensor(token_labels, dtype=torch.float32)
            else:
                targets = torch.full((tokens,), float(label["label"]),
                                     dtype=torch.float32)
            if not bool(torch.isfinite(targets).all()):
                raise ValueError(f"prompt {prompt_id!r} has non-finite targets")
            examples.append(RouterExample(
                prompt_id=prompt_id, split=split, hidden=hidden,
                metadata=metadata, features=features, targets=targets,
                weights=torch.full((tokens,), 1.0 / tokens),
            ))
    if not examples:
        raise ValueError(f"no examples assembled for source {source!r}")
    return examples


# ---------------------------------------------------------------------------
# router training


def build_router(hidden: int, tau: float = routing.DEFAULT_TAU,
                 seed: int = 0) -> routing.PreReadUtilityRouter:
    """Construct a router with a seeded first-layer initialisation.

    The output layer is zero at init by construction; the hidden layer is the
    only random part, so seeding it here is what makes two runs with the same
    seed, data and batch order reproduce the same parameter hash.
    """
    torch.manual_seed(int(seed))
    return routing.PreReadUtilityRouter(hidden=int(hidden), tau=float(tau))


def _parameter_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().to(torch.float32).cpu().numpy().tobytes())
    return digest.hexdigest()


def train_router(router: routing.PreReadUtilityRouter,
                 fit: Sequence[RouterExample],
                 calib: Sequence[RouterExample],
                 *, lr: float = 1e-3, epochs: int = 60, batch: int = 256,
                 weight_decay: float = 0.0, seed: int = 0,
                 threshold: float = 0.5) -> dict:
    """Deterministic weighted-MSE training; the epoch is chosen on calibration.

    The fit split trains, the calibration split chooses the epoch (lowest
    weighted MSE; earliest epoch wins a tie), and the holdout split is never
    touched here. The same seed reproduces the parameter hash.
    """
    if not fit or not calib:
        raise ValueError("fit and calib examples are both required")
    torch.manual_seed(int(seed))
    optimizer = torch.optim.Adam(router.parameters(), lr=float(lr),
                                 weight_decay=float(weight_decay))
    x_fit = torch.cat([example.features.detach() for example in fit], dim=0)
    y_fit = torch.cat([example.targets for example in fit], dim=0)
    w_fit = torch.cat([example.weights for example in fit], dim=0)
    x_cal = torch.cat([example.features.detach() for example in calib], dim=0)
    y_cal = torch.cat([example.targets for example in calib], dim=0)
    w_cal = torch.cat([example.weights for example in calib], dim=0)
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 17)
    history = []
    best = {"epoch": None, "loss": None, "state": None}
    for epoch in range(1, int(epochs) + 1):
        router.train()
        order = torch.randperm(x_fit.shape[0], generator=generator)
        total = 0.0
        rows = 0
        for start in range(0, x_fit.shape[0], int(batch)):
            index = order[start:start + int(batch)]
            prediction = router.predictor(x_fit[index]).squeeze(-1)
            weight = w_fit[index]
            loss = ((prediction - y_fit[index]) ** 2 * weight).sum() / weight.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * weight.sum().item()
            rows += int(weight.sum().item())
        train_loss = total / max(rows, 1e-12)
        router.eval()
        with torch.no_grad():
            prediction = router.predictor(x_cal).squeeze(-1)
            cal_loss = float((((prediction - y_cal) ** 2) * w_cal).sum()
                             / w_cal.sum())
            sign_agreement = float(
                ((prediction > 0.0) == (y_cal > 0.0)).float().mean())
            gate_open = float((torch.sigmoid(prediction / router.tau)
                               >= threshold).float().mean())
        history.append({
            "epoch": epoch, "fit_loss": train_loss, "calib_loss": cal_loss,
            "calib_sign_agreement": sign_agreement,
            "calib_gate_open_fraction": gate_open,
        })
        if best["loss"] is None or cal_loss < best["loss"]:
            best = {
                "epoch": epoch, "loss": cal_loss,
                "state": {name: value.detach().clone()
                          for name, value in router.state_dict().items()},
            }
    router.load_state_dict(best["state"])
    router.eval()
    with torch.no_grad():
        prediction = router.predictor(x_cal).squeeze(-1)
        chosen = float((((prediction - y_cal) ** 2) * w_cal).sum()
                       / w_cal.sum())
    return {
        "epochs": int(epochs), "lr": float(lr), "batch": int(batch),
        "weight_decay": float(weight_decay), "seed": int(seed),
        "threshold": float(threshold),
        "fit_examples": len(fit), "calib_examples": len(calib),
        "fit_tokens": int(x_fit.shape[0]), "calib_tokens": int(x_cal.shape[0]),
        "chosen_epoch": int(best["epoch"]),
        "calib_loss_at_chosen_epoch": chosen,
        "parameter_hash": _parameter_hash(router),
        "history": history,
        "deterministic": "same seed, same data and same torch build reproduce "
                         "the parameter hash; asserted by the test suite",
    }


def save_router(router: routing.PreReadUtilityRouter, path: str,
                report: Mapping) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save({"state_dict": router.state_dict(),
                "hidden": router.hidden,
                "tau": router.tau,
                "training": dict(report)}, path)


def load_router(path: str) -> tuple[routing.PreReadUtilityRouter, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    router = routing.PreReadUtilityRouter(hidden=int(payload["hidden"]),
                                          tau=float(payload.get("tau",
                                                                routing.DEFAULT_TAU)))
    router.load_state_dict(payload["state_dict"])
    router.eval()
    return router, payload.get("training") or {}


# ---------------------------------------------------------------------------
# read cost and masked selection


@dataclass
class StoreLedger:
    """What the sparse read path actually gathered.

    A `GatherStore` is the only object allowed to touch the table. It records
    each call's requests, the unique table rows gathered and the bytes those
    rows cost; the runner reports this beside `physical_read_accounting`.
    """

    row_bytes: int = TABLE_BYTES_PER_ROW
    calls: int = 0
    requested: int = 0
    unique_rows: int = 0
    unique_row_ids: set = field(default_factory=set)
    cache_hits: int = 0

    def gather(self, table: torch.Tensor, addresses: torch.Tensor,
               cache: bool = False) -> torch.Tensor:
        if addresses.ndim != 1:
            raise ValueError("addresses must be 1-D")
        if addresses.numel() == 0:
            # A rejected token requests nothing; an empty gather is not a call.
            return table.new_zeros((0,) + tuple(table.shape[1:]))
        if int(addresses.min()) < 0 or int(addresses.max()) >= table.shape[0]:
            raise ValueError("address outside the table")
        self.calls += 1
        self.requested += int(addresses.numel())
        self.unique_row_ids.update(int(v) for v in addresses.tolist())
        self.unique_rows = len(self.unique_row_ids)
        if cache:
            self.cache_hits += int(addresses.numel())
        return table[addresses]

    def report(self) -> dict:
        return {
            "calls": self.calls,
            "requested_rows": self.requested,
            "unique_rows": self.unique_rows,
            "bytes": self.unique_rows * self.row_bytes,
            "row_bytes": self.row_bytes,
            "cache_hits": self.cache_hits,
            "basis": "measured gather ledger: unique addressed table rows",
        }


def cost_from_accounting(accounting: routing.ReadAccounting) -> dict:
    """Read cost from `physical_read_accounting`, never from read magnitude.

    There is no price per byte here. The cost is reported in physical units
    (unique rows and bytes per token) and the gate is swept over thresholds, so
    no nats-per-byte exchange rate is invented.
    """
    if not isinstance(accounting, routing.ReadAccounting):
        raise TypeError(
            "cost must come from physical_read_accounting; the read-magnitude "
            "surrogate measured_cost(read) is not accepted")
    tokens = max(int(accounting.tokens), 1)
    total_bytes = sum(int(value) for value in accounting.bytes_by_tier.values())
    return {
        "source": "physical_read_accounting",
        "tokens": tokens,
        "requests": int(accounting.requests),
        "unique_rows_requested": int(accounting.unique_rows_requested),
        "unique_rows_per_token": accounting.unique_rows_requested / tokens,
        "bytes_by_tier": dict(accounting.bytes_by_tier),
        "bytes_total": total_bytes,
        "bytes_per_token": total_bytes / tokens,
        "cache_hits": int(accounting.cache_hits),
        "surrogate_used": False,
        "basis": accounting.basis,
    }


def selection_report(router: routing.PreReadUtilityRouter,
                     examples: Sequence[RouterExample], *,
                     threshold: float = 0.5, cost: float = 0.0,
                     row_bytes_by_tier: Mapping[str, int] | None = None,
                     audit_mask: bool = True) -> dict:
    """Gate statistics, measured read cost and the mask-enforcement audit.

    For every example the router selects from hidden state + metadata only. The
    selected rows are counted by `physical_read_accounting`; tokens outside the
    mask contribute no request. When `audit_mask` is set the contribution is
    evaluated with a non-zero synthetic read and its absolute maximum outside
    the mask must be exactly zero, which catches any future regression that
    turns the boolean mask into a soft multiplier.
    """
    total_tokens = 0
    selected_tokens = 0
    predicted_all = []
    gate_all = []
    rows_selected = 0
    bytes_by_tier: dict[str, int] = {}
    cache_hits = 0
    for example in examples:
        mask, predicted, gate = router.select(
            example.hidden, example.metadata, cost=cost, threshold=threshold)
        total_tokens += int(mask.numel())
        selected_tokens += int(mask.sum())
        predicted_all.append(predicted.detach().reshape(-1))
        gate_all.append(gate.detach().reshape(-1))
        accounting = routing.physical_read_accounting(
            mask, row_bytes_by_tier=row_bytes_by_tier)
        rows_selected += accounting.requests
        cache_hits += accounting.cache_hits
        for tier, value in accounting.bytes_by_tier.items():
            bytes_by_tier[tier] = bytes_by_tier.get(tier, 0) + int(value)
        if audit_mask:
            synthetic = torch.ones_like(example.hidden)
            audit_router = routing.TwoStageUtilityRouter(
                hidden=example.hidden.shape[-1], use_post_read=False)
            if hasattr(router, "pre"):
                audit_router.load_state_dict(router.state_dict())
            else:
                audit_router.pre.load_state_dict(router.state_dict())
            contribution = audit_router(
                example.hidden, example.metadata, synthetic, mask, cost=cost)
            outside = contribution[~mask]
            if outside.numel() and float(outside.detach().abs().max()) != 0.0:
                raise AssertionError(
                    "mask enforcement failed: an unselected token received a "
                    "non-zero contribution")
    predicted_cat = torch.cat(predicted_all) if predicted_all else torch.zeros(0)
    gate_cat = torch.cat(gate_all) if gate_all else torch.zeros(0)
    accounting = routing.ReadAccounting(
        tokens=total_tokens, requests=rows_selected,
        unique_rows_requested=rows_selected,
        bytes_by_tier=bytes_by_tier,
        cache_hits=cache_hits, cache_misses=rows_selected - cache_hits,
        basis="sum over prompts; one row per selected token "
              "(store row ids not supplied to the selection audit)")
    return {
        "prompts": len(examples),
        "tokens": total_tokens,
        "selected_tokens": selected_tokens,
        "unselected_tokens": total_tokens - selected_tokens,
        "selected_fraction": selected_tokens / max(total_tokens, 1),
        "requests_outside_mask": 0,
        "predicted_mean": float(predicted_cat.mean()) if predicted_cat.numel() else None,
        "predicted_abs_mean": (float(predicted_cat.abs().mean())
                               if predicted_cat.numel() else None),
        "gate_mean": float(gate_cat.mean()) if gate_cat.numel() else None,
        "gate_min": float(gate_cat.min()) if gate_cat.numel() else None,
        "gate_max": float(gate_cat.max()) if gate_cat.numel() else None,
        "accounting": {
            "tokens": accounting.tokens,
            "requests": accounting.requests,
            "unique_rows_requested": accounting.unique_rows_requested,
            "bytes_by_tier": accounting.bytes_by_tier,
            "cache_hits": accounting.cache_hits,
            "cache_misses": accounting.cache_misses,
            "basis": accounting.basis,
        },
        "cost": cost_from_accounting(accounting),
        "mask_enforcement": ("unselected contribution is exactly zero; the "
                             "post-read stage never sees an unfetched value"),
        "audited": bool(audit_mask),
    }


def evaluate_arms(arms: Mapping[str, Mapping], n_items: int,
                  threshold: float = 0.5) -> dict:
    """Score the registered arms and apply the win bar.

    Each arm provides `correct`, `items`, `read_requests`, `read_bytes` and
    optionally `answered`. Accuracy uses the all-items denominator; the standard
    error is binomial. An arm wins only if it beats both the base and the
    matched gated-dense control by more than the combined standard error **and**
    requests fewer bytes than the dense control.
    """
    required = {"base", "gated-dense"}
    missing = required - set(arms)
    if missing:
        raise ValueError(f"evaluate_arms needs {sorted(missing)}")
    report = {}
    for name, value in arms.items():
        items = int(value.get("items", n_items))
        if items < 1:
            raise ValueError(f"arm {name!r} has no items")
        correct = int(value["correct"])
        accuracy = correct / items
        stderr = math.sqrt(accuracy * (1.0 - accuracy) / items)
        report[name] = {
            "correct": correct,
            "items": items,
            "answered": int(value.get("answered", items)),
            "accuracy": accuracy,
            "accuracy_stderr": stderr,
            "read_requests": int(value.get("read_requests", 0)),
            "read_bytes": int(value.get("read_bytes", 0)),
            "gate_mean": value.get("gate_mean"),
        }
    base = report["base"]
    control = report["gated-dense"]
    dense = arms["gated-dense"]
    dense_bytes = int(dense.get("read_bytes", 0))
    for name, row in report.items():
        if name in ("base", "gated-dense"):
            continue
        delta_base = row["accuracy"] - base["accuracy"]
        se_base = math.hypot(row["accuracy_stderr"], base["accuracy_stderr"])
        delta_control = row["accuracy"] - control["accuracy"]
        se_control = math.hypot(row["accuracy_stderr"], control["accuracy_stderr"])
        row["delta_vs_base"] = delta_base
        row["combined_stderr_vs_base"] = se_base
        row["delta_vs_control"] = delta_control
        row["combined_stderr_vs_control"] = se_control
        row["beats_base"] = delta_base > se_base
        row["beats_control"] = delta_control > se_control
        row["fewer_bytes_than_dense"] = (row["read_bytes"] < dense_bytes
                                         if dense_bytes > 0 else
                                         row["read_requests"] <
                                         int(dense.get("read_requests", 0)))
        row["win"] = bool(row["beats_base"] and row["beats_control"]
                          and row["fewer_bytes_than_dense"])
    verdict = {
        "rule": ("an arm wins only if it beats base and the matched gated-dense "
                 "control by more than the combined binomial standard error and "
                 "requests fewer bytes than the dense control"),
        "threshold": float(threshold),
        "utility_outcome_wins": bool(report.get("utility-outcome", {}).get("win")),
        "utility_ce_wins": bool(report.get("utility-ce", {}).get("win")),
        "no_memory_gate_parity": (
            abs(report.get("no-memory-gate", {}).get("accuracy", -1.0)
                - base["accuracy"]) <= 1e-12),
    }
    if verdict["utility_outcome_wins"]:
        verdict["chosen"] = "utility-outcome"
        verdict["next_route"] = (
            "carry the pre-read outcome router to the 20M routed-memory bar "
            "beside the corrective arm at equal active union")
    elif verdict["utility_ce_wins"]:
        verdict["chosen"] = "utility-ce"
        verdict["next_route"] = (
            "the CE route wins but the outcome route does not: register a "
            "threshold/cost sweep, then a 20M CE-supervised router")
    else:
        verdict["chosen"] = None
        verdict["next_route"] = (
            "the pre-read routing hypothesis is not supported at 2M; register no "
            "20M router arm and report the post-read gate comparison instead")
    return {"arms": report, "verdict": verdict}
