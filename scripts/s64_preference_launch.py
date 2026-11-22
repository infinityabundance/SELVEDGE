#!/usr/bin/env python3
"""S64: launch one reference-anchored preference arm (or its control).

The objective, the LoRA geometry and the merge path live in
`oracle/frontier/preference.py`. This script is the receipted GPU launcher:

* it installs rank-64 / alpha-32 LoRA on **every linear layer of the text
  tower except the output head**, leaves the base frozen, and asserts the
  accounting before a step is taken;
* it trains on the cached pairs to a processed-token budget, checkpointing a
  **merged full-weight** checkpoint at the registered cadence
  (0.5M / 1M / 2M / 5M) in the same schema `train.py` wrote, so
  `scripts/s62_score_milestones.py` and `scripts/s54_post_trained_eval.py`
  read it without a special path;
* it records the merge report and the active-union arithmetic. After the
  merge the adapter is not a deployed identity, so the active union is the
  base text tower, 752,393,024 elements. The float32 tiny-model merge parity
  is pinned by `oracle/tests/test_s63_preference.py` and exercised end to end
  by `oracle/tests/test_s64_preference_pipeline.py`;
* the three arms are `preference` (w = p(1-p)), `matched-sft` (the same
  positive-weight pairs, chosen continuations only, the same processed-token
  budget) and `recovery` (the p=0 teacher pairs with the matched mean positive
  weight). Weight-<=0 pairs are dropped from the normal and matched arms
  before budgeting, so both train on the identical effective problem set; the
  recovery arm is the only arm that keeps them. One arm per invocation; the
  queue runs them serially.

No teacher weights are loaded here. No reference model is loaded here. The
reference cache is consumed from the pair rows.

Usage (GPU; size `SELVEDGE_MEM` from the measured peak before launch)
---------------------------------------------------------------------
    SELVEDGE_MEM=12g scripts/court --gpu oracle -- python3 \
      scripts/s64_preference_launch.py \
        --arm preference \
        --pairs results/artifacts/stage2/s64-preference/pairs-with-reference.jsonl \
        --student-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364... \
        --budget-tokens 5000000 \
        --cadence 500000,1000000,2000000,5000000 \
        --out-dir results/stage2/runs/s64-preference \
        --vram-limit-gib 13.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "oracle"))
sys.path.insert(0, HERE)

from frontier import preference  # noqa: E402

SCHEMA = "selvedge/s64-preference-run/v1"
GIB = float(1 << 30)
ARMS = ("preference", "matched-sft", "recovery")


# ---------------------------------------------------------------------------
# receipts


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def revision() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def load_pairs(path: str) -> list[preference.PreferencePair]:
    pairs = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            chosen = row.get("chosen_reference_logprobs")
            rejected = row.get("rejected_reference_logprobs")
            if not chosen or not rejected:
                raise SystemExit(
                    "s64-launch: a pair has no cached frozen-reference "
                    "log-probabilities; run s64_preference_data.py reference "
                    "first (a scalar teacher cache is not accepted)")
            pairs.append(preference.PreferencePair(
                problem_id=str(row["problem_id"]),
                prompt_ids=tuple(int(x) for x in row["prompt_ids"]),
                chosen_ids=tuple(int(x) for x in row["chosen_ids"]),
                rejected_ids=tuple(int(x) for x in row["rejected_ids"]),
                weight=float(row.get("weight", 1.0)),
                chosen_reference_logprobs=tuple(
                    float(x) for x in chosen),
                rejected_reference_logprobs=tuple(
                    float(x) for x in rejected),
                provenance=str(row.get("provenance", "on-policy"))))
    return pairs


def cgroup_memory() -> dict:
    values = {}
    for name in ("memory.max", "memory.peak", "memory.swap.peak", "memory.events"):
        path = f"/sys/fs/cgroup/{name}"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                values[name.replace(".", "_")] = handle.read().strip()
    return values


def _vram_guard(limit_gib: float) -> None:
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("s64-launch: CUDA required; run through scripts/court --gpu")
    free_bytes, _total = torch.cuda.mem_get_info()
    if limit_gib * GIB > free_bytes:
        raise SystemExit(
            f"s64-launch: vram limit {limit_gib} GiB exceeds free "
            f"{free_bytes / GIB:.2f} GiB")


# ---------------------------------------------------------------------------
# geometry


def prepare_registered_lora(model, *, rank: int = preference.REGISTERED_LORA_RANK,
                            alpha: float = preference.REGISTERED_LORA_ALPHA,
                            expected_active_union: int | None = None) -> dict:
    """Install the registered adapter on the text tower and account for it.

    The text tower (not the outer multimodal wrapper) is the deployed model.
    Installing on the outer wrapper would add vision-tower adapters that are
    not part of the registered geometry and are never evaluated.
    """
    from frontier import arms

    text = arms._text_model(model)
    before = preference.unique_active_elements(text)
    if expected_active_union is not None and before != expected_active_union:
        raise SystemExit(
            f"s64-launch: text tower is {before} elements, expected the "
            f"registered active union {expected_active_union}")
    wrappers = preference.install_lora(text, rank=rank, alpha=alpha)
    adapter_elements = preference.adapter_element_count(model)
    if adapter_elements <= 0 or not wrappers:
        raise SystemExit("s64-launch: no LoRA modules were installed")
    after = preference.unique_active_elements(text)
    if after != before + adapter_elements:
        raise SystemExit(
            "s64-launch: adapter accounting mismatch: "
            f"{after} != {before} + {adapter_elements}")
    union_after_merge = preference.active_union_after_merge(
        before, adapter_elements)
    if expected_active_union is not None and union_after_merge != expected_active_union:
        raise SystemExit(
            f"s64-launch: post-merge union {union_after_merge} != "
            f"{expected_active_union}")
    return {
        "lora_rank": int(rank),
        "lora_alpha": float(alpha),
        "scaling": float(alpha) / float(rank),
        "targets": "every linear layer of the text tower except the output head",
        "wrapped_layers": len(wrappers),
        "base_text_elements": int(before),
        "adapter_elements": int(adapter_elements),
        "active_union_before_merge": int(after),
        "active_union_after_merge": int(union_after_merge),
        "registered_active_union": int(preference.REGISTERED_ACTIVE_UNION),
        "active_union_basis": (
            "scale * B @ A is folded into weights the base already reads; the "
            "adapter adds no deployed identity"),
    }


def _embedding_parameter_names(model) -> set[str]:
    from frontier import arms

    text = arms._text_model(model)
    embedding = model.get_input_embeddings()
    if embedding is None:
        return set()
    pointer = embedding.weight.data_ptr()
    return {name for name, parameter in text.named_parameters()
            if parameter.data_ptr() == pointer}


def snapshot_merged_full_state(model, *,
                               expected_active_union: int | None = None
                               ) -> tuple[dict, dict]:
    """Export the merged text tower without a second copy of the model.

    The adapter is folded into CPU copies of each base weight by
    ``preference.merged_state_from_model``. The training model keeps its GPU
    placement and is neither copied nor mutated, so the merge no longer needs
    a second full model's worth of device memory.
    """
    from frontier import arms

    text = arms._text_model(model)
    embedding_names = _embedding_parameter_names(model)
    merged_elements = sum(
        int(parameter.numel())
        for name, parameter in text.named_parameters()
        if not (name.endswith(".lora_A") or name.endswith(".lora_B")))
    state, merge_report = preference.merged_state_from_model(
        text, exclude_names=embedding_names)
    if expected_active_union is not None and merged_elements != expected_active_union:
        raise SystemExit(
            f"s64-launch: merged text tower is {merged_elements} elements, "
            f"expected {expected_active_union}")
    report = {
        **merge_report,
        "merged_text_elements": int(merged_elements),
        "exported_state_elements": int(sum(v.numel() for v in state.values())),
        "exported_state_keys": len(state),
        "frozen_identity_excluded": sorted(embedding_names),
        "merge_parity": {
            "atol": preference.MERGE_ATOL,
            "rtol": preference.MERGE_RTOL,
            "test": ("oracle/tests/test_s63_preference.py::"
                     "test_merge_reproduces_the_adapter_logits_within_tolerance "
                     "and oracle/tests/test_s64_preference_pipeline.py on a "
                     "float32 tiny model"),
        },
        "active_union_after_merge": int(preference.active_union_after_merge(
            merged_elements, 0)),
    }
    return state, report


def save_full_arm_checkpoint(state: dict, path: str, *, step: int,
                             processed_tokens: int, geometry: dict,
                             merge_report: dict,
                             config_sha256: str | None = None) -> str:
    """Write the same checkpoint schema `train.py` writes for the full arm."""
    import torch

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save({
        "schema": "selvedge/stage2-ckpt/v1",
        "step": int(step),
        "config_sha256": config_sha256,
        "arm": "full",
        "arm_state": state,
        "objective": "reference-anchored prefix-local preference (LoRA merged)",
        "processed_tokens": int(processed_tokens),
        "geometry": geometry,
        "merge": merge_report,
    }, path)
    return path


def save_adapter(model, path: str, *, step: int) -> str:
    import torch

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, preference.LoRALinear):
            state[f"{name}.lora_A"] = module.lora_A.detach().cpu().clone()
            state[f"{name}.lora_B"] = module.lora_B.detach().cpu().clone()
    torch.save({"schema": "selvedge/s64-lora-adapter/v1",
                "step": int(step), "rank": preference.REGISTERED_LORA_RANK,
                "alpha": preference.REGISTERED_LORA_ALPHA,
                "adapter": state}, path)
    return path


# ---------------------------------------------------------------------------
# training


def pair_schedule(pairs, *, seed: int = 0, pair_order: str = "shuffled",
                  batch_pairs: int = 1):
    """Yield ``(step, batch)`` forever from the run's deterministic order.

    This is the single ordering source for a training run. `s69_control_budget.py`
    replays the same order with its own loop over `preference.pair_stream`; the
    two implementations are compared element for element by
    `oracle/tests/test_s70_preference_weighting.py`.
    """
    stream = preference.pair_stream(pairs, seed=seed, order=pair_order)
    step = 0
    while True:
        step += 1
        yield step, [pairs[next(stream)] for _ in range(batch_pairs)]


def _batch_tokens(arm: str, batch) -> int:
    if arm == "matched-sft":
        return preference.sft_processed_tokens(batch)
    return preference.preference_processed_tokens(batch)


def _batch_loss(model, arm: str, batch, beta: float, device,
                weight_normalizer: float | None = None):
    import torch

    if arm == "matched-sft":
        losses = torch.stack([preference.sft_control_loss(model, pair, device)
                              for pair in batch])
        return losses.mean(), {
            "arm": "matched-sft",
            "pairs": len(batch),
            "chosen_tokens": sum(len(pair.chosen_ids) for pair in batch),
            "loss": float(losses.detach().mean()),
        }
    loss, detail = preference.preference_batch_loss(
        model, batch, beta, device, weight_normalizer=weight_normalizer)
    detail["arm"] = arm
    return loss, detail


def _subsample(values: list[float], limit: int = 512) -> list[float]:
    if len(values) <= limit:
        return values
    stride = max(1, len(values) // limit)
    sampled = values[::stride][:limit]
    if sampled[-1] != values[-1]:
        sampled.append(values[-1])
    return sampled


def run_arm(model, pairs, *, arm: str,
            budget_tokens: int | None,
            batch_pairs: int = 1, lr: float = 1e-4, beta: float = 0.1,
            grad_clip: float = 1.0, cadence: tuple[int, ...] = (),
            seed: int = 0, device: str | None = None,
            checkpoint_dir: str | None = None,
            config_sha256: str | None = None,
            expected_active_union: int | None = None,
            rank: int = preference.REGISTERED_LORA_RANK,
            alpha: float = preference.REGISTERED_LORA_ALPHA,
            max_steps: int | None = None,
            pair_order: str = "shuffled",
            weight_normalization: str = "corpus",
            trace_path: str | None = None) -> dict:
    """Train one arm and emit merged checkpoints at the token cadence."""
    import torch
    from frontier import arms

    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if not pairs:
        raise ValueError("the training arm needs at least one pair")
    if batch_pairs < 1:
        raise ValueError("batch_pairs must be >= 1")
    if pair_order not in preference.PAIR_ORDERS:
        raise ValueError(f"unknown pair order {pair_order!r}")
    if weight_normalization not in ("corpus", "batch"):
        raise ValueError(
            f"unknown weight normalization {weight_normalization!r}")

    # The on-policy file carries p=0 rows with weight 0 next to the positive
    # pairs. A zero-weight pair contributes nothing to the preference loss but
    # still consumes a batch slot, and the matched SFT control previously
    # trained on its chosen continuation. Drop it from the normal and matched
    # arms so both use the identical effective problem set; the recovery arm
    # is the only arm that keeps it, by construction of its own file.
    pairs_loaded = len(pairs)
    dropped_zero_weight: list[str] = []
    if arm in ("preference", "matched-sft"):
        dropped_zero_weight = sorted(pair.problem_id for pair in pairs
                                     if pair.weight <= 0)
        pairs = [pair for pair in pairs if pair.weight > 0]
        if not pairs:
            raise ValueError(
                f"{arm}: every pair has weight 0; the effective problem set "
                "is empty")
    effective_ids = sorted(pair.problem_id for pair in pairs)
    effective_ids_sha256 = hashlib.sha256(
        "\n".join(effective_ids).encode("utf-8")).hexdigest()

    if budget_tokens is None:
        budget_tokens = (preference.sft_processed_tokens(pairs)
                         if arm == "matched-sft"
                         else preference.preference_processed_tokens(pairs))
    budget_tokens = int(budget_tokens)
    if budget_tokens <= 0:
        raise ValueError("budget_tokens must be positive")

    geometry = prepare_registered_lora(
        model, rank=rank, alpha=alpha,
        expected_active_union=expected_active_union)
    # The adapter is trained through the frozen backbone. Activation
    # checkpointing is what keeps a full-length preference pair (prompt plus
    # both continuations) inside a 16 GiB card; the flag only takes effect in
    # training mode, so the model must leave `eval()` for the step.
    checkpointing = None
    if hasattr(model, "activation_checkpointing_enable"):
        model.activation_checkpointing_enable()
        checkpointing = "activation_checkpointing"
    elif hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        checkpointing = "gradient_checkpointing"
    model.train()
    trainable = preference.lora_parameters(model)
    if not trainable:
        raise ValueError("the arm needs trainable LoRA parameters")
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    schedule = pair_schedule(pairs, seed=seed, pair_order=pair_order,
                             batch_pairs=batch_pairs)
    # One corpus-level denominator, computed once from the filtered pair file.
    # `corpus` makes the registered w = p(1-p) real; `batch` is the pre-S70
    # denominator, which is identically 1 at batch_pairs = 1 and therefore
    # leaves the weight inert. The choice is recorded in the receipt.
    corpus_mean_weight = preference.corpus_mean_weight(pairs)
    weight_normalizer = (corpus_mean_weight
                         if weight_normalization == "corpus" else None)
    telemetry = (preference.UpdateTelemetry(grad_clip=grad_clip, lr=lr)
                 if trace_path else None)
    if telemetry is not None:
        telemetry.bind(trainable)
    sampled_weights: list[float] = []

    cadence = tuple(sorted(int(value) for value in cadence if int(value) > 0))
    pending = list(cadence)
    checkpoints: list[dict] = []
    losses: list[float] = []
    processed = 0
    supervised = 0
    step = 0
    started = time.time()

    while processed < budget_tokens and (max_steps is None or step < max_steps):
        step, batch = next(schedule)
        optimizer.zero_grad(set_to_none=True)
        loss, detail = _batch_loss(model, arm, batch, beta, device,
                                   weight_normalizer=weight_normalizer)
        loss.backward()
        pre_clip = (torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
                    if grad_clip and grad_clip > 0
                    else None)
        if pre_clip is None:
            pre_clip = preference.grad_norm(trainable)
        post_clip = preference.grad_norm(trainable)
        optimizer.step()
        sampled_weights.extend(float(pair.weight) for pair in batch)
        if telemetry is not None:
            telemetry.row(
                step=step, arm=arm,
                problem_ids=[pair.problem_id for pair in batch],
                weights=[pair.weight for pair in batch],
                loss=float(loss.detach()),
                mean_weight=corpus_mean_weight,
                weight_normalizer=weight_normalizer,
                pre_clip_grad_norm=float(pre_clip),
                post_clip_grad_norm=float(post_clip))
        processed += _batch_tokens(arm, batch)
        supervised += sum(len(pair.chosen_ids) for pair in batch)
        losses.append(float(loss.detach()))

        while pending and processed >= pending[0]:
            target = pending.pop(0)
            if checkpoint_dir is None:
                raise ValueError("a cadence was given without a checkpoint_dir")
            state, merge_report = snapshot_merged_full_state(
                model, expected_active_union=expected_active_union)
            name = f"ckpt-{step:06d}.pt"
            path = os.path.join(checkpoint_dir, name)
            save_full_arm_checkpoint(
                state, path, step=step, processed_tokens=processed,
                geometry=geometry, merge_report=merge_report,
                config_sha256=config_sha256)
            checkpoints.append({
                "cadence_tokens": target, "step": step,
                "processed_tokens": processed,
                "path": path, "sha256": sha256_file(path),
                "merge": merge_report,
            })
            del state, merge_report

    if checkpoint_dir is not None and (not checkpoints
                                       or checkpoints[-1]["processed_tokens"] != processed):
        state, merge_report = snapshot_merged_full_state(
            model, expected_active_union=expected_active_union)
        path = os.path.join(checkpoint_dir, f"ckpt-{step:06d}.pt")
        save_full_arm_checkpoint(
            state, path, step=step, processed_tokens=processed,
            geometry=geometry, merge_report=merge_report,
            config_sha256=config_sha256)
        checkpoints.append({
            "cadence_tokens": None, "step": step,
            "processed_tokens": processed, "path": path,
            "sha256": sha256_file(path), "merge": merge_report,
        })
        del state, merge_report

    wall = time.time() - started
    trace = None
    if telemetry is not None:
        telemetry.write(trace_path)
        trace = {"path": trace_path, "summary": telemetry.summary()}
    return {
        "kind": arm,
        "steps": step,
        "budget_tokens": budget_tokens,
        "pairs_loaded": pairs_loaded,
        "effective_pairs": len(pairs),
        "zero_weight_pairs_dropped": len(dropped_zero_weight),
        "zero_weight_problem_ids": dropped_zero_weight,
        "effective_problem_ids_sha256": effective_ids_sha256,
        "processed_tokens": processed,
        "supervised_chosen_tokens": supervised,
        "lr": lr,
        "beta": beta if arm != "matched-sft" else None,
        "batch_pairs": batch_pairs,
        "grad_clip": grad_clip,
        "normalization": preference.REGISTERED_NORMALIZATION,
        "pair_order": pair_order,
        "weight_normalization": weight_normalization,
        "corpus_mean_weight": corpus_mean_weight,
        "weight_scale_basis": (
            "corpus mean weight over the positive-weight pairs"
            if weight_normalizer is not None else
            "batch weight sum (the pre-S70 denominator; identically 1 at "
            "batch_pairs = 1, so the weight is inert)"),
        "realized_mean_weight": (
            sum(sampled_weights) / len(sampled_weights)
            if sampled_weights else None),
        "realized_mean_weight_over_mean": (
            (sum(sampled_weights) / len(sampled_weights)) / corpus_mean_weight
            if sampled_weights else None),
        "trace": trace,
        "wall_seconds": wall,
        "processed_tokens_per_second": (processed / wall if wall > 0 else None),
        "first_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "mean_loss": (sum(losses) / len(losses)) if losses else None,
        "loss_trace": _subsample(losses),
        "geometry": geometry,
        "activation_checkpointing": checkpointing,
        "checkpoints": checkpoints,
    }


# ---------------------------------------------------------------------------
# CLI


def _parse_cadence(text: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in text.split(",") if part.strip())
    if any(value <= 0 for value in values):
        raise SystemExit("s64-launch: cadence values must be positive")
    return values


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, default=None)
    parser.add_argument("--pairs", default=None,
                        help="on-policy pairs with the reference cache "
                             "(preference and matched-sft)")
    parser.add_argument("--recovery-pairs", default=None,
                        help="teacher-guided recovery pairs with the reference "
                             "cache (recovery)")
    parser.add_argument("--student-dir", required=True)
    parser.add_argument("--init-checkpoint", default=None,
                        help="a stage-2 full-arm checkpoint loaded over "
                             "--student-dir before the adapter is installed; "
                             "the receipt records its path, sha256 and step")
    parser.add_argument("--budget-tokens", type=int, default=5_000_000)
    parser.add_argument("--cadence", default="500000,1000000,2000000,5000000")
    parser.add_argument("--batch-pairs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=preference.REGISTERED_LORA_RANK)
    parser.add_argument("--alpha", type=float,
                        default=preference.REGISTERED_LORA_ALPHA)
    parser.add_argument("--seed", type=int, default=6400)
    parser.add_argument("--pair-order", choices=preference.PAIR_ORDERS,
                        default="shuffled",
                        help="`shuffled` reproduces every pre-S70 run from its "
                             "seed; `weighted` draws a deterministic "
                             "difficulty-proportional permutation per epoch, "
                             "which is the mechanism that makes w = p(1-p) "
                             "change realized updates under AdamW")
    parser.add_argument("--weight-normalization",
                        choices=("corpus", "batch"), default="corpus",
                        help="the preference-loss denominator: `corpus` "
                             "divides by the mean weight over the positive-"
                             "weight pairs, computed once and recorded; "
                             "`batch` keeps the pre-S70 batch weight sum, "
                             "which is identically 1 at batch_pairs = 1")
    parser.add_argument("--trace-out", default=None,
                        help="write the realized-update trace JSON here "
                             "(pre/post-clip gradient norms and the parameter "
                             "delta L2 per step, tagged with the pair weight); "
                             "off by default because it copies the trainable "
                             "parameters once per step")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vram-limit-gib", type=float, default=13.0)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--geometry-only", action="store_true",
                        help="load the student, install the registered LoRA, "
                             "print the geometry and exit; no pairs needed")
    parser.add_argument("--geometry-check-merge", action="store_true",
                        help="with --geometry-only, also snapshot and merge a "
                             "deep copy and print the exported state contract")
    args = parser.parse_args(argv)

    if args.geometry_only:
        import torch
        from frontier import arms
        if args.device == "cuda":
            _vram_guard(args.vram_limit_gib)
        model = arms.load_model(args.student_dir, dtype=torch.bfloat16,
                                device=args.device)
        geometry = prepare_registered_lora(
            model, rank=args.rank, alpha=args.alpha,
            expected_active_union=preference.REGISTERED_ACTIVE_UNION)
        if args.geometry_check_merge:
            state, merge = snapshot_merged_full_state(
                model, expected_active_union=preference.REGISTERED_ACTIVE_UNION)
            merge["exported_state_keys"] = len(state)
            merge["exported_state_elements"] = int(
                sum(value.numel() for value in state.values()))
            geometry["merge"] = merge
            del state
        print(json.dumps(geometry, indent=2, sort_keys=True))
        return 0

    if args.arm is None:
        raise SystemExit("s64-launch: --arm is required for a training run")
    if args.arm == "recovery":
        if not args.recovery_pairs:
            raise SystemExit("s64-launch: the recovery arm needs --recovery-pairs")
        pair_path = args.recovery_pairs
    else:
        if not args.pairs:
            raise SystemExit("s64-launch: this arm needs --pairs")
        pair_path = args.pairs
    if not args.out_dir:
        raise SystemExit("s64-launch: --out-dir is required for a training run")

    _vram_guard(args.vram_limit_gib)
    import torch
    from frontier import arms

    device = args.device
    model = arms.load_model(args.student_dir, dtype=torch.bfloat16,
                            device=device)
    init_checkpoint = None
    if args.init_checkpoint:
        from frontier import full

        checkpoint_path = os.path.join(ROOT, args.init_checkpoint)
        payload = torch.load(checkpoint_path, map_location="cpu",
                             weights_only=False)
        if payload.get("arm") not in (None, "full"):
            raise SystemExit(
                "s64-launch: --init-checkpoint is not a full-arm checkpoint "
                f"(arm={payload.get('arm')!r})")
        full.load_full_state_into(model, payload["arm_state"])
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        init_checkpoint = {
            "path": args.init_checkpoint,
            "sha256": sha256_file(checkpoint_path),
            "step": payload.get("step"),
            "tokens_seen": payload.get("tokens_seen"),
            "arm": "full",
        }
        print(f"s64-launch: initialised the student from "
              f"{args.init_checkpoint} step={payload.get('step')}", flush=True)
    pairs = load_pairs(pair_path)
    out_dir = (args.out_dir if os.path.isabs(args.out_dir)
               else os.path.join(ROOT, args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()

    if (args.rank != preference.REGISTERED_LORA_RANK
            or args.alpha != preference.REGISTERED_LORA_ALPHA):
        print(f"s64-launch: WARNING geometry rank={args.rank} "
              f"alpha={args.alpha} is not the registered rank="
              f"{preference.REGISTERED_LORA_RANK} alpha="
              f"{preference.REGISTERED_LORA_ALPHA}", flush=True)

    # run_arm installs the adapter exactly once and returns the accounting.
    report = run_arm(
        model, pairs, arm=args.arm, budget_tokens=args.budget_tokens,
        batch_pairs=args.batch_pairs, lr=args.lr, beta=args.beta,
        grad_clip=args.grad_clip, cadence=_parse_cadence(args.cadence),
        seed=args.seed, device=device, checkpoint_dir=out_dir,
        config_sha256=None,
        expected_active_union=preference.REGISTERED_ACTIVE_UNION,
        rank=args.rank, alpha=args.alpha, max_steps=args.max_steps,
        pair_order=args.pair_order,
        weight_normalization=args.weight_normalization,
        trace_path=(os.path.join(ROOT, args.trace_out)
                    if args.trace_out and not os.path.isabs(args.trace_out)
                    else args.trace_out))

    adapter_path = os.path.join(
        out_dir, f"adapter-{args.arm}-{report['steps']:06d}.pt")
    save_adapter(model, adapter_path, step=report["steps"])

    peak_alloc = torch.cuda.max_memory_allocated() / GIB
    peak_reserved = torch.cuda.max_memory_reserved() / GIB
    receipt = {
        "schema": SCHEMA,
        "kind": "preference-arm-run",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "registration": (
            "docs/ws19-stage2-correction-s63-preregistration-v1.json"),
        "command": sys.argv,
        "arm": args.arm,
        "student_dir": args.student_dir,
        "init_checkpoint": init_checkpoint,
        "pairs": {
            "path": os.path.relpath(os.path.abspath(pair_path), ROOT),
            "sha256": sha256_file(os.path.abspath(pair_path)),
            "loaded_rows": report.get("pairs_loaded"),
            "effective_rows": report.get("effective_pairs"),
            "zero_weight_pairs_dropped": report.get(
                "zero_weight_pairs_dropped"),
            "zero_weight_problem_ids": report.get(
                "zero_weight_problem_ids"),
            "effective_problem_ids_sha256": report.get(
                "effective_problem_ids_sha256"),
            "effective_set_rule": (
                "preference and matched-sft drop weight<=0 pairs before "
                "budgeting and training, so both arms use the identical "
                "effective problem set; the recovery arm is the only arm "
                "that keeps the p=0 pairs"),
            "normalization": preference.REGISTERED_NORMALIZATION,
            "pair_order": report.get("pair_order"),
            "weight_normalization": report.get("weight_normalization"),
            "corpus_mean_weight": report.get("corpus_mean_weight"),
            "weight_scale_basis": report.get("weight_scale_basis"),
            "realized_mean_weight": report.get("realized_mean_weight"),
            "realized_mean_weight_over_mean": report.get(
                "realized_mean_weight_over_mean"),
            "update_trace": report.get("trace"),
        },
        "budget": {
            "tokens": args.budget_tokens,
            "basis": ("processed forward-token positions; the matched SFT "
                      "control gets the same budget on the same chosen pairs"),
            "cadence": list(_parse_cadence(args.cadence)),
        },
        "geometry": report["geometry"],
        "training": {key: value for key, value in report.items()
                     if key != "geometry"},
        "adapter": {"path": adapter_path, "sha256": sha256_file(adapter_path)},
        "peak_allocated_gib": peak_alloc,
        "peak_reserved_gib": peak_reserved,
        "cgroup_memory": cgroup_memory(),
        "claims": {
            "lora_rank": args.rank,
            "lora_alpha": args.alpha,
            "all_linear_except_output_head": True,
            "base_frozen": True,
            "merged_checkpoints": True,
            "active_union_after_merge": report["geometry"]["active_union_after_merge"],
            "merge_parity_unit_test": (
                "oracle/tests/test_s63_preference.py and "
                "oracle/tests/test_s64_preference_pipeline.py"),
            "teacher_loaded": False,
            "reference_loaded": False,
            "score_used_for_selection": False,
        },
    }
    receipt_path = os.path.join(out_dir, f"receipt-{args.arm}.json")
    write_json(receipt_path, receipt)
    print(f"s64-launch: {args.arm} {report['steps']} steps, "
          f"{report['processed_tokens']} tokens -> {out_dir}")
    print(f"s64-launch: receipt {receipt_path}")
    if report["checkpoints"]:
        milestones = ",".join(
            f"{entry['cadence_tokens'] or 'final'}={entry['step']}"
            for entry in report["checkpoints"])
        print("s64-launch: milestone plan next:\n  SELVEDGE_MEM=4g scripts/court "
              "oracle -- python3 scripts/s62_score_milestones.py --plan "
              f"--run-dir {os.path.relpath(out_dir, ROOT)} "
              f"--milestones {milestones}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
