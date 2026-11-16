#!/usr/bin/env python3
"""P13.14 runner: fit and measure omission-conditioned recovery.

Five arms, one base arm, one data order, one evaluation instrument:

  sparse-only                    the deployed top-k sparse layer, no correction
  sparse+alignment               + the layer-wise output-reconstruction baseline
  sparse+compensator             + C_theta(h, S)
  sparse+alignment+compensator   + both, at the matched element budget
  dense-teacher-reference        all m groups at unit weight; not deployable

The correction is fitted and trained here; the metrics are the registered ones,
computed by the same `harness` primitives `oracle/eval/baseline.py` (T2.9) calls
(KL to the distillation teacher, top-1 agreement, per-domain PPL, and the three
task accuracies). This module does not modify the registered instrument.

Budget rule (the M2 rule, `docs/EXPERIMENT-CONTRACT.md` M2.2): every learned
selector or compensator weight is charged to the arm's active unique element
count, and two arms are compared only at identical realized counts. The ledger
printed by `--plan` is arithmetic from the registered constants.

Modes:

  --plan            print the arm matrix and the exact commands; needs no model
  --smoke           tiny-model end-to-end on CPU: build, capture, fit, train,
                    report; no registered metric is produced
  (default)         the registered run; requires the GPU court
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for _path in (HERE, os.path.join(REPO, "oracle", "train")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

SCHEMA = "selvedge/p13-recovery-court/v1"
ARMS = ("sparse-only", "sparse+alignment", "sparse+compensator",
        "sparse+alignment+compensator", "dense-teacher-reference")
CALIBRATION_WINDOWS = 8
TRAIN_STEPS = 512
LEARNING_RATE = 5e-4
BATCH_WINDOWS = 2


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# activations


class LayerCapture:
    """Record a bank's input and output for every layer, on one stream slice.

    The compensator's input during training is exactly what is captured here, so
    the closed-form baseline and the trained compensator see the same states.
    """

    def __init__(self, arm_model, limit_positions: int = 32768):
        self.arm_model = arm_model
        self.limit = limit_positions
        self.inputs: dict[int, list[torch.Tensor]] = {}
        self.outputs: dict[int, list[torch.Tensor]] = {}
        self._handles = []

    def __enter__(self):
        for index, bank in enumerate(self.arm_model.banks()):
            def pre_hook(_module, args, index=index):
                if args and torch.is_tensor(args[0]):
                    if sum(item.numel() for item in self.inputs.get(index, [])) \
                            < self.limit:
                        self.inputs.setdefault(index, []).append(
                            args[0].detach().to(torch.float32).cpu())

            def post_hook(_module, _args, output, index=index):
                if sum(item.numel() for item in self.outputs.get(index, [])) < self.limit:
                    self.outputs.setdefault(index, []).append(
                        output.detach().to(torch.float32).cpu())

            self._handles.append(bank.register_forward_pre_hook(pre_hook))
            self._handles.append(bank.register_forward_hook(post_hook))
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles = []
        return False

    def stacked(self) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        inputs = {index: torch.cat(parts, dim=0) for index, parts in self.inputs.items()}
        outputs = {index: torch.cat(parts, dim=0) for index, parts in self.outputs.items()}
        return inputs, outputs


def capture_activations(arm_model, stream: torch.Tensor, calibration_tokens: int,
                        device, window: int = 1024) -> dict:
    """Two passes: the deployed sparse output, then the full-bank reference.

    The bank's *input* is identical in both passes, so it is captured once. The
    retained bytes are `3 * positions * hidden * 4` per layer, which the caller
    sizes through `--calibration-tokens`; the receipt records the arithmetic.
    """
    arm_model.model.eval()
    usable = min(calibration_tokens, int(stream.numel()))
    usable -= usable % window
    captured = {"positions": usable, "window": window, "sparse": {}, "dense": {}}
    inputs: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for mode in ("sparse", "dense"):
            for bank in arm_model.banks():
                bank.full_bank = mode == "dense"
                bank.exclude = None
            with LayerCapture(arm_model, limit_positions=usable) as capture:
                for start in range(0, usable, window):
                    ids = stream[start:start + window].reshape(1, window).to(device)
                    arm_model.forward(input_ids=ids)
                pass_inputs, outputs = capture.stacked()
            if not inputs:
                inputs = pass_inputs
            captured[mode] = {"inputs": inputs, "outputs": outputs}
            for bank in arm_model.banks():
                bank.full_bank = False
    hidden = next(iter(inputs.values())).shape[-1] if inputs else 0
    captured["retained_bytes"] = 3 * usable * hidden * 4 * len(inputs)
    return captured


# ---------------------------------------------------------------------------
# fitting


def fit_arm(arm_model, arm, captured: dict, mode: str, alignment_mode: str = "l2",
            ridge: float = 1e-3) -> list[dict]:
    """Fit the alignment maps in closed form for every layer."""
    reports = []
    if mode not in ("alignment", "both"):
        return reports
    for index, alignment in enumerate(arm.alignments):
        features = captured["sparse"]["inputs"][index]
        sparse = captured["sparse"]["outputs"][index]
        dense = captured["dense"]["outputs"][index]
        targets = dense.reshape(-1, dense.shape[-1]) - sparse.reshape(-1, sparse.shape[-1])
        flat = features.reshape(-1, features.shape[-1])
        report = alignment.fit(flat, targets, ridge=ridge, mode=alignment_mode)
        report["layer"] = index
        reports.append(report)
    return reports


def train_compensator(arm_model, arm, targets, stream, device, steps: int,
                      learning_rate: float, window: int = 1024,
                      batch_windows: int = 2, warmup: int = 32) -> dict:
    """Train the compensator by distillation through the frozen sparse model."""
    import loop

    trainable = [parameter for compensator in arm.compensators
                 for parameter in compensator.parameters()]
    if not trainable:
        return {"steps": 0, "reason": "arm has no compensator"}
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, betas=(0.9, 0.95),
                                  eps=1e-8, weight_decay=0.0)
    arm_model.model.train()
    losses = []
    started = time.time()
    for step in range(steps):
        total = 0.0
        for item in range(batch_windows):
            start = ((warmup + step * batch_windows + item) * window)
            ids = stream[start:start + window].reshape(1, window).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=ids.is_cuda):
                logits = arm_model.forward(input_ids=ids).logits
            flat = logits.reshape(-1, logits.shape[-1])
            values, indices, lse = targets.read(start, flat.shape[0], device)
            loss = loop.distillation_loss(flat, values, indices, lse)
            (loss / batch_windows).backward()
            total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(total / batch_windows)
    return {"steps": steps, "lr": learning_rate, "batch_windows": batch_windows,
            "loss_first": losses[0], "loss_last": losses[-1],
            "seconds": time.time() - started,
            "trainable_elements": sum(int(item.numel()) for item in trainable)}


# ---------------------------------------------------------------------------
# measurement


def measure(student, teacher, tokenizer, windows, task_data, device,
            limit: dict | None = None) -> dict:
    """The T2.9 metrics, computed by the registered harness primitives."""
    import harness
    import tasks as task_module

    limit = limit or {}
    result = {"distributional": harness.distributional(student, teacher, windows,
                                                       device)}
    if limit.get("skip_tasks"):
        result["tasks"] = {"skipped": "readers asked for distributional metrics only"}
        return result
    humaneval = task_module.load_humaneval(limit["dataset_root"])
    gsm8k = task_module.gsm8k_subset(limit["dataset_root"], limit.get("gsm8k", 300))
    truthfulqa = task_module.load_truthfulqa(limit["dataset_root"])
    if limit.get("humaneval", 164) < len(humaneval):
        humaneval = humaneval[:limit["humaneval"]]
    if limit.get("truthfulqa", 817) < len(truthfulqa):
        truthfulqa = truthfulqa[:limit["truthfulqa"]]
    result["tasks"] = {
        "humaneval": _strip_generations(harness.run_humaneval(student, tokenizer,
                                                             humaneval, device)),
        "gsm8k": _strip_generations(harness.run_gsm8k(student, tokenizer, gsm8k,
                                                     device)),
        "truthfulqa": _strip_generations(harness.run_truthfulqa(student, tokenizer,
                                                               truthfulqa, device)),
    }
    return result


def _strip_generations(block: dict) -> dict:
    """Keep the outcome fields; the raw generations are in the base receipt."""
    slim = {}
    for key, value in block.items():
        if key == "items":
            slim[key] = [{item: row[item] for item in row
                          if item in ("id", "correct", "passed", "predicted_index")}
                         for row in value]
        else:
            slim[key] = value
    return slim


# ---------------------------------------------------------------------------
# modes


def plan(args) -> int:
    import recovery

    described = recovery.describe_arms(base_active=args.base_active,
                                       bank_k=args.k, bank_experts=args.experts,
                                       hidden=args.hidden,
                                       config=recovery.RecoveryConfig(
                                           rank=args.rank, set_dim=args.set_dim,
                                           set_mode=args.set_mode))
    described["arms_in_order"] = list(ARMS)
    described["instrument"] = {
        "metrics": ["kl", "top1", "per-domain ppl", "humaneval pass@1",
                    "gsm8k exact match", "truthfulqa mc1"],
        "statistics": {"kind": "percentile paired bootstrap over items/documents",
                       "seed": 20261115, "resamples": 10000},
        "source": "oracle/eval/harness.py, the same primitives baseline.py (T2.9) "
                  "calls; the registered instrument is not modified",
    }
    described["omission_definition"] = (
        "the sparse layer runs the router's top-k groups at their softmax weight; "
        "the omitted groups are the complement. The dense reference runs all m at "
        "unit weight, which is the reference T2.5's utility labels use.")
    described["training"] = {"steps": TRAIN_STEPS, "lr": LEARNING_RATE,
                             "calibration_windows": CALIBRATION_WINDOWS,
                             "objective": "distillation (T1.4) through the frozen "
                                          "sparse model, compensator only"}
    print(json.dumps(described, indent=2, sort_keys=True))
    return 0


def smoke(args) -> int:
    """Tiny-model end-to-end: build, capture, fit, train, report. CPU only."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    import modeling
    import recovery

    torch.manual_seed(args.seed)
    config = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=16,
                         max_position_embeddings=64)
    base = Qwen3ForCausalLM(config)
    admission = {"active_unique_params": 0, "admitted": True}
    arm_model = modeling.build_arm(base, "X", {"profile": "t2"}, admission, args.seed,
                                   profile="t2")
    arm = recovery.attach_recovery(
        arm_model, mode="both",
        config=recovery.RecoveryConfig(rank=args.rank, set_dim=args.set_dim,
                                       set_mode=args.set_mode), seed=args.seed)
    installations = recovery.install_recovery(arm_model, arm)
    device = torch.device("cpu")
    stream = torch.randint(0, config.vocab_size, (2048,), generator=torch.Generator()
                           .manual_seed(args.seed))
    captured = capture_activations(arm_model, stream, 2048, device, window=64)
    reports = fit_arm(arm_model, arm, captured, mode="both")
    frozen = recovery.freeze_arm_but_recovery(arm_model, arm, enable="compensator")
    # One real forward and backward through the *installed* wrapper, so the smoke
    # proves the correction is on the graph rather than merely constructed.
    ids = stream[:64].reshape(1, 64).to(device)
    loss = arm_model.forward(input_ids=ids).logits.float().pow(2).mean()
    loss.backward()
    gradients = [parameter.grad for compensator in arm.compensators
                 for parameter in compensator.parameters()]
    connected = all(gradient is not None for gradient in gradients)
    peak = max((float(gradient.abs().max()) for gradient in gradients
                if gradient is not None), default=0.0)
    ledger = arm.ledger(base_active=admission["active_unique_params"],
                        bank_k=arm_model.banks()[0].k,
                        bank_experts=arm_model.banks()[0].experts)
    payload = {
        "schema": SCHEMA,
        "stamp": stamp_now(),
        "created_utc": utc_now(),
        "mode": "smoke",
        "note": "tiny model, two layers, CPU; no registered metric is produced",
        "layers": len(arm.wrappers),
        "installed": len(installations),
        "positions_captured": captured["positions"],
        "alignment_rank": arm.alignment_rank,
        "alignment_reports": reports,
        "freeze": frozen,
        "compensator_gradient_max": peak,
        # The last layer is zero at step 0 by design, so only it has a non-zero
        # gradient on the first backward; the deeper tensors are connected but
        # receive exactly zero until the last layer moves. That is the zero
        # point, not a wiring failure.
        "compensator_graph_connected": connected,
        "compensator_receives_gradient": connected and peak > 0.0,
        "ledger": ledger,
        "config": recovery.RecoveryConfig(rank=args.rank, set_dim=args.set_dim,
                                          set_mode=args.set_mode).declared(),
        "revision": os.environ.get("SELVEDGE_REVISION", "unknown"),
    }
    recovery.remove_recovery(installations)
    write_json_atomic(args.out or f"results/t2/p13-smoke-{payload['stamp']}.json",
                      payload)
    print(json.dumps({key: payload[key] for key in
                      ("layers", "installed", "positions_captured", "alignment_rank",
                       "freeze", "compensator_graph_connected",
                       "compensator_receives_gradient", "ledger")},
                     indent=2, sort_keys=True))
    return 0


def run(args) -> int:
    """The registered run. Requires the GPU court and the base arm's checkpoint."""
    import harness
    import loop
    import modeling
    import recovery
    import tasks as task_module

    if not torch.cuda.is_available():
        raise SystemExit("the P13.14 run requires the GPU court (--gpu); "
                         "use --plan or --smoke without one")
    device = torch.device("cuda")
    started = time.time()
    tokenizer = harness.load_tokenizer(args.hf_root)
    split, split_sha = task_module.load_split(args.split_dir, args.split_receipts)
    windows = tasks_heldout(split)
    teacher = harness.load_model(harness.model_path(args.hf_root, "1.7b"), device)
    config, admission = modeling.load_admission(args.config, args.admission)
    base = harness.load_model(harness.model_path(args.hf_root, args.model_size), device)
    arm_model = modeling.build_arm(base, args.arm, config, admission, args.seed,
                                   profile="t2", mask_order=args.mask_order)
    arm_model.model.to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    arm_model.model.load_state_dict(state["model"], strict=False)
    del state
    arm_model.model.eval()

    stream = np.memmap(args.stream, dtype="<u4", mode="r")
    stream = torch.from_numpy(np.asarray(stream[:args.calibration_tokens * 8]).astype(
        "int64"))
    targets = loop.TargetReader(args.targets)
    captured = capture_activations(arm_model, stream, args.calibration_tokens, device)

    receipt = {
        "schema": SCHEMA,
        "stamp": stamp_now(),
        "created_utc": utc_now(),
        "revision": os.environ.get("SELVEDGE_REVISION", "unknown"),
        "arm": args.arm,
        "base_checkpoint": {"path": args.checkpoint,
                            "sha256": sha256_file(args.checkpoint)},
        "split_sha256": split_sha,
        "config": {"rank": args.rank, "set_dim": args.set_dim,
                   "set_mode": args.set_mode, "mask_order": args.mask_order},
        "calibration": {"windows": args.calibration_windows,
                        "positions": captured["positions"],
                        "retained_bytes": captured.get("retained_bytes")},
        "training": {"steps": args.steps, "lr": args.learning_rate},
        "arms": {},
        "seconds": None,
    }

    for name in ARMS:
        mode = {"sparse-only": "none", "sparse+alignment": "alignment",
                "sparse+compensator": "compensator",
                "sparse+alignment+compensator": "both",
                "dense-teacher-reference": "none"}[name]
        arm = recovery.attach_recovery(
            arm_model, mode=mode,
            config=recovery.RecoveryConfig(rank=args.rank, set_dim=args.set_dim,
                                           set_mode=args.set_mode), seed=args.seed)
        installations = recovery.install_recovery(arm_model, arm)
        alignment_reports = fit_arm(arm_model, arm, captured, mode=mode)
        freeze = recovery.freeze_arm_but_recovery(arm_model, arm,
                                                 enable="compensator")
        training = {}
        if mode in ("compensator", "both"):
            training = train_compensator(arm_model, arm, targets, stream, device,
                                        args.steps, args.learning_rate)
        for bank in arm_model.banks():
            bank.full_bank = name == "dense-teacher-reference"
        metrics = measure(arm_model.model, teacher, tokenizer, windows,
                         {"dataset_root": args.dataset_root,
                          "gsm8k": args.gsm8k_limit,
                          "humaneval": args.humaneval_limit,
                          "truthfulqa": args.truthfulqa_limit,
                          "skip_tasks": args.skip_tasks}, device)
        for bank in arm_model.banks():
            bank.full_bank = False
        receipt["arms"][name] = {
            "mode": mode,
            "ledger": arm.ledger(base_active=admission["active_unique_params"],
                                 bank_k=arm_model.banks()[0].k,
                                 bank_experts=arm_model.banks()[0].experts),
            "freeze": freeze,
            "training": training,
            "alignment_reports": alignment_reports,
            "metrics": metrics,
        }
        recovery.remove_recovery(installations)
        for wrapper in arm.wrappers:
            wrapper.compensator = None
            wrapper.alignment = None
        arm.compensators.clear()
        arm.alignments.clear()
        torch.cuda.empty_cache()

    receipt["seconds"] = time.time() - started
    write_json_atomic(args.out, receipt)
    print(json.dumps({"out": args.out, "arms": list(receipt["arms"]),
                      "seconds": receipt["seconds"]}, indent=2, sort_keys=True))
    return 0


def tasks_heldout(split: dict) -> dict:
    import tasks as task_module

    return task_module.heldout_windows(split)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--arm", default="H")
    parser.add_argument("--model-size", default="0.6b")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--admission", default=None)
    parser.add_argument("--stream", default="results/artifacts/train/train.bin")
    parser.add_argument("--targets", default="results/artifacts/train")
    parser.add_argument("--dataset-root", default="results/artifacts/datasets")
    parser.add_argument("--split-dir", default="results/artifacts")
    parser.add_argument("--split-receipts", default="results/splits")
    parser.add_argument("--hf-root", default=os.environ.get("HF_HOME", "/hf-cache"))
    parser.add_argument("--mask-order", default="legacy", choices=["legacy", "pair"])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--set-dim", type=int, default=8)
    parser.add_argument("--set-mode", default="omitted",
                        choices=["selected", "omitted"])
    parser.add_argument("--calibration-tokens", type=int, default=CALIBRATION_WINDOWS * 1024)
    parser.add_argument("--calibration-windows", type=int, default=CALIBRATION_WINDOWS)
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--base-active", type=int, default=772_440_064,
                        help="the base arm's admitted active unique count (plan only)")
    parser.add_argument("--humaneval-limit", type=int, default=164)
    parser.add_argument("--gsm8k-limit", type=int, default=300)
    parser.add_argument("--truthfulqa-limit", type=int, default=817)
    parser.add_argument("--skip-tasks", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    if args.plan:
        return plan(args)
    if args.smoke:
        return smoke(args)
    for required in ("checkpoint", "config", "admission", "out"):
        if not getattr(args, required):
            raise SystemExit(f"--{required} is required for the registered run")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
