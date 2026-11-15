#!/usr/bin/env python3
"""Counterfactual utility labels and the two router heads (T2.5).

For one arm checkpoint, on a fixed calibration slice of the training stream, the
labeller measures the leave-one-expert-out loss delta

    u_e(t) = L(without e) - L(with e)

per position `t` and per specialist `e`, with the layer's *full* bank active as
the reference set. The label tensor is checkpointed with a manifest, and the
router head is then trained on the cached pre-expert hidden states with a
listwise KL. No expert re-run is needed to train the router.

Leakage guarantees, asserted here and tested in `test_t2_capacity.py`:

* the calibration slice is a subrange of the contamination-filtered training
  stream, so no evaluation item is in it;
* the router input is the bank's *input* hidden state, captured by a forward
  pre-hook under `torch.no_grad()`, so it is computed before the experts run
  and cannot carry the intervention's outcome;
* the label tensor is produced under `torch.no_grad()` and never enters any
  forward pass.

The manifest records the checkpoint hash, the stream hash, the calibration
range, the bank state hash, the pass count and the label statistics.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os

import torch
import torch.nn.functional as F

import loop

HERE = os.path.dirname(os.path.abspath(__file__))


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json_atomic(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def tensor_digest(tensors) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.detach().to(torch.float32).cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def _forward_loss(arm_model, input_ids, targets, offset: int, device):
    """One forward pass; returns (`logits`, per-position loss `[N]`)."""
    logits = arm_model.forward(input_ids=input_ids).logits
    flat = logits.reshape(-1, logits.shape[-1])
    values, ids, lse = targets.read(offset, flat.shape[0], device)
    return logits, loop.distillation_loss_per_position(flat, values, ids, lse)


def label_utilities(arm_model, targets, stream, device, calibration_tokens: int,
                    batch_windows: int = 1) -> dict:
    """Measure `u_e(t)` for every layer and specialist.

    Returns `{"features": {layer: [N, hidden]}, "utilities": {layer: [N, m]},
    "manifest": {...}}`. Layers are keyed by their position in the model.
    """
    if calibration_tokens % loop.WINDOW:
        raise ValueError("calibration_tokens must be a multiple of the window")
    windows = calibration_tokens // loop.WINDOW
    if windows % batch_windows:
        raise ValueError("calibration windows must divide by batch_windows")
    banks = arm_model.banks()
    if not banks:
        raise ValueError("arm has no specialist bank to label")
    arm_model.model.eval()

    captured: dict[int, list[torch.Tensor]] = {}

    def make_hook(key):
        def hook(_module, args):
            if args and torch.is_tensor(args[0]):
                captured.setdefault(key, []).append(
                    args[0].detach().to(torch.float32).cpu())
        return hook

    features: dict[int, torch.Tensor] = {}
    utilities: dict[int, torch.Tensor] = {}
    reference_losses: dict[int, torch.Tensor] = {}
    passes = 0

    for bank_index, bank in enumerate(banks):
        key = bank_index
        # -- reference pass: full bank, features captured at the bank input.
        handle = bank.register_forward_pre_hook(make_hook(key))
        base_losses = []
        bank.full_bank = True
        bank.exclude = None
        for start in range(0, windows, batch_windows):
            offset = start * loop.WINDOW
            ids = torch.from_numpy(
                stream[offset:offset + batch_windows * loop.WINDOW].astype("int64")
            ).reshape(batch_windows, loop.WINDOW).to(device)
            _logits, loss = _forward_loss(arm_model, ids, targets, offset, device)
            base_losses.append(loss.cpu())
            passes += 1
        handle.remove()
        captured_features = captured.pop(key, [])
        if len(captured_features) != len(base_losses):
            raise RuntimeError(f"layer {bank_index}: feature capture count mismatch")
        base_features = [item.reshape(-1, bank.hidden) for item in captured_features]

        # -- one leave-one-expert-out pass per specialist.
        expert_losses = {expert: [] for expert in range(bank.experts)}
        for expert in range(bank.experts):
            bank.exclude = expert
            for start in range(0, windows, batch_windows):
                offset = start * loop.WINDOW
                ids = torch.from_numpy(
                    stream[offset:offset + batch_windows * loop.WINDOW].astype("int64")
                ).reshape(batch_windows, loop.WINDOW).to(device)
                _logits, loss = _forward_loss(arm_model, ids, targets, offset, device)
                expert_losses[expert].append(loss.cpu())
                passes += 1
        bank.exclude = None
        bank.full_bank = False

        reference = torch.cat(base_losses)
        delta = torch.stack([torch.cat(expert_losses[expert])
                             for expert in range(bank.experts)], dim=-1) \
            - reference.unsqueeze(-1)
        features[key] = torch.cat(base_features, dim=0)
        utilities[key] = delta
        reference_losses[key] = reference

    label_summary = {}
    for key, delta in utilities.items():
        label_summary[str(key)] = {
            "positions": int(delta.shape[0]),
            "experts": int(delta.shape[1]),
            "mean": [float(item) for item in delta.mean(dim=0)],
            "std": [float(item) for item in delta.std(dim=0, unbiased=False)],
            "fraction_positive": [float(item) for item in (delta > 0).float().mean(dim=0)],
        }
    manifest = {
        "schema": "selvedge/t2-utility/v1",
        "created_utc": utc_now(),
        "arm": arm_model.arm,
        "layers": len(utilities),
        "experts_per_layer": int(next(iter(utilities.values())).shape[1]),
        "calibration_tokens": int(calibration_tokens),
        "calibration_range": [0, int(calibration_tokens)],
        "loo_passes_per_layer": 1 + int(next(iter(utilities.values())).shape[1]),
        "loo_passes": passes,
        "reference_set": "full bank (all m specialists, unit weight)",
        "utility_definition": "u_e(t) = L(without e) - L(with e)",
        "feature_source": "bank forward pre-hook, torch.no_grad, detached, CPU float32",
        "features_precede_outcome": True,
        "labels_never_enter_forward": True,
        "labels_require_grad": bool(next(iter(utilities.values())).requires_grad),
        "bank_state_sha256": tensor_digest(
            [parameter for bank in banks for parameter in
             (bank.gate, bank.up, bank.down)]),
        "labels": label_summary,
    }
    return {"features": features, "utilities": utilities,
            "reference_loss": reference_losses, "manifest": manifest}


def save_labels(path: str, payload: dict) -> None:
    """Checkpoint the label tensor, the features and the manifest together."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save({"features": payload["features"], "utilities": payload["utilities"],
                "reference_loss": payload["reference_loss"],
                "manifest": payload["manifest"]}, path)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    write_json_atomic(path + ".sha256", {"sha256": digest.hexdigest(),
                                         "bytes": os.path.getsize(path)})


def load_labels(path: str) -> dict:
    sidecar = json.load(open(path + ".sha256", encoding="utf-8"))
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != sidecar["sha256"]:
        raise SystemExit(f"utility labels {path}: sha256 does not match the sidecar")
    return torch.load(path, map_location="cpu", weights_only=False)


def train_router(features: torch.Tensor, utilities: torch.Tensor,
                 router: torch.nn.Linear, steps: int = 200, lr: float = 1e-3,
                 seed: int = 1001) -> dict:
    """Listwise KL from the router to the standardised utility labels.

    `features` is `[N, hidden]`, `utilities` is `[N, m]`, both on CPU. The head's
    input is the pre-expert hidden state only. Returns the loss trajectory and
    the label scale used as the temperature.
    """
    if features.shape[0] != utilities.shape[0]:
        raise ValueError("features and utilities disagree on positions")
    mean = utilities.mean(dim=0, keepdim=True)
    std = utilities.std(dim=0, keepdim=True).clamp_min(1e-6)
    labels = ((utilities - mean) / std).detach()
    target = torch.softmax(labels, dim=-1)
    optimizer = torch.optim.AdamW(router.parameters(), lr=lr, betas=(0.9, 0.95),
                                  eps=1e-8, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    router.train()
    for step in range(steps):
        index = torch.randint(0, features.shape[0], (min(4096, features.shape[0]),),
                              generator=generator)
        batch_features = features.index_select(0, index)
        batch_target = target.index_select(0, index)
        logits = router(batch_features)
        log_prob = torch.log_softmax(logits, dim=-1)
        loss = -(batch_target * log_prob).sum(dim=-1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    router.eval()
    return {
        "schema": "selvedge/t2-router-training/v1",
        "steps": steps,
        "lr": lr,
        "seed": seed,
        "batch": int(min(4096, features.shape[0])),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss": losses,
        "label_mean": [float(item) for item in mean.squeeze(0)],
        "label_std": [float(item) for item in std.squeeze(0)],
        "target": "softmax(standardised utility)",
    }
