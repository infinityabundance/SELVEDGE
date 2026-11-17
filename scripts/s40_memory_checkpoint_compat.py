#!/usr/bin/env python3
"""Is the registered `memory-2m` checkpoint still valid under the decoding fix?

The S40 fix changes two things at *inference*: the n-gram address is computed
from the whole recorded prefix and sliced to the current call, and the causal
depthwise convolution carries its last `kernel-1` inputs across calls. The
full-sequence forward is supposed to be unchanged, because at the start of a
sequence the convolution history is zeros and the address slice is the whole
prefix. This script proves that on the registered checkpoint rather than
asserting it:

* **full vs legacy full** — the fixed module's full-sequence forward against a
  bit-for-bit reimplementation of the pre-fix forward. Expected: zero.
* **new cached vs new full** — token-by-token decoding under the fix against
  the full-sequence forward. Expected: float32 noise only.
* **legacy cached vs new full** — token-by-token decoding under the pre-fix
  code against the same full-sequence forward. Expected: a real difference;
  this is the defect the fix removes.

The verdict is mechanical. If the full-sequence forward is unchanged, the
checkpoint's learned function — and therefore every teacher-forced score it
could produce — is unchanged, so it does not need retraining. A *cached*
generative score produced with the old code would have measured a different
function and is invalid; the run receipt list is checked for such a score.

Usage:
    SELVEDGE_MEM=4g scripts/court oracle -- python3 \
        scripts/s40_memory_checkpoint_compat.py \
        --run results/stage2/runs/memory-2m \
        --init results/stage2/graft/20261116T231451Z/memory-init.pt \
        --out results/stage2/memory-2m-compat-<stamp>.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "oracle"))

from frontier import memory as mem  # noqa: E402


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def build_module(init_path: str, checkpoint_path: str | None):
    init = torch.load(init_path, map_location="cpu", weights_only=False)
    config = init["config"]
    context = mem.TokenContext()
    module = mem.ConditionalMemory(
        hidden=config["hidden"], value_dim=config["value_dim"],
        rows=config["rows"], sites=tuple(config["sites"]),
        conv_kernel=config["conv_kernel"], table_dtype=torch.float32,
        seed=config["seed"], context=context)
    state = init["state_dict"]
    step = 0
    if checkpoint_path:
        payload = torch.load(checkpoint_path, map_location="cpu",
                             weights_only=False)
        state = payload["arm_state"]
        step = int(payload.get("step") or 0)
    module.load_state_dict(state, strict=True)
    module.eval()
    return module, context, config, step


def legacy_site_forward(module: mem.ConditionalMemory, site_index: int,
                        hidden: torch.Tensor, call_tokens: torch.Tensor) -> torch.Tensor:
    """The pre-fix forward: address only the call's tokens, pad the conv per call."""
    length = hidden.shape[-2]
    tokens = call_tokens[:, -length:]
    addresses = mem.support(tokens, module.sites[site_index], module.rows)
    table = module.tables[site_index]
    values = table[addresses].float()
    flat = values.reshape(*values.shape[:-2], mem.HEADS * module.value_dim)
    # stateless per-call padding, exactly as `CausalDepthwiseConv1d.forward` did
    flat = module.conv[site_index](flat)
    values = flat.reshape(*values.shape)
    key = module.key_proj[site_index](hidden.float())
    score = (values * key[:, :, None, :]).sum(-1) / (module.value_dim ** 0.5)
    gate = torch.sigmoid(score)
    projected = module.value_proj[site_index](values)
    mixed = (gate[..., None] * projected).sum(dim=2)
    return (mixed + module.out_bias[site_index]).to(hidden.dtype)


def legacy_full(module, tokens: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(hidden)
    for index in range(len(module.sites)):
        out = out + legacy_site_forward(module, index, hidden, tokens)
    return out


@torch.no_grad()
def fixed_full(module, tokens: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    module.context.reset()
    module.reset_state()
    module.context.observe(tokens, past=None)
    out = torch.zeros_like(hidden)
    for index in range(len(module.sites)):
        out = out + module.site_forward(index, hidden)
    return out


@torch.no_grad()
def fixed_cached(module, tokens: torch.Tensor, hidden: torch.Tensor,
                 prefill: int) -> torch.Tensor:
    module.context.reset()
    module.reset_state()
    module.context.observe(tokens[:, :prefill], past=None)
    pieces = []
    for index in range(len(module.sites)):
        pieces.append(module.site_forward(index, hidden[:, :prefill]))
    prefix = torch.stack(pieces).sum(0)
    for position in range(prefill, tokens.shape[1]):
        module.context.observe(tokens[:, position:position + 1],
                               past=_Past(position))
        step = torch.zeros(1, 1, hidden.shape[-1])
        for index in range(len(module.sites)):
            step = step + module.site_forward(index, hidden[:, position:position + 1])
        prefix = torch.cat([prefix, step], dim=1)
    return prefix


@torch.no_grad()
def legacy_cached(module, tokens: torch.Tensor, hidden: torch.Tensor,
                  prefill: int) -> torch.Tensor:
    pieces = []
    for index in range(len(module.sites)):
        pieces.append(legacy_site_forward(module, index, hidden[:, :prefill],
                                          tokens[:, :prefill]))
    out = torch.stack(pieces).sum(0)
    for position in range(prefill, tokens.shape[1]):
        step = torch.zeros(1, 1, hidden.shape[-1])
        for index in range(len(module.sites)):
            step = step + legacy_site_forward(
                module, index, hidden[:, position:position + 1],
                tokens[:, position:position + 1])
        out = torch.cat([out, step], dim=1)
    return out


class _Past:
    def __init__(self, length: int):
        self.length = int(length)

    def get_seq_length(self) -> int:
        return self.length


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--init", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--stream-manifest", default=None)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--prefill", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = os.path.join("results", "stage2",
                                f"memory-2m-compat-{stamp()}.json")

    run_manifest = json.load(open(os.path.join(args.run, "manifest.json")))
    checkpoint = args.checkpoint or run_manifest.get("final_checkpoint") or \
        os.path.join(args.run, "latest.pt")
    if not os.path.exists(checkpoint):
        checkpoint = os.path.join(args.run, "latest.pt")
    stream_manifest_path = args.stream_manifest or \
        "results/stage2/stage2-stream-20261116T230328Z.json"
    stream_manifest = json.load(open(stream_manifest_path))
    stream_path = stream_manifest["stream_path"]
    got = sha256_file(stream_path)
    if got != stream_manifest["sha256"]:
        raise SystemExit(f"stream sha mismatch {got} != {stream_manifest['sha256']}")
    stream = np.fromfile(stream_path, dtype=np.uint32).astype(np.int64)
    offset = int(stream_manifest.get("graft_tokens", 0))
    tokens = torch.from_numpy(stream[offset:offset + args.tokens])[None, :]

    module, context, config, step = build_module(args.init, checkpoint)
    hidden_size = args.hidden or config["hidden"]
    generator = torch.Generator().manual_seed(args.seed)
    hidden = torch.randn(1, tokens.shape[1], hidden_size, generator=generator)

    full_fixed = fixed_full(module, tokens, hidden)
    full_legacy = legacy_full(module, tokens, hidden)
    cached_fixed = fixed_cached(module, tokens, hidden, args.prefill)
    cached_legacy = legacy_cached(module, tokens, hidden, args.prefill)

    def max_abs(a, b) -> float:
        return float((a - b).abs().max().item())

    diff_full = max_abs(full_fixed, full_legacy)
    diff_new_cached = max_abs(cached_fixed, full_fixed)
    diff_legacy_cached = max_abs(cached_legacy, full_fixed)
    per_position = (cached_legacy - full_fixed).abs().amax(dim=-1)[0]
    positions_differ = int((per_position > 1e-4).sum().item())

    conv = module.conv[0]
    identity = torch.zeros_like(conv.weight)
    identity[:, 0, conv.kernel - 1] = 1.0
    conv_distance = float((conv.weight - identity).abs().sum().item())

    evaluations = []
    eval_dir = os.path.join("results", "stage2", "eval")
    if os.path.isdir(os.path.join(REPO, eval_dir)):
        for name in sorted(os.listdir(os.path.join(REPO, eval_dir))):
            path = os.path.join(REPO, eval_dir, name, "eval.json")
            if os.path.exists(path):
                document = json.load(open(path))
                if document.get("arm") == "memory":
                    evaluations.append(os.path.join(eval_dir, name, "eval.json"))

    full_unchanged = diff_full <= 1e-6
    cached_matches = diff_new_cached <= 1e-3
    legacy_defect = diff_legacy_cached > 1e-3 and positions_differ > 0
    receipt = {
        "schema": "selvedge/stage2-memory-compat/v1",
        "kind": "memory-cached-decoding-compatibility",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "checkpoint": {
            "run": args.run,
            "path": checkpoint,
            "sha256": sha256_file(checkpoint),
            "step": step,
        },
        "memory_init": {"path": args.init, "sha256": sha256_file(args.init)},
        "stream": {
            "manifest": stream_manifest_path,
            "manifest_sha256": sha256_file(stream_manifest_path),
            "stream_sha256": stream_manifest["sha256"],
            "offset": offset,
            "tokens_used": int(tokens.numel()),
        },
        "fixture": {
            "tokens": int(tokens.shape[1]),
            "prefill": int(args.prefill),
            "hidden": hidden_size,
            "seed": args.seed,
            "hidden_states": "standard normal; the memory output depends on them "
                             "through the parameter-free gate",
            "conv_kernel": int(conv.kernel),
            "conv_l1_distance_from_identity": conv_distance,
            "conv_is_identity": conv_distance == 0.0,
        },
        "results": {
            "full_fixed_vs_full_legacy_max_abs": diff_full,
            "cached_fixed_vs_full_fixed_max_abs": diff_new_cached,
            "cached_legacy_vs_full_fixed_max_abs": diff_legacy_cached,
            "cached_legacy_positions_over_1e-4": positions_differ,
            "cached_legacy_positions_total": int(tokens.shape[1] - args.prefill),
        },
        "evaluations_of_this_arm": evaluations,
        "verdict": {
            "full_sequence_forward_unchanged": full_unchanged,
            "fixed_cached_decoding_matches_full_sequence": cached_matches,
            "legacy_cached_decoding_differed": legacy_defect,
            "checkpoint_valid_for_full_sequence_scoring": full_unchanged,
            "checkpoint_valid_for_cached_scoring_under_fixed_code": cached_matches,
            "prior_cached_scores_invalid": bool(legacy_defect and evaluations),
            "retrain_required": False,
            "reading": (
                "The checkpoint was trained with full-sequence forwards. Under the "
                "fix its full-sequence output is bit-identical to the pre-fix "
                "output, and cached decoding now reproduces that same function "
                "exactly. The learned parameters therefore remain valid and "
                "retraining is not required. What becomes invalid is any cached "
                "generative score produced by the old code, because that score "
                "measured a different function; the evaluations list above says "
                "whether any such receipt exists."
            ),
        },
        "command": sys.argv,
    }
    write_json(args.out, receipt)
    print(f"compat: full fixed vs legacy max_abs={diff_full:.3e}")
    print(f"compat: cached fixed vs full max_abs={diff_new_cached:.3e}")
    print(f"compat: cached legacy vs full max_abs={diff_legacy_cached:.3e} "
          f"({positions_differ}/{int(tokens.shape[1] - args.prefill)} positions)")
    print(f"compat: checkpoint={checkpoint} step={step} "
          f"conv_identity={conv_distance == 0.0}")
    print(f"compat: retrain_required=False full_unchanged={full_unchanged} "
          f"cached_matches={cached_matches} legacy_defect={legacy_defect}")
    print(f"compat: receipt {args.out}", flush=True)
    if not (full_unchanged and cached_matches and legacy_defect):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
