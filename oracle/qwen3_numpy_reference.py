#!/usr/bin/env python3
"""Independent NumPy f32 Qwen3 forward, used as the runtime's parity oracle.

It shares no code with the Rust runtime: it reads the same checkpoint with its
own safetensors parser, implements the published Qwen3 equations in float32
NumPy, and writes logits in the same raw format `selvedge run --dump-logits`
produces, so the two can be compared element by element.

Two output forms:

* `--json` writes a JSON document (used for the tiny fixture, whose vocabulary is
  64 tokens and which is committed as a fixture);
* otherwise the script writes a raw little-endian f32 file of shape
  `[positions][vocab]`, which `selvedge run --compare-logits` reads.

Usage:

    # tiny fixture (committed reference)
    scripts/court oracle -- python3 oracle/qwen3_numpy_reference.py \
        --model /work/fixtures/tiny-qwen3 --prompt-ids 1,5,9,17 --decode 2 \
        --json /work/fixtures/tiny-qwen3/reference-logits.json

    # pinned 0.6B control, four prompt tokens, no continuation
    scripts/court oracle -- python3 oracle/qwen3_numpy_reference.py \
        --model /hf-cache/models/Qwen/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca \
        --prompt-ids 785,6722,315,374 --decode 0 \
        --bin /work/results/runtime/qwen3-0.6b-ref-logits.bin \
        --meta /work/results/runtime/qwen3-0.6b-ref-meta.json

Reduction-order honesty: NumPy's `matmul` uses its own pairwise/vectorised
order, and the Rust kernels use eight interleaved f32 accumulators, so the two
agree to a tolerance rather than bit for bit. `docs/RUNTIME.md` states the
policy; the observed maxima are recorded in the receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

SCHEMA = "selvedge/qwen3-numpy-reference/v1"


def load_safetensors(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
        data = handle.read()
    tensors: dict[str, np.ndarray] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        begin, end = meta["data_offsets"]
        dtype = meta["dtype"]
        if dtype == "F32":
            values = np.frombuffer(data[begin:end], dtype="<f4").astype(np.float32)
        elif dtype == "BF16":
            raw = np.frombuffer(data[begin:end], dtype="<u2").astype(np.uint32)
            values = (raw << 16).view(np.float32).astype(np.float32)
        elif dtype == "F16":
            values = np.frombuffer(data[begin:end], dtype="<f2").astype(np.float32)
        else:
            raise SystemExit(f"{name} has unsupported dtype {dtype}")
        tensors[name] = values.reshape(tuple(meta["shape"]))
    return tensors


def load_checkpoint(path: Path) -> dict[str, np.ndarray]:
    if path.is_file():
        return load_safetensors(path)
    merged: dict[str, np.ndarray] = {}
    for shard in sorted(path.glob("*.safetensors")):
        for key, value in load_safetensors(shard).items():
            if key in merged:
                raise SystemExit(f"duplicate tensor {key} across checkpoint shards")
            merged[key] = value
    if not merged:
        raise SystemExit(f"no .safetensors files under {path}")
    return merged


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    mean_square = np.mean(np.square(x, dtype=np.float32), dtype=np.float32)
    scale = np.float32(1.0) / np.sqrt(mean_square + np.float32(eps), dtype=np.float32)
    return (x * scale * weight).astype(np.float32)


def rope_tables(head_dim: int, theta: float, positions: int) -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(0, head_dim, 2, dtype=np.float32)
    exponent = index / np.float32(head_dim)
    inv_freq = (np.float32(1.0) / (np.float32(theta) ** exponent)).astype(np.float32)
    steps = np.arange(positions, dtype=np.float32)
    angles = np.outer(steps, inv_freq).astype(np.float32)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """`x` is `[heads, t, head_dim]`; `cos`/`sin` are `[t, head_dim/2]`."""
    half = x.shape[-1] // 2
    low = x[..., :half]
    high = x[..., half:]
    cos_b = cos[None, :, :]
    sin_b = sin[None, :, :]
    out = np.empty_like(x)
    out[..., :half] = low * cos_b - high * sin_b
    out[..., half:] = high * cos_b + low * sin_b
    return out


class Qwen3Reference:
    """Qwen3 in float32 NumPy, one token at a time, with a per-layer KV cache."""

    def __init__(self, config: dict, tensors: dict[str, np.ndarray]) -> None:
        if config.get("model_type") != "qwen3":
            raise SystemExit(f"model_type {config.get('model_type')} is not qwen3")
        if not config.get("tie_word_embeddings", False):
            raise SystemExit("only tied embeddings are implemented")
        self.t = tensors
        self.hidden = int(config["hidden_size"])
        self.intermediate = int(config["intermediate_size"])
        self.heads = int(config["num_attention_heads"])
        self.kv_heads = int(config["num_key_value_heads"])
        self.head_dim = int(config["head_dim"])
        self.group = self.heads // self.kv_heads
        self.layers = int(config["num_hidden_layers"])
        self.vocab = int(config["vocab_size"])
        self.eps = float(config["rms_norm_eps"])
        self.scale = np.float32(1.0 / np.sqrt(self.head_dim))
        capacity = int(config.get("max_position_embeddings", 4096))
        self.cos, self.sin = rope_tables(self.head_dim, float(config["rope_theta"]), capacity)

    def forward(self, token: int, position: int, cache: dict) -> np.ndarray:
        x = self.t["model.embed_tokens.weight"][token].astype(np.float32)
        for layer in range(self.layers):
            prefix = f"model.layers.{layer}"
            residual = x
            h = rms_norm(x, self.t[f"{prefix}.input_layernorm.weight"], self.eps)
            q = (self.t[f"{prefix}.self_attn.q_proj.weight"] @ h).astype(np.float32)
            k = (self.t[f"{prefix}.self_attn.k_proj.weight"] @ h).astype(np.float32)
            v = (self.t[f"{prefix}.self_attn.v_proj.weight"] @ h).astype(np.float32)
            q = q.reshape(self.heads, self.head_dim)
            k = k.reshape(self.kv_heads, self.head_dim)
            v = v.reshape(self.kv_heads, self.head_dim)
            q = np.stack(
                [
                    rms_norm(
                        q[head], self.t[f"{prefix}.self_attn.q_norm.weight"], self.eps
                    )
                    for head in range(self.heads)
                ]
            )
            k = np.stack(
                [
                    rms_norm(
                        k[head], self.t[f"{prefix}.self_attn.k_norm.weight"], self.eps
                    )
                    for head in range(self.kv_heads)
                ]
            )
            q = q[:, None, :]
            k = k[:, None, :]
            v = v[:, None, :]
            window = slice(position, position + 1)
            q = apply_rope(q, self.cos[window], self.sin[window])
            k = apply_rope(k, self.cos[window], self.sin[window])
            layer_cache = cache.setdefault(layer, {"k": [], "v": []})
            layer_cache["k"].append(k[:, 0, :])
            layer_cache["v"].append(v[:, 0, :])
            keys = np.stack(layer_cache["k"], axis=1)
            values = np.stack(layer_cache["v"], axis=1)
            query = q[:, 0, :]
            attended = np.zeros((self.heads, self.head_dim), dtype=np.float32)
            for head in range(self.heads):
                kv_head = head // self.group
                scores = (keys[kv_head] @ query[head]).astype(np.float32) * self.scale
                scores = scores - scores.max()
                weights = np.exp(scores).astype(np.float32)
                weights = weights / np.sum(weights, dtype=np.float32)
                attended[head] = (weights @ values[kv_head]).astype(np.float32)
            o = (
                self.t[f"{prefix}.self_attn.o_proj.weight"] @ attended.reshape(-1)
            ).astype(np.float32)
            x = residual + o
            residual = x
            h = rms_norm(x, self.t[f"{prefix}.post_attention_layernorm.weight"], self.eps)
            gate = (self.t[f"{prefix}.mlp.gate_proj.weight"] @ h).astype(np.float32)
            up = (self.t[f"{prefix}.mlp.up_proj.weight"] @ h).astype(np.float32)
            act = (gate / (np.float32(1.0) + np.exp(-gate)) * up).astype(np.float32)
            mlp = (self.t[f"{prefix}.mlp.down_proj.weight"] @ act).astype(np.float32)
            x = residual + mlp
        x = rms_norm(x, self.t["model.norm.weight"], self.eps)
        return (self.t["model.embed_tokens.weight"] @ x).astype(np.float32)


def digest_logits(logits: list[list[float]]) -> str:
    digest = hashlib.sha256()
    for position in logits:
        for value in position:
            digest.update(struct.pack("<f", value))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-ids", required=True, help="comma-separated token ids")
    parser.add_argument("--decode", type=int, default=0, help="greedy continuation steps")
    parser.add_argument("--json", help="write a JSON document (small vocabularies)")
    parser.add_argument("--bin", help="write raw little-endian f32 [positions][vocab]")
    parser.add_argument("--meta", help="write a small JSON receipt next to --bin")
    args = parser.parse_args()

    model_dir = Path(args.model)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    tensors = load_checkpoint(model_dir)
    model = Qwen3Reference(config, tensors)

    prompt_ids = [int(part) for part in args.prompt_ids.split(",") if part.strip()]
    cache: dict = {}
    logits: list[list[float]] = []
    position = 0
    for token in prompt_ids:
        scores = model.forward(token, position, cache)
        logits.append([float(value) for value in scores])
        position += 1
    generated: list[int] = []
    next_token = int(np.argmax(np.asarray(logits[-1], dtype=np.float32)))
    for _ in range(args.decode):
        generated.append(next_token)
        scores = model.forward(next_token, position, cache)
        logits.append([float(value) for value in scores])
        position += 1
        next_token = int(np.argmax(scores))

    weights_path = model_dir / "model.safetensors"
    meta = {
        "schema": SCHEMA,
        "fixture_model_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest()
        if weights_path.exists()
        else None,
        "model": str(model_dir),
        "prompt_ids": prompt_ids,
        "generated_ids": generated,
        "positions": len(logits),
        "vocab": int(config["vocab_size"]),
        "dtype": "float32",
        "implementation": "numpy f32; np.matmul uses NumPy's own reduction order",
        "numpy_version": np.__version__,
        "logits_sha256": digest_logits(logits),
        "note": (
            "Reference logits for the SELVEDGE runtime parity court; regenerate with "
            "`scripts/court oracle -- python3 oracle/qwen3_numpy_reference.py`."
        ),
    }

    if args.bin:
        out = Path(args.bin)
        out.parent.mkdir(parents=True, exist_ok=True)
        flat = np.asarray(logits, dtype="<f4").reshape(-1)
        out.write_bytes(flat.tobytes())
        meta["bin"] = str(out)
        meta["bin_bytes"] = out.stat().st_size
    if args.meta:
        meta_out = Path(args.meta)
        meta_out.parent.mkdir(parents=True, exist_ok=True)
        meta_out.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        document = dict(meta)
        document["logits"] = logits
        out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({k: v for k, v in meta.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
