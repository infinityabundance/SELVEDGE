#!/usr/bin/env python3
"""Generate the deterministic tiny Qwen3 checkpoint used by the runtime courts.

The fixture is a *tiny* Qwen3 with the same architecture shape as the pinned
0.6B control (tied embeddings, RMSNorm, RoPE, GQA with a per-head q/k norm,
SwiGLU MLP) but two layers and a 64-token vocabulary, so the whole forward pass
can be compared against an independent NumPy implementation element by element.

Everything here is deterministic and standard-library only:

* weights come from SplitMix64 seeded per tensor name. The stream is fixed, the
  mapping from 64-bit state to float is fixed, and the write order is sorted by
  key, so the output bytes are identical on every run and every machine;
* the safetensors header is written by hand (little-endian, sorted keys, no
  metadata), so no library version can change the bytes.

Usage (inside the python court):

    scripts/court python -- python3 fixtures/tiny-qwen3/generate.py

The committed files are `config.json` and `model.safetensors`. See
`fixtures/tiny-qwen3/README.md` for the digests and the regeneration command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

MASK = (1 << 64) - 1

CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "head_dim": 8,
    "hidden_act": "silu",
    "hidden_size": 32,
    "initializer_range": 0.02,
    "intermediate_size": 64,
    "max_position_embeddings": 64,
    "model_type": "qwen3",
    "num_attention_heads": 4,
    "num_hidden_layers": 2,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-06,
    "rope_scaling": None,
    "rope_theta": 10000.0,
    "sliding_window": None,
    "tie_word_embeddings": True,
    "torch_dtype": "float32",
    "transformers_version": "4.51.0",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 64,
}

SEED = 0x05E1DE6E0BABE001


def splitmix64(state: int) -> int:
    z = (state + 0x9E3779B97F4A7C15) & MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


def stream_for(name: str) -> int:
    state = SEED
    for byte in name.encode("utf-8"):
        state = splitmix64(state ^ byte)
    return state


def weights(name: str, shape: tuple[int, ...], scale: float) -> bytes:
    """`prod(shape)` values in `[-scale, scale)`, deterministic in `name`."""
    state = stream_for(name)
    count = 1
    for dim in shape:
        count *= dim
    values = []
    for _ in range(count):
        state = splitmix64(state)
        # Top 24 bits into [0, 1), then to [-scale, scale).
        unit = (state >> 40) / float(1 << 24)
        values.append((unit * 2.0 - 1.0) * scale)
    return struct.pack("<" + "f" * count, *values)


def tensor_table(config: dict) -> list[tuple[str, tuple[int, ...], float]]:
    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    vocab = config["vocab_size"]
    q_dim = heads * head_dim
    kv_dim = kv_heads * head_dim
    layers = config["num_hidden_layers"]
    inv_hidden = 1.0 / (hidden**0.5)
    inv_q = 1.0 / (q_dim**0.5)
    inv_inter = 1.0 / (intermediate**0.5)

    table: list[tuple[str, tuple[int, ...], float]] = [
        ("model.embed_tokens.weight", (vocab, hidden), inv_hidden),
        ("model.norm.weight", (hidden,), 0.5),
    ]
    for layer in range(layers):
        prefix = f"model.layers.{layer}"
        table.extend(
            [
                (f"{prefix}.input_layernorm.weight", (hidden,), 0.5),
                (f"{prefix}.post_attention_layernorm.weight", (hidden,), 0.5),
                (f"{prefix}.self_attn.q_proj.weight", (q_dim, hidden), inv_hidden),
                (f"{prefix}.self_attn.k_proj.weight", (kv_dim, hidden), inv_hidden),
                (f"{prefix}.self_attn.v_proj.weight", (kv_dim, hidden), inv_hidden),
                (f"{prefix}.self_attn.o_proj.weight", (hidden, q_dim), inv_q),
                (f"{prefix}.self_attn.q_norm.weight", (head_dim,), 0.5),
                (f"{prefix}.self_attn.k_norm.weight", (head_dim,), 0.5),
                (
                    f"{prefix}.mlp.gate_proj.weight",
                    (intermediate, hidden),
                    inv_hidden,
                ),
                (f"{prefix}.mlp.up_proj.weight", (intermediate, hidden), inv_hidden),
                (f"{prefix}.mlp.down_proj.weight", (hidden, intermediate), inv_inter),
            ]
        )
    table.sort(key=lambda item: item[0])
    return table


def write_safetensors(path: Path, table: list[tuple[str, tuple[int, ...], float]]) -> None:
    header: dict[str, dict] = {}
    chunks: list[bytes] = []
    offset = 0
    for name, shape, scale in table:
        data = weights(name, shape, scale)
        header[name] = {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [offset, offset + len(data)],
        }
        chunks.append(data)
        offset += len(data)
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    # The format pads the header so the data section starts 8-byte aligned.
    header_bytes += b" " * ((-len(header_bytes)) % 8)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        for chunk in chunks:
            handle.write(chunk)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="fixtures/tiny-qwen3")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "config.json"
    config_path.write_text(
        json.dumps(CONFIG, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    table = tensor_table(CONFIG)
    model_path = out / "model.safetensors"
    write_safetensors(model_path, table)

    elements = 0
    for _, shape, _ in table:
        count = 1
        for dim in shape:
            count *= dim
        elements += count
    report = {
        "schema": "selvedge/tiny-qwen3/v1",
        "config_sha256": sha256(config_path),
        "model_sha256": sha256(model_path),
        "model_bytes": model_path.stat().st_size,
        "tensors": len(table),
        "elements": elements,
        "seed": f"0x{SEED:016x}",
        "generator": "fixtures/tiny-qwen3/generate.py",
        "note": "SplitMix64-seeded weights; no library call affects the bytes",
    }
    (out / "GENERATION.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
