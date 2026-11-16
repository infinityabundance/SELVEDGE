# tiny-qwen3 fixture

A deterministic, tiny Qwen3 checkpoint used by the runtime parity courts. It has
the same architecture shape as the pinned 0.6B control — tied embeddings, RMSNorm,
RoPE, grouped-query attention with a per-head q/k norm, SwiGLU MLP — but two
layers and a 64-token vocabulary, so an entire forward pass can be compared
element by element against an independent implementation.

| file | role |
| --- | --- |
| `generate.py` | generator, standard library only |
| `config.json` | architecture, written by the generator |
| `model.safetensors` | weights, written by the generator |
| `GENERATION.json` | digests and provenance of the two files above |
| `reference-logits.json` | logits from `oracle/qwen3_numpy_reference.py` for a fixed prompt and two greedy steps |

Regenerate (the bytes do not change; the digests in `GENERATION.json` are the
check):

```bash
scripts/court python -- python3 fixtures/tiny-qwen3/generate.py
scripts/court oracle -- python3 oracle/qwen3_numpy_reference.py \
  --model /work/fixtures/tiny-qwen3 --prompt-ids 1,5,9,17 --decode 2 \
  --json /work/fixtures/tiny-qwen3/reference-logits.json
```

Why the weights are deterministic without a library: each tensor's values come
from a SplitMix64 stream seeded by the tensor name, mapped to `[-scale, scale)`
with `scale = 1/sqrt(fan_in)` (gains use a fixed half scale), and the safetensors
header is written by hand with sorted keys. No NumPy, torch or safetensors version
can influence the bytes.

The fixture is generated, so it is committed as data rather than derived at test
time; `crates/selvedge-runtime/tests/tiny_parity.rs` converts it into a real
bundle through the S2 converter and compares the Rust forward against
`reference-logits.json`.
