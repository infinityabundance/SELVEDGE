#!/usr/bin/env python3
"""Tiny fake backbone and checkpoints for the S40 comparison instrument's CPU test.

The shapes are deliberately small and the architecture is positionwise, so a
CPU can run the whole six-arm pipeline in seconds. It exercises the same arm
construction, checkpoint keys, token-context hook and scoring path as the real
models; only the weights are fake.
"""

from __future__ import annotations

import json
import os
import struct
from types import SimpleNamespace

import torch
import torch.nn as nn

from . import arms, full, memory


class TinyBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)

    def forward(self, hidden: torch.Tensor):
        return (hidden + torch.tanh(self.linear(hidden)),)


class TinyText(nn.Module):
    def __init__(self, vocab: int, hidden: int, layers: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([TinyBlock(hidden) for _ in range(layers)])

    def forward(self, input_ids: torch.Tensor, past_key_values=None,
                use_cache: bool = False):
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)[0]
        return hidden


class TinyLM(nn.Module):
    def __init__(self, vocab: int = 64, hidden: int = 32, layers: int = 4):
        super().__init__()
        self.model = TinyText(vocab, hidden, layers)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(self, input_ids: torch.Tensor, past_key_values=None,
                use_cache: bool = False):
        hidden = self.model(input_ids, past_key_values, use_cache)
        return SimpleNamespace(logits=self.lm_head(hidden))


MEMORY_CONFIG = {
    "hidden": 32,
    "value_dim": 8,
    "rows": 64,
    "sites": [2],
    "conv_kernel": 3,
    "seed": 0,
}


def _new_memory(seed: int = 0) -> memory.ConditionalMemory:
    return memory.ConditionalMemory(
        hidden=MEMORY_CONFIG["hidden"], value_dim=MEMORY_CONFIG["value_dim"],
        rows=MEMORY_CONFIG["rows"], sites=tuple(MEMORY_CONFIG["sites"]),
        conv_kernel=MEMORY_CONFIG["conv_kernel"], table_dtype=torch.float32,
        seed=seed)


def write_fake_checkpoints(directory: str, vocabulary: int = 64,
                           hidden: int = 32, layers: int = 4,
                           seed: int = 0, tokens: int = 130,
                           seq_len: int = 32) -> dict:
    """Write adapter, memory and full-arm checkpoints plus a token stream."""
    os.makedirs(directory, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)

    adapter = arms.DenseAdapter(hidden=hidden)
    with torch.no_grad():
        adapter.down.weight.normal_(generator=generator, std=0.05)
        adapter.up.weight.normal_(generator=generator, std=0.05)
        adapter.bias.normal_(generator=generator, std=0.01)
    adapter_path = os.path.join(directory, "adapter.pt")
    torch.save({"schema": "selvedge/stage2-ckpt/v1", "arm": "adapter",
                "step": 1, "tokens_seen": 0,
                "arm_state": adapter.state_dict()}, adapter_path)

    module = _new_memory(seed=0)
    with torch.no_grad():
        # A real graft sets key_proj = B^T, value_proj = B and the table to row
        # means, so the step-0 grafted arm is a non-trivial function. The
        # *ungrafted* module keeps the constructor's zero value projection.
        module.key_proj[0].weight.normal_(generator=generator, std=0.05)
        module.value_proj[0].weight.normal_(generator=generator, std=0.05)
        module.out_bias[0].normal_(generator=generator, std=0.01)
        module.tables[0].normal_(generator=generator, std=0.1)
    init_state = {key: value.detach().clone()
                  for key, value in module.state_dict().items()}
    init = {
        "schema": "selvedge/stage2-memory-init/v1",
        "config": dict(MEMORY_CONFIG),
        "state_dict": init_state,
        "graft": {"graft_tokens": 0, "site": 2},
    }
    init_path = os.path.join(directory, "memory-init.pt")
    torch.save(init, init_path)

    trained = _new_memory(seed=0)
    with torch.no_grad():
        trained.key_proj[0].weight.normal_(generator=generator, std=0.05)
        trained.value_proj[0].weight.normal_(generator=generator, std=0.05)
        trained.out_bias[0].normal_(generator=generator, std=0.01)
        trained.tables[0].normal_(generator=generator, std=0.1)
        trained.conv[0].weight.normal_(generator=generator, std=0.05)
    memory_path = os.path.join(directory, "memory-trained.pt")
    torch.save({"schema": "selvedge/stage2-ckpt/v1", "arm": "memory",
                "step": 1, "tokens_seen": 0,
                "arm_state": trained.state_dict()}, memory_path)

    lm = TinyLM(vocabulary, hidden, layers)
    with torch.no_grad():
        for parameter in lm.parameters():
            parameter.normal_(generator=generator, std=0.05)
    arm = full.FullBackboneArm(lm)
    full_path = os.path.join(directory, "full.pt")
    torch.save({"schema": "selvedge/stage2-ckpt/v1", "arm": "full",
                "step": 1, "tokens_seen": 0,
                "arm_state": arm.state_dict()}, full_path)

    stream = torch.randint(0, vocabulary, (tokens,), generator=generator)
    stream_path = os.path.join(directory, "heldout.u32")
    with open(stream_path, "wb") as handle:
        handle.write(struct.pack(f"<{tokens}I", *stream.tolist()))
    import hashlib
    digest = hashlib.sha256(open(stream_path, "rb").read()).hexdigest()
    manifest = {
        "schema": "selvedge/stage2-stream/v1",
        "tokens": tokens,
        "graft_tokens": 0,
        "stream_path": stream_path,
        "sha256": digest,
        "seq_len": seq_len,
        "note": "fake CPU fixture; never a benchmark",
    }
    manifest_path = os.path.join(directory, "heldout.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return {
        "adapter": adapter_path,
        "memory_init": init_path,
        "memory_trained": memory_path,
        "full": full_path,
        "stream_manifest": manifest_path,
        "stream_path": stream_path,
        "vocab": vocabulary,
        "hidden": hidden,
        "layers": layers,
        "seq_len": seq_len,
    }
