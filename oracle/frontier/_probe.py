#!/usr/bin/env python3
"""S37 scratch probe: module layout and a one-token forward. Deleted before commit."""
import json
import os
import sys

import torch
from transformers import AutoTokenizer

ROOT = os.environ.get("HF_HUB_CACHE", "/hf-cache/hub")
MODEL = ("/hf-cache/models/Qwen/Qwen3.5-0.8B-Base/"
         "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")


def main() -> int:
    import transformers
    print("transformers", transformers.__version__, "torch", torch.__version__)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, low_cpu_mem_usage=True,
        trust_remote_code=False)
    print("class", type(model).__name__)
    for name, module in model.named_children():
        print("child", name, type(module).__name__)
    text = getattr(model, "model", None)
    if text is not None:
        for name, module in text.named_children():
            print("  text child", name, type(module).__name__)
        layers = getattr(text, "layers", None)
        print("  layers", len(layers) if layers is not None else None,
              type(layers[0]).__name__ if layers is not None else None)
        if layers is not None:
            print("  layer0 children", [n for n, _ in layers[0].named_children()])
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)
    ids = tok("The capital of France is", return_tensors="pt").input_ids
    print("ids", ids.tolist(), "vocab", len(tok))
    out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    print("logits", tuple(out.logits.shape), "hidden",
          len(out.hidden_states), tuple(out.hidden_states[0].shape))
    return 0


if __name__ == "__main__":
    sys.exit(main())
