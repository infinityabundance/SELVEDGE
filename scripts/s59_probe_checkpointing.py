#!/usr/bin/env python3
"""S59 probe: does activation checkpointing actually engage on Qwen3.5-0.8B?

The stage-2b smoke OOMs at the device's free-memory ceiling with and without
`--gradient-checkpointing`, and the peak is byte-identical, which means the
switch is a no-op. transformers 5.x renamed the call; this probe asks the loaded
model directly whether the decoder layers are checkpointed, so the answer is a
property of the object rather than of the warning text.

CPU-only prints; the model is loaded to the GPU only to read its module flags.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "oracle"))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-dir", required=True)
    args = parser.parse_args(argv)

    import torch
    from frontier import arms

    model = arms.load_model(args.student_dir, dtype=torch.bfloat16,
                           device="cpu")
    report = {
        "class": type(model).__name__,
        "has_activation_checkpointing_enable": hasattr(
            model, "activation_checkpointing_enable"),
        "has_gradient_checkpointing_enable": hasattr(
            model, "gradient_checkpointing_enable"),
        "is_gradient_checkpointing_before": bool(
            getattr(model, "is_gradient_checkpointing", False)),
    }
    called = None
    if hasattr(model, "activation_checkpointing_enable"):
        model.activation_checkpointing_enable()
        called = "activation_checkpointing_enable"
    else:
        model.gradient_checkpointing_enable()
        called = "gradient_checkpointing_enable"
    report["called"] = called
    report["is_gradient_checkpointing_after"] = bool(
        getattr(model, "is_gradient_checkpointing", False))
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    report["inner_class"] = type(inner).__name__ if inner is not None else None
    report["layers"] = len(layers) if layers is not None else None
    if layers:
        layer = layers[0]
        report["layer_class"] = type(layer).__name__
        report["layer_gradient_checkpointing_attr"] = bool(
            getattr(layer, "gradient_checkpointing", False))
        report["layer_has_set_gc"] = hasattr(layer, "_set_gradient_checkpointing")
    report["text_config_gradient_checkpointing"] = bool(
        getattr(getattr(model, "config", None), "gradient_checkpointing", False))
    report["use_cache"] = bool(getattr(getattr(model, "config", None),
                                       "use_cache", True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
