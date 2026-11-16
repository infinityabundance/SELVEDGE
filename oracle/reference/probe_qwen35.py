#!/usr/bin/env python3
"""Probe whether the pinned `transformers` supports Qwen3.5 offline.

Prints the version, the available Qwen3.5 classes, and — if the class exists —
loads the config and reports the facts the reference forward needs. Never
enables `trust_remote_code`.
"""

from __future__ import annotations

import sys

import transformers

MODEL_DIR = (
    "/hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17"
)


def main() -> int:
    print("transformers", transformers.__version__)
    print("torch", __import__("torch").__version__)
    names = sorted(n for n in dir(transformers) if "Qwen3" in n)
    print("qwen3 classes:", names)
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    print("config mapping keys:", [k for k in CONFIG_MAPPING if "qwen" in k])
    try:
        config = transformers.AutoConfig.from_pretrained(
            MODEL_DIR, local_files_only=True, trust_remote_code=False
        )
    except Exception as exc:  # noqa: BLE001 - the point is to print the failure
        print("AUTOCONFIG_FAILED", type(exc).__name__, exc)
        return 1
    print("autoconfig", type(config).__name__, config.model_type)
    text = config.text_config
    print("text_config", type(text).__name__, text.model_type)
    for field in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "tie_word_embeddings",
        "linear_conv_kernel_dim",
        "linear_key_head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_value_head_dim",
        "mtp_num_hidden_layers",
        "mtp_use_dedicated_embeddings",
        "attn_output_gate",
        "full_attention_interval",
    ):
        print(f"  {field} = {getattr(text, field, '<absent>')}")
    print("  layer_types =", list(text.layer_types))
    print("  rope_parameters =", getattr(text, "rope_parameters", "<absent>"))
    print("  rms_norm_eps =", text.rms_norm_eps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
