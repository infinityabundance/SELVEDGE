#!/usr/bin/env python3
"""Full-backbone distillation arm: freeze the tied embedding, train the rest.

The WS19 bar is not a frozen-backbone adapter. This arm distils the whole text
tower of `Qwen/Qwen3.5-0.8B-Base` from the online 2B teacher on the registered
stream. The tied token embedding is frozen:

* text tower: 752,393,024 elements;
* token embedding (= output projection, tied): 254,279,680 elements;
* trained: 498,113,344 elements.

Finetuning changes values, not identities, so the deployed active union is the
whole text tower, 752,393,024 elements. That is what the comparison records.

The state dict exported here is deliberately **only** the trainable parameters,
keyed relative to the text model, so a checkpoint cannot silently carry a frozen
embedding and so the eval driver can reconstruct the arm on top of the pinned
base checkpoint.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import arms


class FullBackboneArm:
    """Trainable view over the text tower with the tied embedding frozen."""

    def __init__(self, model: nn.Module, freeze_embedding: bool = True):
        self.model = model
        self.text_model = arms._text_model(model)
        embedding = model.get_input_embeddings()
        if embedding is None:
            raise RuntimeError("the model has no input embedding")
        embedding_pointer = embedding.weight.data_ptr()

        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in self.text_model.parameters():
            parameter.requires_grad_(True)

        self.frozen_identities: list[str] = []
        if freeze_embedding:
            for name, parameter in self.text_model.named_parameters():
                if parameter.data_ptr() == embedding_pointer:
                    parameter.requires_grad_(False)
                    self.frozen_identities.append(name)
            if not self.frozen_identities:
                raise RuntimeError("could not identify the tied embedding parameter")

        self.text_elements = sum(p.numel() for p in self.text_model.parameters())
        self.frozen_elements = sum(p.numel() for p in self.text_model.parameters()
                                   if not p.requires_grad)
        self.trainable_elements = sum(p.numel() for p in self.text_model.parameters()
                                      if p.requires_grad)
        self.embedding_elements = int(embedding.weight.numel())
        if self.trainable_elements + self.frozen_elements != self.text_elements:
            raise RuntimeError("trainable/frozen partition does not cover the text tower")

    # -- the interface train.py uses ------------------------------------------
    def parameters(self):
        return [parameter for parameter in self.text_model.parameters()
                if parameter.requires_grad]

    def named_trainable(self):
        for name, parameter in self.text_model.named_parameters():
            if parameter.requires_grad:
                yield name, parameter

    def state_dict(self) -> dict:
        # CPU copies: a 498M-element arm must not need a second GPU copy at
        # checkpoint time, and the payload is loaded with map_location="cpu".
        return {name: parameter.detach().to("cpu").clone()
                for name, parameter in self.named_trainable()}

    def load_state_dict(self, state: dict) -> None:
        expected = {name for name, _ in self.named_trainable()}
        if set(state) != expected:
            missing = sorted(expected - set(state))[:5]
            extra = sorted(set(state) - expected)[:5]
            raise ValueError(f"full-arm state mismatch: missing {missing}, extra {extra}")
        with torch.no_grad():
            for name, parameter in self.named_trainable():
                value = state[name].to(device=parameter.device, dtype=parameter.dtype)
                if value.shape != parameter.shape:
                    raise ValueError(f"{name}: {tuple(value.shape)} != "
                                     f"{tuple(parameter.shape)}")
                parameter.copy_(value)

    def to(self, device):
        self.model.to(device)
        return self

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def eval(self):
        self.model.eval()
        return self

    def zeros_like_parameters(self):
        return [torch.zeros_like(parameter) for parameter in self.parameters()]

    # -- accounting -----------------------------------------------------------
    def active_parameter_report(self) -> dict:
        return {
            "kind": "stage2_full_backbone",
            "text_tower_elements": int(self.text_elements),
            "frozen_embedding_elements": int(self.frozen_elements),
            "trainable_elements": int(self.trainable_elements),
            "embedding_elements": int(self.embedding_elements),
            "deployed_active_union_elements": int(self.text_elements),
            "active_union_basis": "finetuning changes values, not identities",
            "frozen_identities": list(self.frozen_identities),
            "tied_output_projection": True,
            "reading": (
                "the tied embedding is frozen, so the output projection is frozen "
                "with it; every other text-tower identity is trained, and the "
                "deployed model reads the full text tower"
            ),
        }


def load_full_state_into(model: nn.Module, state: dict,
                         freeze_embedding: bool = True) -> FullBackboneArm:
    """Build the arm around a base model and load an exported trainable state."""
    arm = FullBackboneArm(model, freeze_embedding=freeze_embedding)
    arm.load_state_dict(state)
    return arm


def bytes_report(text_elements: int, frozen_embedding_elements: int,
                 trainable_elements: int, optimizer_bytes: int,
                 dtype_bytes: int = 2) -> dict:
    """Exact parameter/gradient/optimizer bytes for the arm."""
    return {
        "kind": "stage2_full_backbone_bytes",
        "dtype": "bfloat16",
        "dtype_bytes": dtype_bytes,
        "text_tower_elements": int(text_elements),
        "frozen_embedding_elements": int(frozen_embedding_elements),
        "trainable_elements": int(trainable_elements),
        "student_weights_bytes": int(text_elements * dtype_bytes),
        "frozen_embedding_bytes": int(frozen_embedding_elements * dtype_bytes),
        "trainable_weights_bytes": int(trainable_elements * dtype_bytes),
        "gradients_bytes": int(trainable_elements * dtype_bytes),
        "optimizer_state_bytes": int(optimizer_bytes),
        "total_training_state_bytes": int(
            text_elements * dtype_bytes + trainable_elements * dtype_bytes
            + optimizer_bytes),
    }
