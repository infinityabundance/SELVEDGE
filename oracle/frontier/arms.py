#!/usr/bin/env python3
"""Stage-2 arms: the memory, its two required controls, and model installation.

Three arms share one injection protocol (a residual added to the output of a
decoder layer):

* `memory`  — the teacher-grafted conditional n-gram table (P19.2);
* `adapter` — the parameter-matched dense control. Its active union is
  **exactly** the memory's (537,600 elements at the registered geometry):
  `2 * 1024 * 262 + 1024`, no gate parameters;
* `ngm`     — the training-free NGM-style control: a causal n-gram encoder over
  the frozen token-embedding space with a cosine gate, zero new parameters.

`base` is the fourth arm: the frozen checkpoint with no hook at all.

The installation helpers keep the backbone a genuine `PreTrainedModel`, so the
pinned evaluation harness can be handed the model object directly (`HFLM`
accepts a pre-initialised model); nothing about the harness changes between
arms.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory import MemoryGeometry, TokenContext


# ---------------------------------------------------------------------------
# controls


class DenseAdapter(nn.Module):
    """Parameter-matched dense residual adapter.

    `rank` is chosen so that `2 * hidden * rank + hidden` equals the memory's
    active union exactly at the registered geometry. The gate is the same
    parameter-free form the memory uses (`sigmoid(<h, z>/sqrt(rank))`), which is
    why the match is exact rather than approximate.
    """

    def __init__(self, hidden: int = 1024, rank: int | None = None,
                 target_active: int | None = None):
        super().__init__()
        self.hidden = int(hidden)
        if rank is None:
            if target_active is None:
                target_active = MemoryGeometry(hidden=hidden).active_per_token_per_site
            rank = max(1, round((target_active - hidden) / (2 * hidden)))
        self.rank = int(rank)
        self.down = nn.Linear(self.hidden, self.rank, bias=False, dtype=torch.float32)
        self.up = nn.Linear(self.rank, self.hidden, bias=False, dtype=torch.float32)
        self.bias = nn.Parameter(torch.zeros(self.hidden, dtype=torch.float32))
        with torch.no_grad():
            nn.init.zeros_(self.up.weight)
        self._scale = self.rank ** 0.5

    @property
    def active_elements(self) -> int:
        return 2 * self.hidden * self.rank + self.hidden

    def active_parameter_report(self) -> dict:
        return {
            "kind": "stage2_dense_adapter_active",
            "hidden": self.hidden,
            "rank": self.rank,
            "active_per_token": self.active_elements,
            "projection_elements": 2 * self.hidden * self.rank,
            "bias_elements": self.hidden,
            "gate_elements": 0,
        }

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states.float()
        z = F.gelu(self.down(x))
        # Parameter-free gate in the adapter's own rank space: the analogue of
        # the memory's <key, row> gate, so the active union is exact.
        gate = torch.sigmoid((z * z).sum(-1, keepdim=True) / self._scale)
        return (gate * self.up(z) + self.bias).to(hidden_states.dtype)


class NGMInjector(nn.Module):
    """Training-free NGM-style n-gram memory over the pretrained embeddings.

    Causal n-gram encoder: for each order the encoder averages the token
    embeddings of the causal window ending at `t`. Multi-order memories are
    combined with fixed equal weights. The cosine-gated injector adds
    `relu(cos(h_t, m_t)) * m_t`. No parameter is introduced or trained.
    """

    def __init__(self, hidden: int = 1024, orders: tuple[int, ...] = (2, 3)):
        super().__init__()
        self.hidden = int(hidden)
        self.orders = tuple(int(order) for order in orders)
        self.embedding: nn.Embedding | None = None
        self.context: TokenContext | None = None

    def bind(self, embedding: nn.Embedding, context: TokenContext) -> None:
        if embedding.embedding_dim != self.hidden:
            raise ValueError("embedding width does not match the injection width")
        self.embedding = embedding
        self.context = context

    @property
    def active_elements(self) -> int:
        """Embedding elements read per token (pre-existing table, no new params)."""
        return sum(self.orders) * self.hidden

    def active_parameter_report(self) -> dict:
        return {
            "kind": "stage2_ngm_active",
            "hidden": self.hidden,
            "orders": list(self.orders),
            "read_elements_per_token": self.active_elements,
            "added_elements": 0,
            "gate_elements": 0,
            "basis": "the embedding table is the backbone's own; it is read, not added",
        }

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.embedding is None or self.context is None:
            raise RuntimeError("NGMInjector.bind must be called before forward")
        length = hidden_states.shape[-2]
        tokens = self.context.require(length).to(torch.int64)
        memories = []
        weight = 0.0
        for order in self.orders:
            window = F.pad(tokens, (order - 1, 0), value=0)
            chunks = [self.embedding(window[:, position:position + length]).float()
                      for position in range(order)]
            memories.append(torch.stack(chunks, dim=2).mean(dim=2))
            weight += 1.0
        memory = sum(memories) / weight
        x = hidden_states.float()
        cosine = F.cosine_similarity(x, memory, dim=-1, eps=1e-6)
        gate = cosine.clamp(min=0.0, max=1.0)
        return (gate[..., None] * memory).to(hidden_states.dtype)


# ---------------------------------------------------------------------------
# installation


def load_model(model_dir: str, dtype=torch.bfloat16, device: str = "cuda"):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=dtype, trust_remote_code=False,
        attn_implementation="eager")
    model.to(device)
    model.eval()
    return model


def freeze_backbone(model: nn.Module) -> int:
    """Freeze every parameter of the core; returns the frozen element count."""
    frozen = 0
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        frozen += parameter.numel()
    return frozen


def _text_model(model: nn.Module):
    core = getattr(model, "model", None)
    return core if core is not None and hasattr(core, "layers") else model


def _layer(model: nn.Module, index: int) -> nn.Module:
    layers = _text_model(model).layers
    if index < 0 or index >= len(layers):
        raise IndexError(f"site {index} outside 0..{len(layers) - 1}")
    return layers[index]


def install_token_context(model: nn.Module, context: TokenContext):
    """Pre-hook that records the causal prefix the layers are about to see."""
    text_model = _text_model(model)

    def pre_hook(module, args, kwargs):
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        context.observe(ids, kwargs.get("past_key_values"))

    return text_model.register_forward_pre_hook(pre_hook, with_kwargs=True)


def install_residual(model: nn.Module, module: nn.Module, sites: tuple[int, ...],
                     site_index_of=None, context: TokenContext | None = None):
    """Add `module(hidden)` to the output of each site layer.

    `site_index_of(site)` maps a layer index to the module's own index (the
    memory has one sub-module per configured site; a single-site adapter uses
    the constant 0). The layer output is a tuple in this architecture; the
    hidden state is element 0.
    """
    handles = []
    if site_index_of is None:
        site_index_of = lambda site: None
    for site in sites:
        layer = _layer(model, site)
        module_index = site_index_of(site)

        def hook(module_, inputs, output, module_index=module_index):
            if isinstance(output, tuple):
                hidden = output[0]
                if module_index is not None:
                    delta = module(hidden, module_index)
                else:
                    delta = module(hidden)
                return (hidden + delta,) + output[1:]
            if module_index is not None:
                return output + module(output, module_index)
            return output + module(output)

        handles.append(layer.register_forward_hook(hook))
    return handles


def install_memory(model: nn.Module, memory) -> list:
    handles = [install_token_context(model, memory.context)]
    handles += install_residual(model, memory, memory.sites,
                                site_index_of=lambda site: memory.sites.index(site))
    return handles


def install_adapter(model: nn.Module, adapter: DenseAdapter, sites: tuple[int, ...],
                    context: TokenContext | None = None) -> list:
    handles = []
    if context is not None:
        handles.append(install_token_context(model, context))
    handles += install_residual(model, adapter, sites, site_index_of=None)
    return handles


def install_ngm(model: nn.Module, ngm: NGMInjector, sites: tuple[int, ...]) -> list:
    if ngm.context is None:
        raise RuntimeError("bind the NGM injector before installing it")
    handles = [install_token_context(model, ngm.context)]
    handles += install_residual(model, ngm, sites, site_index_of=None)
    return handles


def uninstall(handles) -> None:
    for handle in handles:
        handle.remove()
