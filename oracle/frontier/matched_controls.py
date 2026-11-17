#!/usr/bin/env python3
"""S47 matched controls for the stage-2b corrective comparison.

The corrective arm is a token gate over an external n-gram memory. Its active
union is 539,649 elements. A win over the 20M full-distillation bar would still
leave two attribution questions open:

1. **Is the gain the data?** A dense adapter trained on the *same* 20M math
   stream, with the *same* teacher and optimizer budget, can win for the same
   reason the corrective arm does. `MatchedDenseAdapter` is that control. Its
   active added count is exactly 539,649, not the 537,600 of the S39
   `adapter-*.` controls, so a comparison is at equal deployed active union.

2. **Is the gain the gate?** A gate that can read the site's hidden state and
   open a residual is a mechanism independent of what the residual is. The
   `IdentityGatedDenseResidual` control keeps the *same* zero-initialised
   `tanh(w . [h ; r] + b)` gate and replaces the external memory with a dense
   adapter. It is the base bit for bit at init, like the corrective arm, and it
   has the same 539,649-element active union: 537,600 dense elements plus the
   2,049-element gate.

Both classes are installable through `arms.install_adapter`, so the training
and scoring paths already know how to attach them. No GPU work happens here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import arms as arm_install
from .arms import DenseAdapter
from .memory import MemoryGeometry

#: The corrective arm's registered active union: 537,600 memory elements plus
#: 2,049 gate elements. Derived, never typed as a free constant.
CORRECTIVE_ACTIVE = (
    MemoryGeometry().active_per_token_per_site + 2 * MemoryGeometry().hidden + 1)


class MatchedDenseAdapter(nn.Module):
    """A dense adapter with exactly the corrective arm's active union.

    `rank` is chosen so that `2 * hidden * rank + hidden + 1 == target`. At the
    registered geometry that is rank 263 and one scalar output scale, which
    gives 539,649 = 537,600 + 2,049. The extra scalar is a single learned scale,
    not a token gate: this control still answers the data question, not the
    gate question.
    """

    def __init__(self, hidden: int = 1024, target_active: int | None = None):
        super().__init__()
        self.hidden = int(hidden)
        target = int(target_active if target_active is not None
                     else CORRECTIVE_ACTIVE)
        rank = (target - self.hidden - 1) // (2 * self.hidden)
        if rank < 1:
            raise ValueError("target_active is too small for a dense adapter")
        self.rank = int(rank)
        self.down = nn.Linear(self.hidden, self.rank, bias=False,
                              dtype=torch.float32)
        self.up = nn.Linear(self.rank, self.hidden, bias=False,
                            dtype=torch.float32)
        self.bias = nn.Parameter(torch.zeros(self.hidden, dtype=torch.float32))
        self.scale = nn.Parameter(torch.ones((), dtype=torch.float32))
        with torch.no_grad():
            nn.init.zeros_(self.up.weight)
        self._gate_scale = self.rank ** 0.5

    @property
    def active_elements(self) -> int:
        return 2 * self.hidden * self.rank + self.hidden + 1

    def active_parameter_report(self) -> dict:
        return {
            "kind": "stage2_matched_dense_adapter_active",
            "variant": "matched-dense",
            "hidden": self.hidden,
            "rank": self.rank,
            "active_per_token": self.active_elements,
            "projection_elements": 2 * self.hidden * self.rank,
            "bias_elements": self.hidden,
            "scale_elements": 1,
            "gate_elements": 0,
            "corrective_active_union": CORRECTIVE_ACTIVE,
            "match_delta": self.active_elements - CORRECTIVE_ACTIVE,
            "role": "matched-data control: the external memory and the token "
                    "gate are removed, only a dense residual remains",
        }

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states.float()
        z = F.gelu(self.down(x))
        gate = torch.sigmoid((z * z).sum(-1, keepdim=True) / self._gate_scale)
        return (self.scale * (gate * self.up(z)) + self.bias).to(
            hidden_states.dtype)


class IdentityGatedDenseResidual(nn.Module):
    """The corrective gate over a dense adapter instead of an external memory.

    The gate is the corrective arm's gate, with the same feature map
    `phi(h, r) = [h ; r]`, the same `2 * hidden + 1` elements, and the same
    zero initialisation. The read is a 537,600-element dense adapter. At init
    `alpha_t = tanh(0) = 0`, so the injected model is the base bit for bit.
    """

    def __init__(self, hidden: int = 1024, dense_rank: int | None = None):
        super().__init__()
        self.hidden = int(hidden)
        if dense_rank is None:
            dense_rank = (MemoryGeometry(hidden=hidden).active_per_token_per_site
                          - hidden) // (2 * hidden)
        self.dense = DenseAdapter(hidden=hidden, rank=int(dense_rank))
        self.gate_w = nn.Parameter(torch.zeros(2 * self.hidden))
        self.gate_b = nn.Parameter(torch.zeros(1))
        self.last_gate: torch.Tensor | None = None

    @property
    def gate_parameters(self) -> int:
        return 2 * self.hidden + 1

    @property
    def active_elements(self) -> int:
        return self.dense.active_elements + self.gate_parameters

    def gate(self, hidden_states: torch.Tensor,
             read: torch.Tensor) -> torch.Tensor:
        features = torch.cat([hidden_states.float(), read.float()], dim=-1)
        logits = (features * self.gate_w.float()).sum(-1) + self.gate_b.float()
        return torch.tanh(logits)

    def identity_at_init(self) -> bool:
        return (float(self.gate_w.detach().abs().sum()) == 0.0
                and float(self.gate_b.detach().abs().sum()) == 0.0)

    def read(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dense(hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        read = self.read(hidden_states)
        alpha = self.gate(hidden_states, read)
        self.last_gate = alpha
        return (alpha[..., None] * read).to(hidden_states.dtype)

    def active_parameter_report(self) -> dict:
        return {
            "kind": "stage2_identity_gated_dense_active",
            "variant": "identity-gated-dense",
            "hidden": self.hidden,
            "dense_rank": self.dense.rank,
            "dense_elements": self.dense.active_elements,
            "gate_elements": self.gate_parameters,
            "gate_feature_map": "phi(h_t, r_t) = [h_t ; r_t] (parameter-free)",
            "gate_init": "w = 0 and b = 0, so alpha_t = tanh(0) = 0 exactly",
            "active_per_token": self.active_elements,
            "corrective_active_union": CORRECTIVE_ACTIVE,
            "match_delta": self.active_elements - CORRECTIVE_ACTIVE,
            "role": "gate-only control: the same token gate with no external "
                    "memory, so a gain cannot be attributed to the gate alone",
        }


#: The two registered control kinds, as they will be named by a training run.
CONTROL_KINDS = ("adapter-matched", "gated-dense")


def build_control(kind: str, args, device: str):
    """Build and install one control; the hook a training loop calls.

    `args` needs `hidden` and `site`. This is deliberately side-effect free: it
    does not load a checkpoint, so a fresh run and a scoring run share the same
    construction path. A caller that resumes loads `arm_state` afterwards, as
    `oracle/frontier/train.py` already does for every arm.
    """
    if kind not in CONTROL_KINDS:
        raise ValueError(f"unknown matched control {kind!r}")
    if kind == "adapter-matched":
        module = MatchedDenseAdapter(hidden=args.hidden).to(device)
    else:
        module = IdentityGatedDenseResidual(hidden=args.hidden).to(device)
    handles = arm_install.install_adapter(
        args.model, module, (args.site,)) if getattr(args, "model", None) else []
    report = module.active_parameter_report()
    if kind == "gated-dense":
        report["identity_at_init"] = module.identity_at_init()
    return module, handles, report
