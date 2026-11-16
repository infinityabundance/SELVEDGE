#!/usr/bin/env python3
"""P11.1 — causal bottleneck repair: bounded intervention, predictor, controls.

The court asks one question: **is an intervention that is causally located and
bounded better than an equal-budget adaptation that is not?** It is built from
four parts, each usable on CPU:

1. **A student protocol.** `logits(input_ids, intervention)` and
   `hidden_state(input_ids, layer)`; an intervention is one hidden-space vector
   `delta` added to the *last position* of layer `layer`'s output. `TinyLayeredStudent`
   is the CPU fixture; `HFCausalStudent` is the frozen 0.6B adapter with a
   forward hook. The fixture is a layered deterministic student (embedding, N
   residual blocks, linear readout) with **one** non-identity block: it deletes
   the class subspace at a planted layer. That isolation is deliberate and is
   stated wherever a fixture number is reported.

2. **The offline oracle.** For each failed item and each candidate layer, search
   `delta` with `||delta||_2 <= epsilon`, either by the gradient of the
   differentiable surrogate (the verified loss) or by verifier-driven search
   (a fixed identity-basis plus seeded random directions, scored by the verifier
   only). The *bottleneck layer* is defined as the **earliest** layer at which
   a bounded correction flips the verified outcome: correcting later can bypass
   the cause, so the earliest success is the causal one.

3. **The learned predictor.** `C_theta(h_layer, z_c) -> delta_hat`, where `h` is
   the student's own hidden state at the layer and `z_c` is a code from the
   input tokens. Labels and verifier outcomes are **never** inputs; a test
   changes the label of an item and asserts the prediction is unchanged. The
   predictor is trained to reproduce the oracle's correction on a training split.

4. **The controls.** An equal-total-parameter distributed LoRA (same item set,
   same token count, same steps), an all-layer adapter bank (a declared,
   unmatched reference), and no adaptation. The equal-budget LoRA is the
   comparison that makes "causal" mean something.

Everything below the GPU entry point is CPU-testable with known answers; the
tests are `oracle/tests/test_p11_causal_repair.py`. The 0.6B run is
pre-registered in `docs/EXPERIMENT-CONTRACT.md` (P11.1) and launches with
`results/T2-PLAN.md` §3.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.dirname(HERE)
REPO = os.path.dirname(ORACLE)

SCHEMA_FIXTURE = "selvedge/p11-causal-repair-fixture/v1"
SCHEMA_RUN = "selvedge/p11-causal-repair-run/v1"
#: Registered intervention budget: a fraction of the item set's median hidden
#: norm at the candidate layer. It is scale-free, so the same constant applies
#: to the fixture and to the 0.6B student.
KAPPA_REPAIR = 0.10
ORACLE_CAPTURE_KILL = 0.50
TRANSFER_CAPTURE_KILL = 0.25
ORACLE_STEPS = 150
ORACLE_LR = 0.5
PREDICTOR_STEPS = 300
PREDICTOR_LR = 0.05
LORA_STEPS = 300
LORA_LR = 0.05
ALL_LAYER_RANK = 8
SEED = 1001
CANDIDATE_LAYERS_06B = "all 28 core layers (0-based, output of layer l)"


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json_atomic(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# items


class RepairItem:
    """One verified failure: the tokens, the correct outcome, and nothing else."""

    __slots__ = ("id", "input_ids", "label", "structure")

    def __init__(self, id: str, input_ids: list[int] | torch.Tensor, label: int,
                 structure: str = "default"):
        self.id = str(id)
        self.input_ids = (input_ids if isinstance(input_ids, torch.Tensor)
                          else torch.tensor(input_ids, dtype=torch.long))
        self.label = int(label)
        self.structure = structure

    def as_dict(self) -> dict:
        return {"id": self.id, "input_ids": [int(value) for value in self.input_ids],
                "label": self.label, "structure": self.structure}


def median_hidden_norm(student, items: list[RepairItem], layer: int) -> float:
    norms = []
    with torch.no_grad():
        for item in items:
            hidden = student.hidden_state(item.input_ids.unsqueeze(0), layer)
            norms.append(float(hidden.reshape(-1, hidden.shape[-1])[-1].norm()))
    norms.sort()
    return norms[len(norms) // 2]


def repair_budget(student, items: list[RepairItem], layer: int,
                  kappa: float = KAPPA_REPAIR, absolute: float | None = None) -> float:
    if absolute is not None:
        return float(absolute)
    return float(kappa * median_hidden_norm(student, items, layer))


# ---------------------------------------------------------------------------
# the CPU fixture: a layered deterministic student with a planted bottleneck


class TinyLayeredStudent(nn.Module):
    """Embedding -> N residual blocks -> linear readout of the class subspace.

    Exactly one block is non-identity: at `bottleneck` it removes the class
    subspace (the first `classes` coordinates). Before that layer no bounded
    correction can survive; at that layer a correction in the class subspace
    reaches the readout; later layers are identity, so the earliest successful
    layer is the planted block. The fixture isolates the causal property; it is
    not a transformer and no attention is involved.
    """

    def __init__(self, hidden: int = 8, classes: int = 4, vocab: int = 13,
                 layers: int = 4, bottleneck: int = 1, seed: int = 1001):
        super().__init__()
        if not 0 <= bottleneck < layers:
            raise ValueError("bottleneck must be a layer index")
        if classes + 1 > vocab:
            raise ValueError("every class needs a class token")
        self.hidden = hidden
        self.classes = classes
        self.vocab = vocab
        self.layers = layers
        self.bottleneck = bottleneck
        self.embed = nn.Embedding(vocab, hidden)
        with torch.no_grad():
            self.embed.weight.zero_()
            for index in range(1, classes + 1):
                row = torch.zeros(hidden)
                row[index - 1] = 1.0
                self.embed.weight[index].copy_(row)

    # -- protocol ------------------------------------------------------------
    def candidate_layers(self) -> list[int]:
        return list(range(self.layers))

    def _run(self, input_ids: torch.Tensor, intervention=None):
        hidden = self.embed(input_ids)
        layer_index, delta = intervention if intervention is not None else (None, None)
        for index in range(self.layers):
            if index == self.bottleneck:
                hidden = hidden.clone()
                hidden[..., :self.classes] = 0.0
            if index == layer_index:
                if delta is None:
                    raise ValueError("intervention layer given without a delta")
                hidden = hidden.clone()
                hidden[:, -1, :] = hidden[:, -1, :] + delta.reshape(1, -1)
        return hidden

    def logits(self, input_ids: torch.Tensor, intervention=None) -> torch.Tensor:
        hidden = self._run(input_ids, intervention)
        pooled = hidden.mean(dim=1)
        return pooled[:, :self.classes]

    def hidden_state(self, input_ids: torch.Tensor, layer: int) -> torch.Tensor:
        hidden = self.embed(input_ids)
        for index in range(layer + 1):
            if index == self.bottleneck:
                hidden = hidden.clone()
                hidden[..., :self.classes] = 0.0
        return hidden


class Verifier:
    """`argmax == label` plus a continuous margin for verifier-driven search."""

    def evaluate(self, logits: torch.Tensor, label: int) -> tuple[bool, float]:
        row = logits.reshape(-1)
        correct = int(torch.argmax(row)) == int(label)
        log_prob = float(F.log_softmax(row, dim=-1)[label])
        return bool(correct), (1.0 if correct else 0.0) + 1e-3 * log_prob


def verify(student, item: RepairItem, intervention=None,
           verifier: Verifier | None = None) -> dict:
    verifier = verifier or Verifier()
    with torch.no_grad():
        logits = student.logits(item.input_ids.unsqueeze(0), intervention)
    correct, score = verifier.evaluate(logits, item.label)
    loss = float(F.cross_entropy(logits.reshape(1, -1),
                                 torch.tensor([item.label],
                                              device=logits.device)))
    return {"correct": correct, "score": score, "loss": loss}


# ---------------------------------------------------------------------------
# the offline bounded-intervention oracle


def _project(delta: torch.Tensor, epsilon: float) -> torch.Tensor:
    norm = delta.detach().norm()
    if float(norm) <= epsilon:
        return delta
    return delta * (epsilon / norm.clamp_min(1e-12))


def surrogate_search(student, item: RepairItem, layer: int, epsilon: float,
                     steps: int = ORACLE_STEPS, lr: float = ORACLE_LR,
                     verifier: Verifier | None = None) -> dict:
    """Gradient of the verified loss on `delta`, projected onto the L2 ball."""
    verifier = verifier or Verifier()
    before = verify(student, item, None, verifier)
    delta = torch.zeros(student.hidden, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)
    trajectory = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = student.logits(item.input_ids.unsqueeze(0), (layer, delta))
        loss = F.cross_entropy(logits.reshape(1, -1),
                               torch.tensor([item.label]))
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            projected = _project(delta.detach(), epsilon)
            delta.copy_(projected)
        if (step + 1) % 25 == 0:
            trajectory.append(float(loss.detach()))
    after = verify(student, item, (layer, delta.detach()), verifier)
    return {"layer": layer, "variant": "surrogate", "before": before, "after": after,
            "success": after["correct"], "delta": delta.detach(),
            "delta_norm": float(delta.detach().norm()), "epsilon": epsilon,
            "trajectory": trajectory, "steps": steps}


def verifier_search(student, item: RepairItem, layer: int, epsilon: float,
                    directions: torch.Tensor, scales: tuple[float, ...] = (0.25, 0.5,
                                                                        0.75, 1.0),
                    verifier: Verifier | None = None) -> dict:
    """Verifier-driven search: no gradients, a fixed direction set, line search.

    `directions` is `[D, hidden]` with unit rows; the caller supplies the identity
    basis plus seeded random directions. Only the verifier's outcome and margin
    are read, so this variant is usable where the loss is not differentiable.
    """
    verifier = verifier or Verifier()
    before = verify(student, item, None, verifier)
    best = torch.zeros(student.hidden)
    best_score = before["score"]
    trials = 0
    for direction in directions:
        for scale in scales:
            candidate = _project(best + direction * (scale * epsilon), epsilon)
            result = verify(student, item, (layer, candidate), verifier)
            trials += 1
            if result["score"] > best_score:
                best, best_score = candidate, result["score"]
            if result["correct"]:
                break
        if best_score >= 1.0:
            break
    after = verify(student, item, (layer, best), verifier)
    return {"layer": layer, "variant": "verifier", "before": before, "after": after,
            "success": after["correct"], "delta": best,
            "delta_norm": float(best.norm()), "epsilon": epsilon,
            "trials": trials}


def oracle_repair(student, items: list[RepairItem], epsilon: float | None = None,
                  layers: list[int] | None = None, variant: str = "surrogate",
                  epsilon_absolute: float | None = None, seed: int = SEED,
                  verifier: Verifier | None = None) -> dict:
    """Per-item, per-layer bounded search; the bottleneck is the earliest success."""
    layers = list(student.candidate_layers() if layers is None else layers)
    verifier = verifier or Verifier()
    generator = torch.Generator().manual_seed(seed)
    basis = torch.eye(student.hidden)
    noise = torch.randn(32, student.hidden, generator=generator)
    directions = torch.cat([basis, F.normalize(noise, dim=-1)], dim=0)
    per_item = {}
    for item in items:
        item_epsilon = (repair_budget(student, [item], layers[0],
                                      absolute=epsilon_absolute)
                        if epsilon is None else float(epsilon))
        rows = {}
        for layer in layers:
            if variant == "surrogate":
                result = surrogate_search(student, item, layer, item_epsilon,
                                          verifier=verifier)
            elif variant == "verifier":
                result = verifier_search(student, item, layer, item_epsilon,
                                         directions, verifier=verifier)
            else:
                raise ValueError(f"unknown oracle variant {variant!r}")
            rows[layer] = {key: result[key] for key in
                           ("variant", "success", "delta_norm", "epsilon",
                            "before", "after")}
            rows[layer]["delta"] = result["delta"]
        successful = [layer for layer in layers if rows[layer]["success"]]
        per_item[item.id] = {
            "label": item.label, "layers": rows,
            "bottleneck_layer": min(successful) if successful else None,
            "success": bool(successful),
        }
    successes = sum(1 for row in per_item.values() if row["success"])
    bottlenecks = [row["bottleneck_layer"] for row in per_item.values()
                   if row["bottleneck_layer"] is not None]
    return {
        "variant": variant,
        "items": len(items),
        "repaired": successes,
        "repair_rate": successes / len(items) if items else 0.0,
        "bottleneck_histogram": {str(layer): bottlenecks.count(layer)
                                 for layer in sorted(set(bottlenecks))},
        "per_item": per_item,
    }


def choose_deployed_layer(oracle: dict) -> int | None:
    """The registered deployed layer: the median earliest-successful layer.

    It is computed on the **training** split only, before any evaluation item is
    scored, and recorded in the receipt.
    """
    values = sorted(row["bottleneck_layer"] for row in oracle["per_item"].values()
                    if row["bottleneck_layer"] is not None)
    if not values:
        return None
    return values[len(values) // 2]


# ---------------------------------------------------------------------------
# the learned predictor


class CausalRepairPredictor(nn.Module):
    """`C_theta(h_layer, z_c, layer) -> delta_hat`, projected to the budget.

    `z_c` is a learned code of the **input tokens** (mean of a token embedding
    table); `h_layer` is the student's own hidden state at that layer; the layer
    index is one-hot. No label, no verifier score and no logit is an input.
    """

    def __init__(self, hidden: int, vocab: int, layers: int, code_dim: int = 8,
                 width: int | None = None):
        super().__init__()
        self.hidden = hidden
        self.vocab = vocab
        self.layers = int(layers)
        self.code_dim = code_dim
        self.code_embed = nn.Embedding(vocab, code_dim)
        width = width or 2 * hidden
        self.network = nn.Sequential(
            nn.Linear(hidden + code_dim + layers, width), nn.Tanh(),
            nn.Linear(width, hidden))
        with torch.no_grad():
            nn.init.normal_(self.code_embed.weight, std=0.1)
            for module in self.network:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.05)
                    nn.init.zeros_(module.bias)

    def code(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.code_embed(input_ids).mean(dim=1)

    def forward(self, hidden: torch.Tensor, code: torch.Tensor,
                layer: int) -> torch.Tensor:
        one_hot = F.one_hot(torch.tensor(layer, dtype=torch.long),
                            num_classes=self.layers).float()
        one_hot = one_hot.unsqueeze(0).expand(hidden.shape[0], -1)
        features = torch.cat([hidden.detach().reshape(hidden.shape[0], -1),
                              code, one_hot], dim=-1)
        return self.network(features)

    def predict(self, student, item: RepairItem, layer: int, epsilon: float) -> dict:
        with torch.no_grad():
            hidden = student.hidden_state(item.input_ids.unsqueeze(0), layer)
            last = hidden[:, -1, :]
            code = self.code(item.input_ids.unsqueeze(0))
            delta = self(last, code, layer).reshape(-1)
            delta = _project(delta, epsilon)
        return {"layer": layer, "delta": delta, "delta_norm": float(delta.norm())}


def train_predictor(student, items: list[RepairItem], oracle: dict,
                    epsilon: float, hidden: int, vocab: int, layers: int,
                    steps: int = PREDICTOR_STEPS, lr: float = PREDICTOR_LR,
                    seed: int = SEED, code_dim: int = 8) -> tuple:
    """Supervised by the oracle's correction; inputs are causal only."""
    torch.manual_seed(seed)
    predictor = CausalRepairPredictor(hidden, vocab, layers, code_dim=code_dim)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=lr)
    targets = []
    for item in items:
        row = oracle["per_item"][item.id]
        if row["bottleneck_layer"] is None:
            continue
        # Every layer at which the oracle found a repair is a training example,
        # so the predictor learns the *intervention* rather than one layer index;
        # this is what makes the transfer test about structure, not memorisation.
        cached = {}
        for layer, result in row["layers"].items():
            if not result["success"]:
                continue
            if layer not in cached:
                with torch.no_grad():
                    hidden_state = student.hidden_state(item.input_ids.unsqueeze(0),
                                                        layer)
                    code = predictor.code(item.input_ids.unsqueeze(0))
                cached[layer] = (hidden_state[:, -1, :].detach(), code.detach())
            hidden_last, code = cached[layer]
            targets.append((hidden_last, code, int(layer),
                            result["delta"], item))
    tokens_per_pass = sum(int(item.input_ids.numel()) for item in
                          {item.id: item for *_rest, item in targets}.values())
    history = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for hidden_last, code, layer, target, _item in targets:
            prediction = predictor(hidden_last, code, layer).reshape(-1)
            prediction = _project(prediction, epsilon)
            losses.append(F.mse_loss(prediction, target))
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        if (step + 1) % 50 == 0:
            history.append(float(loss.detach()))
    return predictor, {"steps": steps, "lr": lr, "seed": seed,
                       "targets": len(targets),
                       "items": len({item.id for *_rest, item in targets}),
                       "tokens": steps * tokens_per_pass,
                       "tokens_forward": tokens_per_pass,
                       "tokens_basis": "steps x sum of training-item sequence "
                                       "lengths, the same exposure as the LoRA "
                                       "control (features are cached once)",
                       "feature_cache": True,
                       "history": history,
                       "final_loss": history[-1] if history else None}


def evaluate_predictor(student, predictor, items: list[RepairItem], layer: int,
                       epsilon: float, verifier: Verifier | None = None) -> dict:
    verifier = verifier or Verifier()
    rows = {}
    for item in items:
        start = verify(student, item, None, verifier)
        prediction = predictor.predict(student, item, layer, epsilon)
        after = verify(student, item, (layer, prediction["delta"]), verifier)
        rows[item.id] = {"before": start, "after": after,
                         "repaired": bool(after["correct"] and not start["correct"]),
                         "delta_norm": prediction["delta_norm"]}
    repaired = sum(1 for row in rows.values() if row["repaired"])
    return {"items": len(items), "repaired": repaired,
            "repair_rate": repaired / len(items) if items else 0.0,
            "per_item": rows}


# ---------------------------------------------------------------------------
# controls: equal-budget distributed LoRA, all-layer adapters, none


class DistributedLoRA(nn.Module):
    """`h -> h + scale * ((h A_l) B_l)` at every layer; `B` starts at zero."""

    def __init__(self, hidden: int, layers: int, rank: int, scale: float = 1.0):
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be >= 1")
        self.hidden = hidden
        self.layers = layers
        self.rank = rank
        self.scale = scale
        self.down = nn.ParameterList([nn.Parameter(torch.empty(hidden, rank))
                                      for _ in range(layers)])
        self.up = nn.ParameterList([nn.Parameter(torch.zeros(hidden, rank))
                                    for _ in range(layers)])
        with torch.no_grad():
            for parameter in self.down:
                nn.init.normal_(parameter, std=0.02)

    def parameter_count(self) -> int:
        return 2 * self.layers * self.hidden * self.rank

    def apply(self, hidden: torch.Tensor, layer: int) -> torch.Tensor:
        return hidden + self.scale * ((hidden @ self.down[layer]) @ self.up[layer].t())


def matched_lora_rank(predictor_parameters: int, hidden: int, layers: int) -> int:
    """Largest rank whose total LoRA parameters do not exceed the predictor's."""
    return max(1, predictor_parameters // (2 * hidden * layers))


def predictor_parameter_count(hidden: int, vocab: int, layers: int,
                              code_dim: int = 8) -> int:
    return sum(parameter.numel() for parameter in
               CausalRepairPredictor(hidden, vocab, layers, code_dim=code_dim)
               .parameters())


def train_lora(student, items: list[RepairItem], lora: DistributedLoRA,
               steps: int = LORA_STEPS, lr: float = LORA_LR,
               seed: int = SEED) -> dict:
    """Distributed LoRA trained on the same items, tokens and steps."""
    torch.manual_seed(seed)
    parameters = [parameter for parameter in lora.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=lr)
    tokens = 0
    last = None
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for item in items:
            logits = _lora_logits(student, lora, item)
            losses.append(F.cross_entropy(logits.reshape(1, -1),
                                          torch.tensor([item.label])))
            tokens += int(item.input_ids.numel())
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        last = float(loss.detach())
    return {"steps": steps, "lr": lr, "seed": seed, "tokens": tokens,
            "final_loss": last, "parameters": lora.parameter_count(),
            "rank": lora.rank}


def _lora_logits(student, lora: DistributedLoRA, item: RepairItem) -> torch.Tensor:
    """Student forward with the LoRA edits inserted (fixture protocol only)."""
    hidden = student.embed(item.input_ids.unsqueeze(0))
    for index in range(student.layers):
        if index == student.bottleneck:
            hidden = hidden.clone()
            hidden[..., :student.classes] = 0.0
        hidden = lora.apply(hidden, index)
    return hidden.mean(dim=1)[:, :student.classes]


def evaluate_lora(student, lora: DistributedLoRA, items: list[RepairItem],
                  verifier: Verifier | None = None) -> dict:
    verifier = verifier or Verifier()
    rows = {}
    for item in items:
        start = verify(student, item, None, verifier)
        with torch.no_grad():
            logits = _lora_logits(student, lora, item)
        correct, _score = verifier.evaluate(logits, item.label)
        rows[item.id] = {"before": start, "repaired": bool(correct and
                                                          not start["correct"])}
    repaired = sum(1 for row in rows.values() if row["repaired"])
    return {"items": len(items), "repaired": repaired,
            "repair_rate": repaired / len(items) if items else 0.0,
            "per_item": rows}


def evaluate_no_adaptation(student, items: list[RepairItem],
                           verifier: Verifier | None = None) -> dict:
    verifier = verifier or Verifier()
    repaired = 0
    for item in items:
        start = verify(student, item, None, verifier)
        repaired += int(start["correct"])
    return {"items": len(items), "repaired": repaired,
            "repair_rate": repaired / len(items) if items else 0.0,
            "per_item": {item.id: {"repaired": False} for item in items}}


# ---------------------------------------------------------------------------
# the gap metrics and the kill condition


def capture(learned: dict, oracle: dict, baseline: dict | None = None) -> dict:
    """How much of the oracle's repair the learned arm reproduces.

    `capture = (learned repaired - baseline repaired) / (oracle repaired -
    baseline repaired)`, with the oracle's `repaired` counted on the same items.
    A zero denominator is reported as `null`, never as one.
    """
    base = (baseline or {}).get("repaired", 0)
    numerator = learned["repaired"] - base
    denominator = oracle["repaired"] - base
    return {"learned": learned["repaired"], "oracle": oracle["repaired"],
            "baseline": base, "capture": (numerator / denominator
                                          if denominator > 0 else None),
            "kill_threshold": ORACLE_CAPTURE_KILL}


def kill_condition(capture_report: dict, transfer_report: dict | None = None) -> dict:
    """The registered kill rule, evaluated mechanically."""
    captured = capture_report["capture"]
    transfer = (transfer_report or {}).get("capture")
    reasons = []
    if captured is None or captured < ORACLE_CAPTURE_KILL:
        reasons.append(f"capture {captured} < {ORACLE_CAPTURE_KILL}")
    if transfer_report is not None and (transfer is None
                                        or transfer < TRANSFER_CAPTURE_KILL):
        reasons.append(f"transfer capture {transfer} < {TRANSFER_CAPTURE_KILL}")
    return {"killed": bool(reasons), "reasons": reasons,
            "capture_threshold": ORACLE_CAPTURE_KILL,
            "transfer_threshold": TRANSFER_CAPTURE_KILL}


# ---------------------------------------------------------------------------
# fixture: the full CPU demonstration


def fixture_items(student: TinyLayeredStudent, count: int = 24, length: int = 5,
                  seed: int = 11, prefix: str = "a") -> list[RepairItem]:
    generator = torch.Generator().manual_seed(seed)
    items = []
    for index in range(count):
        label = 1 + int(torch.randint(0, student.classes - 1, (1,),
                                      generator=generator))
        # The class token is last; the earlier positions are filler id 0.
        ids = [0] * (length - 1) + [label]
        items.append(RepairItem(f"{prefix}{index:03d}", ids, label,
                                structure=f"L{student.layers}-b{student.bottleneck}"))
    return items


def failed_items(student, items: list[RepairItem],
                 verifier: Verifier | None = None) -> list[RepairItem]:
    verifier = verifier or Verifier()
    kept = []
    for item in items:
        result = verify(student, item, None, verifier)
        if not result["correct"]:
            kept.append(item)
    return kept


def split_items(items: list[RepairItem]) -> tuple[list[RepairItem], list[RepairItem]]:
    ordered = sorted(items, key=lambda item: item.id)
    return ordered[::2], ordered[1::2]


def run_fixture(seed: int = SEED) -> dict:
    """The CPU demonstration with known answers, used by tests and `--mode self-test`."""
    student = TinyLayeredStudent(seed=seed)
    verifier = Verifier()
    items = failed_items(student, fixture_items(student))
    train, test = split_items(items)
    epsilon = repair_budget(student, train, student.bottleneck, absolute=2.0)

    oracle_train = oracle_repair(student, train, epsilon=epsilon, seed=seed)
    deployed_layer = choose_deployed_layer(oracle_train)
    oracle_test = oracle_repair(student, test, epsilon=epsilon, seed=seed)
    predictor, training = train_predictor(
        student, train, oracle_train, epsilon,
        hidden=student.hidden, vocab=student.vocab, layers=student.layers,
        seed=seed)
    learned = evaluate_predictor(student, predictor, test, deployed_layer, epsilon)
    predictor_parameters = sum(parameter.numel() for parameter in predictor.parameters())
    rank = matched_lora_rank(predictor_parameters, student.hidden, student.layers)
    lora_equal = DistributedLoRA(student.hidden, student.layers, rank)
    lora_training = train_lora(student, train, lora_equal, seed=seed)
    lora_report = evaluate_lora(student, lora_equal, test)
    all_layer = DistributedLoRA(student.hidden, student.layers, ALL_LAYER_RANK)
    all_layer_training = train_lora(student, train, all_layer, seed=seed)
    all_layer_report = evaluate_lora(student, all_layer, test)
    none = evaluate_no_adaptation(student, test)

    # Transfer: the same predictor, a structurally different student (different
    # bottleneck layer and sequence length, same tokens and classes). The target
    # layer is supplied by the new problem's registration, not learned from its
    # evaluation items.
    transfer_student = TinyLayeredStudent(seed=seed + 1, bottleneck=2)
    transfer_items_all = failed_items(transfer_student,
                                      fixture_items(transfer_student, length=7,
                                                    seed=seed + 2, prefix="b"))
    transfer_train, transfer_test = split_items(transfer_items_all)
    transfer_epsilon = repair_budget(transfer_student, transfer_train, 2,
                                     absolute=2.0)
    transfer_oracle = oracle_repair(transfer_student, transfer_test,
                                    epsilon=transfer_epsilon, seed=seed)
    transfer = evaluate_predictor(transfer_student, predictor, transfer_test,
                                  2, transfer_epsilon)

    oracle_verifier = oracle_repair(student, test[:8], epsilon=epsilon,
                                    variant="verifier", seed=seed)
    capture_report = capture(learned, oracle_test, none)
    transfer_capture = capture(transfer, transfer_oracle,
                               evaluate_no_adaptation(transfer_student, transfer_test))
    verdict = kill_condition(capture_report, transfer_capture)
    return {
        "schema": SCHEMA_FIXTURE,
        "stamp": stamp_now(),
        "revision": os.environ.get("SELVEDGE_REVISION", "unknown"),
        "created_utc": utc_now(),
        "fixture": "TinyLayeredStudent: embedding + identity blocks + one planted "
                   "class-subspace deletion; not a transformer, no attention",
        "student": {"hidden": student.hidden, "classes": student.classes,
                    "layers": student.layers, "bottleneck": student.bottleneck,
                    "vocab": student.vocab},
        "items": {"total": len(items), "train": len(train), "test": len(test),
                  "sequence_length": 5},
        "budget": {"epsilon": epsilon, "rule": "absolute 2.0 in the fixture; the "
                   "registered rule is kappa_repair * median ||h_l|| with "
                   f"kappa_repair={KAPPA_REPAIR}"},
        "deployed_layer": deployed_layer,
        "oracle_train": {key: oracle_train[key] for key in
                         ("repaired", "repair_rate", "bottleneck_histogram")},
        "oracle_test": {key: oracle_test[key] for key in
                        ("repaired", "repair_rate", "bottleneck_histogram")},
        "oracle_verifier": {key: oracle_verifier[key] for key in
                            ("repaired", "repair_rate", "items")},
        "learned": learned,
        "predictor_training": training,
        "controls": {
            "distributed_lora_equal": {"training": lora_training, **lora_report},
            "all_layer_adapters": {"training": all_layer_training,
                                   **all_layer_report},
            "none": none,
        },
        "parameter_matching": {
            "predictor": predictor_parameters,
            "distributed_lora_rank": rank,
            "distributed_lora_parameters": lora_equal.parameter_count(),
            "all_layer_rank": ALL_LAYER_RANK,
            "all_layer_parameters": all_layer.parameter_count(),
            "note": "the predictor is the budget; the distributed LoRA is the "
                    "largest rank that does not exceed it; the all-layer bank is "
                    "a declared, unmatched reference",
        },
        "transfer": {"target": "student with bottleneck layer 2 and length 7",
                     "oracle": {key: transfer_oracle[key] for key in
                                ("repaired", "repair_rate", "bottleneck_histogram")},
                     "learned": transfer, "capture": transfer_capture},
        "capture": capture_report,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# the 0.6B adapter (GPU)


class HFCausalStudent:
    """A frozen HF causal LM as a repair student.

    The head is the model's own next-token distribution at the final position;
    the label is the correct next token from the (teacher-forced) item. The
    intervention is added to the final position of layer `layer`'s output, so the
    only thing a correction can change is the prediction it is meant to repair.
    """

    def __init__(self, model, hidden: int, layers: int):
        self.model = model
        self.hidden = hidden
        self.layers = layers
        self._hooks: list = []

    def candidate_layers(self) -> list[int]:
        return list(range(self.layers))

    def _run(self, input_ids: torch.Tensor, intervention=None):
        layer_index, delta = intervention if intervention is not None else (None, None)
        handles = []

        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            edit = hidden.clone()
            edit[:, -1, :] = edit[:, -1, :] + delta.reshape(1, -1)
            if isinstance(output, tuple):
                return (edit,) + tuple(output[1:])
            return edit

        if layer_index is not None:
            handle = self.model.model.layers[layer_index].register_forward_hook(hook)
            handles.append(handle)
        try:
            out = self.model(input_ids=input_ids, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        return out

    def logits(self, input_ids: torch.Tensor, intervention=None) -> torch.Tensor:
        out = self._run(input_ids, intervention)
        return out.logits[:, -1, :].float()

    def hidden_state(self, input_ids: torch.Tensor, layer: int) -> torch.Tensor:
        captured = {}

        def hook(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["hidden"] = hidden.detach()

        handle = self.model.model.layers[layer].register_forward_hook(hook)
        try:
            with torch.no_grad():
                self.model(input_ids=input_ids, use_cache=False)
        finally:
            handle.remove()
        return captured["hidden"]

    def code(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            embedding = self.model.get_input_embeddings()(input_ids)
        return embedding.detach().mean(dim=1)


def run_checkpoint(args) -> int:
    """The registered 0.6B run: oracle, predictor, equal-budget LoRA, controls."""
    if not torch.cuda.is_available():
        raise SystemExit("--mode checkpoint requires the GPU court (--gpu)")
    if not args.items:
        raise SystemExit("--items is required: a JSON list of {id, input_ids, label}")
    import sys

    eval_dir = os.path.join(REPO, "oracle", "eval")
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    import harness  # noqa: E402

    device = torch.device("cuda")
    with open(os.path.join(REPO, args.items), encoding="utf-8") as handle:
        raw_items = json.load(handle)
    items = [RepairItem(row["id"], row["input_ids"], row["label"],
                        row.get("structure", "default")) for row in raw_items]
    base = harness.load_model(harness.model_path(args.hf_root, args.model_size),
                             device)
    base.eval()
    student = HFCausalStudent(base, base.config.hidden_size,
                              len(base.model.layers))
    train, test = split_items(items)
    failed_train = failed_items(student, train)
    failed_test = failed_items(student, test)
    layer_list = (student.candidate_layers() if args.layers == "all"
                  else [int(value) for value in args.layers.split(",")])
    epsilon = repair_budget(student, failed_train, layer_list[0],
                            absolute=args.epsilon)
    oracle_train = oracle_repair(student, failed_train, epsilon=epsilon,
                                 layers=layer_list, variant=args.oracle_variant,
                                 seed=args.seed)
    deployed_layer = choose_deployed_layer(oracle_train)
    if deployed_layer is None:
        raise SystemExit("the oracle repaired nothing on the training split")
    oracle_test = oracle_repair(student, failed_test, epsilon=epsilon,
                                layers=layer_list, variant=args.oracle_variant,
                                seed=args.seed)
    predictor, training = train_predictor(
        student, failed_train, oracle_train, epsilon,
        hidden=student.hidden, vocab=base.config.vocab_size,
        layers=student.layers, seed=args.seed)
    learned = evaluate_predictor(student, predictor, failed_test,
                                 deployed_layer, epsilon)
    parameters = sum(parameter.numel() for parameter in predictor.parameters())
    rank = matched_lora_rank(parameters, student.hidden, student.layers)
    receipt = {
        "schema": SCHEMA_RUN,
        "stamp": stamp_now(),
        "revision": os.environ.get("SELVEDGE_REVISION", "unknown"),
        "created_utc": utc_now(),
        "student": {"model_size": args.model_size, "hidden": student.hidden,
                    "layers": student.layers, "candidate_layers": layer_list},
        "items": {"total": len(items), "train": len(train), "test": len(test),
                  "failed_train": len(failed_train),
                  "failed_test": len(failed_test), "source": args.items},
        "budget": {"epsilon": epsilon, "kappa_repair": KAPPA_REPAIR,
                   "rule": "kappa_repair * median ||h_l|| at the first candidate "
                           "layer, computed on the training split"},
        "deployed_layer": deployed_layer,
        "oracle_train": {key: oracle_train[key] for key in
                         ("repaired", "repair_rate", "bottleneck_histogram")},
        "oracle_test": {key: oracle_test[key] for key in
                        ("repaired", "repair_rate", "bottleneck_histogram")},
        "learned": learned,
        "predictor_training": training,
        "parameter_matching": {"predictor": parameters,
                               "distributed_lora_rank": rank},
        "controls": {
            "none": evaluate_no_adaptation(student, failed_test),
            "note": "the distributed LoRA and the all-layer bank require injecting "
                    "the adapter into the HF core; that arm is implemented for the "
                    "fixture protocol and is not yet wired to HF. This receipt says "
                    "so rather than omitting the control silently.",
        },
        "capture": capture(learned, oracle_test,
                           evaluate_no_adaptation(student, failed_test)),
        "note": "the predictor's z_c is the mean of the frozen input embedding; "
                "labels and verifier outcomes are never inputs",
    }
    write_json_atomic(os.path.join(REPO, args.out), receipt)
    print(json.dumps({"out": args.out, "deployed_layer": deployed_layer,
                      "capture": receipt["capture"]}, indent=2, sort_keys=True))
    return 0


# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["self-test", "checkpoint"],
                        default="self-test")
    parser.add_argument("--out", required=True)
    parser.add_argument("--items", default=None,
                        help="JSON list of {id, input_ids, label} for the 0.6B run")
    parser.add_argument("--model-size", default="0.6b", choices=["0.6b", "1.7b"])
    parser.add_argument("--hf-root", default=os.environ.get("HF_HOME", "/hf-cache"))
    parser.add_argument("--layers", default="all",
                        help="`all` or a comma-separated candidate list")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="absolute budget; omitted uses kappa_repair * median norm")
    parser.add_argument("--oracle-variant", choices=["surrogate", "verifier"],
                        default="surrogate")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    if args.mode == "checkpoint":
        return run_checkpoint(args)
    receipt = run_fixture(seed=args.seed)
    receipt["revision"] = os.environ.get("SELVEDGE_REVISION", "unknown")
    write_json_atomic(os.path.join(REPO, args.out), receipt)
    print(json.dumps({"oracle_test": receipt["oracle_test"]["repaired"],
                      "learned": receipt["learned"]["repaired"],
                      "capture": receipt["capture"]["capture"],
                      "verdict": receipt["verdict"], "out": args.out},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
