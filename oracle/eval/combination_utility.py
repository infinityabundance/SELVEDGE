#!/usr/bin/env python3
"""P9.3 — combination utility: four-condition data, interaction, controller.

The quantity is the interaction of a capability pair for one item:

```
I(m, e; x) = R(S + m + e) - R(S + m) - R(S + e) + R(S)
```

with `R` a reward (higher is better). With losses `L = -R` the same quantity is
`-(L_both - L_m - L_e + L_none)`. Positive is synergy, negative antagonism, and
a confidence interval that contains zero is reported as **no detectable
interaction** — never as "independence".

The court has four parts:

1. **Four-condition counterfactual generation.** For every item the runner is
   called with `(use_m, use_e) in {(0,0), (1,0), (0,1), (1,1)}` and receives a
   **restricted causal view** of the item: outcome keys (`label`, `target`,
   `answer`, `reward`, ...) raise on access. The artifact carries a manifest with
   the item-id hash, the feature hash, the condition list and the leakage
   assertions that were actually executed.
2. **The interaction estimator**, with a percentile paired bootstrap over items,
   a standard error, and a decision label (`synergy` / `antagonism` / `no
   detectable interaction`).
3. **The controller prototype**: conditional marginal utility from causal inputs
   only. `AdditiveController` predicts `u_m`, `u_e` and chooses the subset with
   the largest additive value; `InteractionController` also predicts `I` and
   scores pair subsets with the cross term. `IndependentGates` and a
   load-balanced router are the cheaper controls.
4. **The small combinatorial oracle**: the best subset per item from the measured
   four conditions, which is not deployable and is reported separately.

The CPU tests (`oracle/tests/test_p9_combination_utility.py`) run on synthetic
ground truth where `I` is known by construction, including sign recovery and CI
coverage. The 0.6B run is `--mode checkpoint`: the H arm with the memory and
expert paths toggled on a calibration slice, with the same CPU-tested estimator.
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

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.dirname(HERE)
REPO = os.path.dirname(ORACLE)

SCHEMA_DATASET = "selvedge/p9.3-combination-dataset/v1"
SCHEMA_REPORT = "selvedge/p9.3-combination-report/v1"
CONDITIONS = ("none", "m", "e", "both")
SUBSETS = ("none", "m", "e", "both")
BLOCKED_KEYS = ("label", "labels", "target", "targets", "answer", "answers",
                "reward", "rewards", "correct", "outcome", "verifier")
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261115
ALPHA = 0.05


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
# the causal view and four-condition generation


class CausalView:
    """The runner's view of an item: causal inputs only, outcome keys blocked."""

    __slots__ = ("_data",)

    def __init__(self, data: dict):
        self._data = dict(data)

    def __getitem__(self, key):
        if str(key).lower() in BLOCKED_KEYS:
            raise KeyError(f"the runner may not read outcome key {key!r}")
        return self._data[key]

    def get(self, key, default=None):
        return self[ key ] if key in self._data else default

    def __contains__(self, key):
        return key in self._data and str(key).lower() not in BLOCKED_KEYS

    def keys(self) -> list[str]:
        return [key for key in self._data if str(key).lower() not in BLOCKED_KEYS]

    def as_dict(self) -> dict:
        return {key: self._data[key] for key in self.keys()}


def assert_no_outcome_keys(items: list[dict]) -> dict:
    """Fail closed if a casual input dict carries an outcome field."""
    found = sorted({key for item in items for key in item
                    if str(key).lower() in BLOCKED_KEYS})
    if found:
        raise ValueError(f"items carry outcome keys, so the view would leak: {found}")
    return {"causal_inputs_only": True, "blocked_keys": list(BLOCKED_KEYS),
            "exposed_keys": sorted({key for item in items for key in item})}


def generate_four_condition(items: list[dict], runner, capabilities: dict | None = None,
                            noise_note: str = "none") -> dict:
    """Run every item under the four conditions and build the reusable artifact."""
    leakage = assert_no_outcome_keys(items)
    if not items:
        raise ValueError("no items")
    identifiers = [str(item["id"]) for item in items]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("item ids must be unique")
    records = {}
    for item in items:
        view = CausalView(item)
        row = {}
        input_digest = sha256_text(json.dumps(view.as_dict(), sort_keys=True))
        for use_m, use_e in ((0, 0), (1, 0), (0, 1), (1, 1)):
            reward = runner(view, bool(use_m), bool(use_e))
            condition = ("both" if use_m and use_e else "m" if use_m else
                         "e" if use_e else "none")
            row[condition] = float(reward)
        row["id"] = str(item["id"])
        row["input_sha256"] = input_digest
        row["interaction"] = interaction_from_rewards(row)
        records[str(item["id"])] = row
    train, test = split_ids(identifiers)
    reward_vector = [record["interaction"] for record in records.values()]
    return {
        "schema": SCHEMA_DATASET,
        "stamp": stamp_now(),
        "revision": os.environ.get("SELVEDGE_REVISION", "unknown"),
        "created_utc": utc_now(),
        "capabilities": capabilities or {"m": "memory path", "e": "expert path"},
        "metric": "reward (higher is better); interaction from rewards",
        "conditions": list(CONDITIONS),
        "items": len(items),
        "records": records,
        "manifest": {
            "item_id_sha256": sha256_text(json.dumps(sorted(identifiers))),
            "feature_sha256": sha256_text(json.dumps(
                [item.get("features", []) for item in items], sort_keys=True)),
            "input_sha256": sha256_text(json.dumps(
                sorted(record["input_sha256"] for record in records.values()))),
            "conditions": list(CONDITIONS),
            "leakage": leakage,
            "runner_input": "CausalView (outcome keys raise)",
            "noise": noise_note,
        },
        "split": {"train": train, "test": test,
                  "rule": "alternating over the id-sorted set"},
    }


def split_ids(identifiers: list[str]) -> tuple[list[str], list[str]]:
    ordered = sorted(identifiers)
    return ordered[::2], ordered[1::2]


# ---------------------------------------------------------------------------
# the interaction estimator


def interaction_from_rewards(rewards: dict) -> float:
    return (float(rewards["both"]) - float(rewards["m"])
            - float(rewards["e"]) + float(rewards["none"]))


def interaction_from_losses(losses: dict) -> float:
    return -(float(losses["both"]) - float(losses["m"])
             - float(losses["e"]) + float(losses["none"]))


def paired_bootstrap(differences: list[float], resamples: int = BOOTSTRAP_RESAMPLES,
                     seed: int = BOOTSTRAP_SEED, alpha: float = ALPHA) -> dict:
    values = torch.tensor(differences, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    index = torch.randint(0, values.numel(), (resamples, values.numel()),
                          generator=generator)
    means = values[index].mean(dim=1)
    return {"mean": float(values.mean()),
            "low": float(torch.quantile(means, alpha / 2)),
            "high": float(torch.quantile(means, 1 - alpha / 2)),
            "n": int(values.numel()), "resamples": resamples, "seed": seed}


def interaction_summary(dataset: dict, ids: list[str] | None = None,
                        resamples: int = BOOTSTRAP_RESAMPLES,
                        seed: int = BOOTSTRAP_SEED) -> dict:
    """Mean interaction with a paired bootstrap CI and a decision label."""
    ids = ids or sorted(dataset["records"])
    values = [dataset["records"][identifier]["interaction"] for identifier in ids]
    if not values:
        raise ValueError("no records to summarise")
    bootstrap = paired_bootstrap(values, resamples=resamples, seed=seed)
    standard_error = (float(torch.tensor(values).std(unbiased=True))
                      / math.sqrt(len(values))) if len(values) > 1 else None
    if bootstrap["low"] > 0:
        label = "synergy"
    elif bootstrap["high"] < 0:
        label = "antagonism"
    else:
        label = "no detectable interaction"
    return {**bootstrap, "se": standard_error, "label": label,
            "alpha": ALPHA,
            "convention": "reward; positive = super-additive, negative = sub-additive"}


# ---------------------------------------------------------------------------
# controllers


class _Ridge:
    """Ridge regression with an unpenalised intercept, closed form."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.weights: torch.Tensor | None = None
        self.bias = 0.0
        self.mean: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None

    def fit(self, features: torch.Tensor, targets: torch.Tensor) -> "_Ridge":
        self.mean = features.mean(dim=0)
        self.scale = features.std(dim=0).clamp_min(1e-6)
        standard = (features - self.mean) / self.scale
        design = torch.cat([standard, torch.ones(standard.shape[0], 1)], dim=1)
        count, width = design.shape
        penalty = self.alpha * torch.eye(width)
        penalty[-1, -1] = 0.0
        solution = torch.linalg.solve(design.t() @ design + penalty,
                                      design.t() @ targets)
        self.weights = solution[:-1]
        self.bias = float(solution[-1])
        return self

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        standard = (features - self.mean) / self.scale
        return standard @ self.weights + self.bias


def _features(item_ids: list[str], records: dict) -> torch.Tensor:
    rows = [records[identifier]["features"] for identifier in item_ids]
    return torch.tensor(rows, dtype=torch.float32)


def _with_features(dataset: dict, item_features: dict[str, list[float]]) -> dict:
    for identifier, features in item_features.items():
        dataset["records"][identifier]["features"] = [float(value) for value in features]
    return dataset


def _marginals(records: dict, ids: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    baseline = torch.tensor([records[i]["none"] for i in ids], dtype=torch.float32)
    margin_m = torch.tensor([records[i]["m"] for i in ids], dtype=torch.float32) - baseline
    margin_e = torch.tensor([records[i]["e"] for i in ids], dtype=torch.float32) - baseline
    return margin_m, margin_e


def _choose_additive(utility_m: torch.Tensor, utility_e: torch.Tensor,
                     interaction: torch.Tensor | None = None) -> list[str]:
    picks = []
    for index in range(utility_m.numel()):
        values = {"none": 0.0, "m": float(utility_m[index]),
                  "e": float(utility_e[index])}
        both = float(utility_m[index] + utility_e[index])
        if interaction is not None:
            both += float(interaction[index])
        values["both"] = both
        picks.append(max(values, key=values.get))
    return picks


class AdditiveController:
    """Independent conditional marginals; the combination is their sum."""

    name = "additive-utility"

    def __init__(self, alpha: float = 1.0):
        self.model_m = _Ridge(alpha)
        self.model_e = _Ridge(alpha)

    def fit(self, dataset: dict) -> "AdditiveController":
        train = dataset["split"]["train"]
        features = _features(train, dataset["records"])
        margin_m, margin_e = _marginals(dataset["records"], train)
        self.model_m.fit(features, margin_m)
        self.model_e.fit(features, margin_e)
        return self

    def choose(self, dataset: dict, ids: list[str]) -> list[str]:
        features = _features(ids, dataset["records"])
        return _choose_additive(self.model_m.predict(features),
                                self.model_e.predict(features))

    def cost(self) -> dict:
        return {"learned_parameters": int(self.model_m.weights.numel()
                                          + self.model_e.weights.numel() + 2)}


class InteractionController(AdditiveController):
    """Adds a predicted cross term `I(x)` to the value of the pair subset."""

    name = "interaction-aware"

    def __init__(self, alpha: float = 1.0):
        super().__init__(alpha)
        self.model_i = _Ridge(alpha)

    def fit(self, dataset: dict) -> "InteractionController":
        super().fit(dataset)
        train = dataset["split"]["train"]
        features = _features(train, dataset["records"])
        interaction = torch.tensor([dataset["records"][i]["interaction"]
                                    for i in train], dtype=torch.float32)
        self.model_i.fit(features, interaction)
        return self

    def choose(self, dataset: dict, ids: list[str]) -> list[str]:
        features = _features(ids, dataset["records"])
        return _choose_additive(self.model_m.predict(features),
                                self.model_e.predict(features),
                                self.model_i.predict(features))

    def cost(self) -> dict:
        return {"learned_parameters": int(self.model_m.weights.numel()
                                          + self.model_e.weights.numel()
                                          + self.model_i.weights.numel() + 3)}


class IndependentGates:
    """One logistic gate per capability on the marginals, thresholded at zero."""

    name = "independent-gates"

    def __init__(self, alpha: float = 1.0):
        self.gate_m = _Ridge(alpha)
        self.gate_e = _Ridge(alpha)

    def fit(self, dataset: dict) -> "IndependentGates":
        train = dataset["split"]["train"]
        features = _features(train, dataset["records"])
        margin_m, margin_e = _marginals(dataset["records"], train)
        self.gate_m.fit(features, (margin_m > 0).float())
        self.gate_e.fit(features, (margin_e > 0).float())
        return self

    def choose(self, dataset: dict, ids: list[str]) -> list[str]:
        features = _features(ids, dataset["records"])
        open_m = self.gate_m.predict(features) > 0.5
        open_e = self.gate_e.predict(features) > 0.5
        picks = []
        for index in range(len(ids)):
            picks.append("both" if open_m[index] and open_e[index] else
                         "m" if open_m[index] else "e" if open_e[index] else "none")
        return picks

    def cost(self) -> dict:
        return {"learned_parameters": int(self.gate_m.weights.numel()
                                          + self.gate_e.weights.numel() + 2)}


class LoadBalanced:
    """Equal load on both capabilities; no learned parameters."""

    name = "load-balanced"

    def fit(self, dataset: dict) -> "LoadBalanced":
        return self

    def choose(self, dataset: dict, ids: list[str]) -> list[str]:
        return ["both"] * len(ids)

    def cost(self) -> dict:
        return {"learned_parameters": 0}


class CombinatorialOracle:
    """The best subset per item from the measured four conditions (not deployable)."""

    name = "combinatorial-oracle"

    def choose(self, dataset: dict, ids: list[str]) -> list[str]:
        picks = []
        for identifier in ids:
            row = dataset["records"][identifier]
            picks.append(max(SUBSETS, key=lambda subset: row[subset]))
        return picks

    def cost(self) -> dict:
        return {"learned_parameters": 0, "deployable": False}


def evaluate_controllers(dataset: dict) -> dict:
    """Fit every controller on the training split, score on the test split."""
    if "features" not in next(iter(dataset["records"].values())):
        raise ValueError("records carry no features; call _with_features first")
    test = dataset["split"]["test"]
    controllers = [AdditiveController(), InteractionController(),
                   IndependentGates(), LoadBalanced()]
    oracle = CombinatorialOracle()
    oracle_picks = oracle.choose(dataset, test)
    realized_oracle = [dataset["records"][i][pick] for i, pick in zip(test, oracle_picks)]
    rows = {}
    picks_by_arm = {}
    for controller in controllers:
        controller.fit(dataset)
        picks = controller.choose(dataset, test)
        realized = [dataset["records"][i][pick] for i, pick in zip(test, picks)]
        differences = [realized[index] - realized_oracle[index]
                       for index in range(len(test))]
        rows[controller.name] = {
            "mean_reward": sum(realized) / len(realized),
            "oracle_gap": -sum(differences) / len(differences),
            "accuracy": sum(1 for left, right in zip(picks, oracle_picks)
                            if left == right) / len(test),
            "picks": {subset: picks.count(subset) for subset in SUBSETS},
            "cost": controller.cost(),
            "vs_oracle": paired_bootstrap(differences),
        }
        picks_by_arm[controller.name] = picks
    rows[oracle.name] = {
        "mean_reward": sum(realized_oracle) / len(realized_oracle),
        "oracle_gap": 0.0, "accuracy": 1.0,
        "picks": {subset: oracle_picks.count(subset) for subset in SUBSETS},
        "cost": oracle.cost(),
    }
    additive = picks_by_arm["additive-utility"]
    interaction = picks_by_arm["interaction-aware"]
    differences = [dataset["records"][test[index]][interaction[index]]
                   - dataset["records"][test[index]][additive[index]]
                   for index in range(len(test))]
    return {"items": len(test),
            "controllers": rows,
            "interaction_minus_additive": paired_bootstrap(differences),
            "decision_pair": "interaction-minus-additive"}


# ---------------------------------------------------------------------------
# synthetic ground truth


class SyntheticCapabilities:
    """Known `u_m`, `u_e` and interaction, with deterministic measurement noise.

    Feature `x0` gates memory, `x1` gates experts, and the interaction is a
    constant chosen by `kind`, so the true sign is known before measurement. Each
    condition's noise is a deterministic function of `(item id, condition, seed)`
    so a replication is exactly reproducible and independent across conditions.
    """

    KINDS = {"synergy": 0.6, "antagonism": -0.6, "independence": 0.0}

    def __init__(self, kind: str = "synergy", noise: float = 0.05, seed: int = 7):
        if kind not in self.KINDS:
            raise ValueError(f"kind must be one of {sorted(self.KINDS)}")
        self.kind = kind
        self.interaction = self.KINDS[kind]
        self.noise = float(noise)
        self.seed = int(seed)

    def margins(self, features: list[float]) -> tuple[float, float]:
        x0, x1, x2 = features[:3]
        utility_m = 0.8 * x0 + 0.3 * x1 - 0.2 + 0.1 * x2
        utility_e = 0.7 * x1 - 0.4 * x0 + 0.1
        return utility_m, utility_e

    def reward(self, view: CausalView, use_m: bool, use_e: bool) -> float:
        utility_m, utility_e = self.margins(view["features"])
        value = 0.0
        if use_m:
            value += utility_m
        if use_e:
            value += utility_e
        if use_m and use_e:
            value += self.interaction
        condition = f"{int(use_m)}{int(use_e)}"
        generator = torch.Generator().manual_seed(
            int(hashlib.sha256(f"{self.seed}|{view['id']}|{condition}".encode())
                .hexdigest()[:8], 16))
        return value + self.noise * float(torch.randn(1, generator=generator))


def synthetic_items(count: int = 96, seed: int = 3, prefix: str = "s") -> list[dict]:
    generator = torch.Generator().manual_seed(seed)
    items = []
    for index in range(count):
        features = [float(value) for value in torch.randn(3, generator=generator)]
        items.append({"id": f"{prefix}{index:03d}", "features": features,
                      "input_ids": [1, 2, 3]})
    return items


def synthetic_dataset(kind: str = "synergy", count: int = 96, noise: float = 0.05,
                      seed: int = 7) -> dict:
    capabilities = SyntheticCapabilities(kind=kind, noise=noise, seed=seed)
    dataset = generate_four_condition(
        synthetic_items(count=count, seed=seed + 1),
        capabilities.reward,
        capabilities={"m": "synthetic memory", "e": "synthetic expert"},
        noise_note=f"iid gaussian sigma={noise} per condition, seeded by "
                   f"(seed, item id, condition)")
    _with_features(dataset, {item["id"]: item["features"]
                            for item in synthetic_items(count=count, seed=seed + 1)})
    dataset["ground_truth"] = {"kind": kind, "interaction": capabilities.interaction,
                               "noise": noise}
    return dataset


def coverage_study(kind: str, replications: int = 40, count: int = 80,
                   noise: float = 0.05, resamples: int = 1000,
                   seed: int = 11) -> dict:
    """How often the 95% interval covers the true interaction."""
    truth = SyntheticCapabilities.KINDS[kind]
    covered = 0
    estimates = []
    for replication in range(replications):
        dataset = synthetic_dataset(kind=kind, count=count, noise=noise,
                                    seed=seed + replication)
        summary = interaction_summary(dataset, resamples=resamples,
                                      seed=BOOTSTRAP_SEED + replication)
        estimates.append(summary["mean"])
        covered += int(summary["low"] <= truth <= summary["high"])
    return {"kind": kind, "truth": truth, "replications": replications,
            "coverage": covered / replications,
            "mean_estimate": sum(estimates) / len(estimates),
            "resamples": resamples}


# ---------------------------------------------------------------------------
# the 0.6B checkpoint path


class _PathToggles:
    """Zero the output of the memory and/or expert paths for one forward pass."""

    def __init__(self, model, use_m: bool, use_e: bool):
        self.model = model
        self.use_m = use_m
        self.use_e = use_e
        self.handles = []

    def __enter__(self):
        from modeling import HashedMemory, SpecialistBank

        for module in self.model.modules():
            wanted = (isinstance(module, HashedMemory) and self.use_m) or \
                     (isinstance(module, SpecialistBank) and self.use_e)
            if not wanted:
                self.handles.append(module.register_forward_hook(
                    lambda _module, _args, output: torch.zeros_like(output)))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        return False


def run_checkpoint(args) -> int:
    if not torch.cuda.is_available():
        raise SystemExit("--mode checkpoint requires the GPU court (--gpu)")
    import sys

    train_dir = os.path.join(ORACLE, "train")
    if train_dir not in sys.path:
        sys.path.insert(0, train_dir)
    import numpy as np

    import harness  # noqa: E402
    from loop import TargetReader, distillation_loss_per_position  # noqa: E402
    from modeling import build_arm, load_admission  # noqa: E402

    device = torch.device("cuda")
    config, admission = load_admission(args.config, args.admission)
    base = harness.load_model(harness.model_path(args.hf_root, args.model_size),
                             device)
    arm = build_arm(base, args.arm, config, admission, args.seed, profile="t2")
    arm.model.to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    arm.model.load_state_dict(state["model"], strict=False)
    arm.model.eval()
    index = json.load(open(os.path.join(REPO, args.artifact_dir, "train.index.json"),
                           encoding="utf-8"))
    stream = np.memmap(os.path.join(REPO, args.artifact_dir, "train.bin"),
                       dtype="<u4", mode="r", shape=(index["tokens"],))
    targets = TargetReader(os.path.join(REPO, args.artifact_dir))
    window = 1024
    windows = args.calibration_windows
    tokens = windows * window
    slice_ids = torch.from_numpy(np.asarray(stream[0:tokens], dtype="int64")
                                 .reshape(windows, window)).to(device)
    # One window at a time: 64 x 1024 in one forward would exceed the 16 GiB
    # device. The target cache is indexed by window.
    cache = {"values": [], "target_ids": [], "lse": []}
    with torch.no_grad():
        for row in range(windows):
            ids = slice_ids[row:row + 1]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = arm.model(input_ids=ids, use_cache=False).logits
            flat = logits.reshape(-1, logits.shape[-1])
            values, target_ids, lse = targets.read(row * window, window, device)
            cache["values"].append(values)
            cache["target_ids"].append(target_ids)
            cache["lse"].append(lse)

    def runner(view, use_m: bool, use_e: bool) -> float:
        row = view["row"]
        ids = slice_ids[row:row + 1]
        with torch.no_grad(), _PathToggles(arm.model, use_m, use_e):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = arm.model(input_ids=ids, use_cache=False).logits
            flat = logits.reshape(-1, logits.shape[-1])
            offset = row * window
            losses = distillation_loss_per_position(
                flat, cache["values"][row], cache["target_ids"][row],
                cache["lse"][row])
        return -float(losses.mean())

    items = []
    for row in range(windows):
        ids = slice_ids[row]
        features = [
            math.log1p(window),
            math.log1p(int(ids.numel())),
            float(ids.float().mean()) / 1000.0,
            float(ids.float().std()) / 1000.0,
            float(ids[0].item()) / 1000.0,
            float(ids[-1].item()) / 1000.0,
            float(torch.unique(ids).numel()) / window,
            float((ids > 1000).float().mean()),
        ]
        items.append({"id": f"w{row:03d}", "features": features, "row": row})
    dataset = generate_four_condition(
        items, runner,
        capabilities={"m": f"{args.arm} hashed memory", "e": f"{args.arm} specialists"},
        noise_note="deterministic teacher-forced loss on the calibration slice")
    _with_features(dataset, {item["id"]: item["features"] for item in items})
    report = {
        "schema": SCHEMA_REPORT,
        "stamp": stamp_now(),
        "revision": os.environ.get("SELVEDGE_REVISION", "unknown"),
        "created_utc": utc_now(),
        "arm": args.arm,
        "checkpoint": args.checkpoint,
        "calibration": {"windows": windows, "tokens": tokens,
                        "source": args.artifact_dir},
        "dataset": {key: value for key, value in dataset.items()
                    if key not in ("records",)},
        "interaction_all": interaction_summary(dataset),
        "interaction_train": interaction_summary(dataset, dataset["split"]["train"]),
        "interaction_test": interaction_summary(dataset, dataset["split"]["test"]),
        "controllers": evaluate_controllers(dataset),
        "note": "the runner toggles the two paths with forward hooks that zero "
                "their outputs; labels/teacher targets are used only by the "
                "measurement, never by the controller's inputs",
    }
    write_json_atomic(os.path.join(REPO, args.out), report)
    print(json.dumps({"out": args.out,
                      "interaction": report["interaction_all"],
                      "controllers": {name: row["mean_reward"] for name, row
                                      in report["controllers"]["controllers"].items()}},
                     indent=2, sort_keys=True))
    return 0


# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["self-test", "checkpoint"],
                        default="self-test")
    parser.add_argument("--kind", choices=sorted(SyntheticCapabilities.KINDS),
                        default="synergy")
    parser.add_argument("--out", required=True)
    parser.add_argument("--arm", default="H*")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--admission", default=None)
    parser.add_argument("--model-size", default="0.6b", choices=["0.6b", "1.7b"])
    parser.add_argument("--hf-root", default=os.environ.get("HF_HOME", "/hf-cache"))
    parser.add_argument("--artifact-dir", default="results/artifacts/train")
    parser.add_argument("--calibration-windows", type=int, default=32)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args(argv)

    if args.mode == "checkpoint":
        for name in ("checkpoint", "config", "admission"):
            if not getattr(args, name):
                raise SystemExit(f"--{name} is required for --mode checkpoint")
        return run_checkpoint(args)

    dataset = synthetic_dataset(kind=args.kind, seed=args.seed % 97)
    interaction = interaction_summary(dataset)
    controllers = evaluate_controllers(dataset)
    report = {
        "schema": SCHEMA_REPORT,
        "stamp": stamp_now(),
        "revision": os.environ.get("SELVEDGE_REVISION", "unknown"),
        "created_utc": utc_now(),
        "mode": "self-test",
        "fixture": "synthetic units where u_m, u_e and I are known by construction; "
                   "no 0.6B number is claimed",
        "kind": args.kind,
        "dataset": {key: value for key, value in dataset.items()
                    if key not in ("records",)},
        "ground_truth": dataset["ground_truth"],
        "interaction": interaction,
        "controllers": controllers,
        "coverage": coverage_study(args.kind, replications=16, count=80,
                                   resamples=500, seed=13),
    }
    write_json_atomic(os.path.join(REPO, args.out), report)
    print(json.dumps({"kind": args.kind,
                      "interaction": {key: interaction[key] for key in
                                      ("mean", "low", "high", "label")},
                      "coverage": report["coverage"]["coverage"],
                      "interaction_minus_additive":
                          controllers["interaction_minus_additive"],
                      "out": args.out}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
