#!/usr/bin/env python3
"""P8.1 — task-level capability selection on the fresh held-out set.

Three selectors, in the order the plan registers them:

1. **`fixed-rule`** — a pre-registered rule with no learned parameters
   (`code_score >= 0.5` picks `D`, otherwise `M`; `code_score` counts code and
   Rust markers in the task prefix). It is written here, before measurement, and
   is not tuned on the data.
2. **`trained-router`** — a logistic head on standardised task features
   (prefix statistics plus an optional cheap hidden-state summary). Trained on
   one half of the fresh set against the measured per-item outcome, then frozen.
   It reads no gold answer at decision time: its inputs are the prompt and the
   forward state.
3. **`oracle`** — picks the better capability per item. It needs the outcome, so
   it is **not deployable**; it is the bound that separates "the selector is
   bad" from "the capabilities are not complementary".

Everything the harness reports is paired: the metric is the per-item loss of the
selected capability, and every selector is compared to `always-D` and `always-M`
with a percentile paired bootstrap over items (seed fixed). Selector cost is
reported per decision (learned parameters read), separately from the
capabilities' own cost.

The harness is capability-agnostic so it can be exercised on CPU with synthetic
capabilities (`--mode self-test`) and then pointed at the trained arms
(`--mode checkpoint`, GPU: it loads the T2 `D` and `M` checkpoints and scores the
frozen completions). Only the self-test runs in this session.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.dirname(HERE)
REPO = os.path.dirname(ORACLE)
sys.path.insert(0, HERE)

import fresh_tasks  # noqa: E402

SCHEMA = "selvedge/p8-selectors/v1"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261115
ROUTER_STEPS = 300
ROUTER_LR = 0.05
ROUTER_SEED = 1001
FIXED_RULE_THRESHOLD = 0.5

FEATURE_NAMES = (
    "log_words", "digit_fraction", "punct_fraction", "upper_fraction",
    "code_marker_score", "rust_keyword_score", "log_lines", "mean_word_length",
    "has_code_fence", "log_chars",
)

CODE_MARKERS = ("def ", "fn ", "impl ", "struct ", "let ", "->", "```", "{", ";",
                "()", "class ", "return ")
RUST_KEYWORDS = ("rust", "cargo", "borrow", "lifetime", "trait", "mutex", "async",
                 "compil", "thread", "ownership")


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


# ---------------------------------------------------------------------------
# features


def task_features(item: dict, token_ids: list[int] | None = None,
                  hidden_summary: list[float] | None = None) -> tuple[list[float], list[str]]:
    """Task-prefix statistics; no gold label, no completion text.

    `token_ids` (optional) replaces the whitespace-word proxy with the model's
    own token count. `hidden_summary` appends the caller's cheap summary of the
    prefix state (`norm`, `mean absolute`, `last-token norm`), which is what the
    deployed selector may read.
    """
    text = item["prompt"]
    lowered = text.lower()
    words = re.findall(r"[A-Za-z0-9_']+", text)
    characters = text or " "
    digits = sum(character.isdigit() for character in characters)
    punctuation = sum(not character.isalnum() and not character.isspace()
                      for character in characters)
    upper = sum(character.isupper() for character in characters)
    marker_hits = sum(1 for marker in CODE_MARKERS if marker in lowered)
    rust_hits = sum(1 for keyword in RUST_KEYWORDS if keyword in lowered)
    values = [
        math.log1p(len(token_ids) if token_ids is not None else len(words)),
        digits / len(characters),
        punctuation / len(characters),
        upper / len(characters),
        min(1.0, marker_hits / 4.0),
        min(1.0, rust_hits / 4.0),
        math.log1p(text.count("\n")),
        (sum(len(word) for word in words) / len(words)) if words else 0.0,
        1.0 if "```" in text else 0.0,
        math.log1p(len(characters)),
    ]
    names = list(FEATURE_NAMES)
    if hidden_summary is not None:
        if len(hidden_summary) != 3:
            raise ValueError("hidden_summary must be [norm, mean_abs, last_norm]")
        values.extend(float(value) for value in hidden_summary)
        names.extend(("hidden_norm", "hidden_mean_abs", "hidden_last_norm"))
    return values, names


# ---------------------------------------------------------------------------
# capabilities and selectors


def capability_losses(tasks: list[dict], capabilities: dict) -> dict[str, list[float]]:
    """`{capability: [loss per task]}`; lower is better, both are required."""
    losses = {}
    for name, function in capabilities.items():
        losses[name] = [float(function(item)) for item in tasks]
        if len(losses[name]) != len(tasks):
            raise ValueError(f"capability {name} did not score every task")
        if any(math.isnan(value) or math.isinf(value) for value in losses[name]):
            raise ValueError(f"capability {name} produced a non-finite loss")
    return losses


class FixedRule:
    """The pre-registered rule. No learned parameters, no outcome."""

    name = "fixed-rule"
    deployable = True
    learned_parameters = 0

    def __init__(self, first: str = "D", second: str = "M",
                 threshold: float = FIXED_RULE_THRESHOLD):
        self.first = first
        self.second = second
        self.threshold = threshold

    def choose(self, features: list[float], names: list[str]) -> str:
        score = features[names.index("code_marker_score")]
        rust = features[names.index("rust_keyword_score")]
        return self.first if max(score, rust) >= self.threshold else self.second

    def cost(self) -> dict:
        return {"learned_parameters": 0, "elements_read_per_decision": 0,
                "needs_outcome": False, "deployable": True}


class AlwaysOne:
    """A baseline: always pick the same capability."""

    deployable = True
    learned_parameters = 0

    def __init__(self, capability: str):
        self.capability = capability
        self.name = f"always-{capability}"

    def choose(self, features: list[float], names: list[str]) -> str:
        return self.capability

    def cost(self) -> dict:
        return {"learned_parameters": 0, "elements_read_per_decision": 0,
                "needs_outcome": False, "deployable": True}


class OracleSelector:
    """Per-item best capability. Needs the outcome: not deployable."""

    name = "oracle"
    deployable = False
    learned_parameters = 0

    def __init__(self, losses: dict[str, list[float]], order: list[str]):
        self.order = order
        self.best = [min(order, key=lambda name: losses[name][index])
                     for index in range(len(losses[order[0]]))]

    def choose(self, features: list[float], names: list[str]) -> str:
        raise RuntimeError("the oracle is indexed by item, not by features")

    def cost(self) -> dict:
        return {"learned_parameters": 0, "elements_read_per_decision": 0,
                "needs_outcome": True, "deployable": False}


class TrainedRouter:
    """A logistic head on standardised features, trained once and frozen."""

    name = "trained-router"
    deployable = True

    def __init__(self, first: str, second: str):
        self.first = first
        self.second = second
        self.weights: torch.Tensor | None = None
        self.bias: torch.Tensor | None = None
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        self.training: dict = {}

    def fit(self, features: list[list[float]], labels: list[int],
            steps: int = ROUTER_STEPS, lr: float = ROUTER_LR,
            seed: int = ROUTER_SEED) -> dict:
        """`labels[i] = 1` if the first capability is better on item `i`."""
        data = torch.tensor(features, dtype=torch.float32)
        target = torch.tensor(labels, dtype=torch.float32)
        if data.shape[0] != target.shape[0]:
            raise ValueError("features and labels disagree on items")
        self.mean = data.mean(dim=0)
        self.std = data.std(dim=0, unbiased=False).clamp_min(1e-6)
        standardised = (data - self.mean) / self.std
        self.weights = torch.zeros(standardised.shape[1], requires_grad=True)
        self.bias = torch.zeros((), requires_grad=True)
        optimizer = torch.optim.Adam([self.weights, self.bias], lr=lr)
        trajectory = []
        for _step in range(steps):
            logits = standardised @ self.weights + self.bias
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            trajectory.append(float(loss.detach()))
        self.training = {"steps": steps, "lr": lr, "seed": seed,
                         "items": int(data.shape[0]),
                         "loss_first": trajectory[0], "loss_last": trajectory[-1],
                         "features": int(data.shape[1])}
        return self.training

    @property
    def learned_parameters(self) -> int:
        if self.weights is None:
            return 0
        return int(self.weights.numel() + 1)

    def probability_first(self, features: list[float]) -> float:
        if self.weights is None:
            raise RuntimeError("router is not fitted")
        value = torch.tensor(features, dtype=torch.float32)
        standardised = (value - self.mean) / self.std
        logit = (standardised @ self.weights + self.bias).detach()
        return float(torch.sigmoid(logit))

    def choose(self, features: list[float], names: list[str]) -> str:
        return self.first if self.probability_first(features) >= 0.5 else self.second

    def cost(self) -> dict:
        return {"learned_parameters": self.learned_parameters,
                "elements_read_per_decision": self.learned_parameters,
                "needs_outcome": False, "deployable": True}


# ---------------------------------------------------------------------------
# the language-model instrument


def completion_loss(model, tokenizer, item: dict, device, max_tokens: int = 8192) -> float:
    """Mean NLL per reference token under `model`, given the item's prompt.

    This is the capability score the selectors are evaluated on: lower is better.
    It reads the prompt and the **reference continuation only** — never the
    model's own generation — so two capabilities are compared on the same text
    and the paired comparison is exact.

    `model` follows the pinned interface: `model(input_ids=..., use_cache=False)`
    returning `.logits` of shape `[1, T, vocab]`.
    """
    prompt_ids = tokenizer(item["prompt"], return_tensors="pt",
                           add_special_tokens=True).input_ids
    reference_ids = tokenizer(item["reference"], return_tensors="pt",
                              add_special_tokens=False).input_ids
    if reference_ids.numel() == 0:
        raise ValueError(f"{item['id']}: the reference continuation is empty")
    ids = torch.cat([prompt_ids, reference_ids], dim=1)
    if ids.shape[1] > max_tokens:
        raise ValueError(f"{item['id']}: prompt plus reference exceeds {max_tokens} "
                         "tokens")
    ids = ids.to(device)
    with torch.inference_mode():
        logits = model(input_ids=ids, use_cache=False).logits.float()
    start = prompt_ids.shape[1]
    target = ids[:, start:]
    log_probs = torch.nn.functional.log_softmax(logits[:, start - 1:-1, :], dim=-1)
    values = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return float(values.mean())


def score_capabilities(models: dict, tokenizer, tasks: list[dict], device) -> dict:
    """`{capability: [completion loss per item]}` over the same items and text."""
    losses = {}
    for name, model in models.items():
        losses[name] = [completion_loss(model, tokenizer, item, device)
                        for item in tasks]
    return losses


def load_arm(model_size: str, checkpoint: str, config_path: str, admission_path: str,
             hf_root: str, device, seed: int = 1001):
    """Build one T2 arm from its checkpoint; the caller passes the result to
    `score_capabilities`. GPU-only in practice: the frozen core does not fit the
    CPU court's memory limit."""
    import harness
    from modeling import build_arm, load_admission

    base = harness.load_model(harness.model_path(hf_root, model_size), device)
    config, admission = load_admission(config_path, admission_path)
    arm_model = build_arm(base, config["arm"], config, admission, seed, profile="t2")
    arm_model.model.to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    arm_model.model.load_state_dict(state["model"], strict=False)
    arm_model.model.eval()
    return arm_model.model


# ---------------------------------------------------------------------------
# evaluation


def paired_bootstrap(differences: list[float], resamples: int = BOOTSTRAP_RESAMPLES,
                     seed: int = BOOTSTRAP_SEED, alpha: float = 0.05) -> dict:
    values = torch.tensor(differences, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    index = torch.randint(0, values.numel(), (resamples, values.numel()),
                          generator=generator)
    means = values[index].mean(dim=1)
    return {"mean": float(values.mean()),
            "low": float(torch.quantile(means, alpha / 2)),
            "high": float(torch.quantile(means, 1 - alpha / 2)),
            "n": int(values.numel()), "resamples": resamples, "seed": seed}


def evaluate(tasks: list[dict], losses: dict[str, list[float]],
             selectors: list, hidden_summaries: dict[str, list[float]] | None = None) -> dict:
    """Loss per selector, paired against each single-capability baseline."""
    order = list(losses)
    if len(order) < 2:
        raise ValueError("at least two capabilities are required")
    summaries = hidden_summaries or {}
    rows = {}
    for selector in selectors:
        picks = []
        for index, item in enumerate(tasks):
            if isinstance(selector, OracleSelector):
                picks.append(selector.best[index])
            else:
                features, feature_names = task_features(
                    item, hidden_summary=summaries.get(item["id"]))
                picks.append(selector.choose(features, feature_names))
        selected = [losses[pick][index] for index, pick in enumerate(picks)]
        best_possible = [min(losses[name][index] for name in order)
                         for index in range(len(tasks))]
        per_domain = {}
        for domain in fresh_tasks.DOMAINS:
            indices = [index for index, item in enumerate(tasks)
                       if item["domain"] == domain]
            if indices:
                per_domain[domain] = sum(selected[index] for index in indices) \
                    / len(indices)
        comparisons = {}
        for name in order:
            differences = [selected[index] - losses[name][index]
                           for index in range(len(tasks))]
            comparisons[f"minus-{name}"] = paired_bootstrap(differences)
        rows[selector.name] = {
            "mean_loss": sum(selected) / len(selected),
            "oracle_gap": sum(selected[index] - best_possible[index]
                              for index in range(len(tasks))) / len(tasks),
            "accuracy": sum(1.0 for index, pick in enumerate(picks)
                            if abs(losses[pick][index] - best_possible[index]) < 1e-12)
            / len(tasks),
            "picks": {name: sum(1 for pick in picks if pick == name) for name in order},
            "per_domain": per_domain,
            "comparisons": comparisons,
            "cost": selector.cost(),
        }
    return {"selectors": rows,
            "capability_mean": {name: sum(losses[name]) / len(losses[name])
                                for name in order},
            "tasks": len(tasks)}


def split_indices(tasks: list[dict]) -> tuple[list[int], list[int]]:
    """Alternating train/test split over the id-sorted set, fixed before use."""
    ordered = sorted(range(len(tasks)), key=lambda index: tasks[index]["id"])
    train = ordered[::2]
    test = ordered[1::2]
    return train, test


def flat_features(tasks: list[dict], indices: list[int],
                  hidden_summaries: dict[str, list[float]] | None = None
                  ) -> tuple[list[list[float]], list[str]]:
    summaries = hidden_summaries or {}
    rows, names = [], None
    for index in indices:
        item = tasks[index]
        values, names = task_features(item, hidden_summary=summaries.get(item["id"]))
        rows.append(values)
    return rows, names or list(FEATURE_NAMES)


def run(tasks: list[dict], capabilities: dict,
        hidden_summaries: dict[str, list[float]] | None = None) -> dict:
    losses = capability_losses(tasks, capabilities)
    order = list(losses)
    first, second = order[0], order[1]
    train, test = split_indices(tasks)

    labels = [1 if losses[first][index] < losses[second][index] else 0
              for index in train]
    router = TrainedRouter(first, second)
    train_features, _names = flat_features(tasks, train, hidden_summaries)
    training = router.fit(train_features, labels)

    # Every selector is scored on the *test* half only, on the same items, so the
    # comparisons are paired and the router never sees its evaluation items.
    test_tasks = [tasks[index] for index in test]
    test_losses = {name: [losses[name][index] for index in test] for name in order}
    selectors = [AlwaysOne(first), AlwaysOne(second),
                 FixedRule(first, second),
                 router,
                 OracleSelector(test_losses, order)]
    evaluation = evaluate(test_tasks, test_losses, selectors, hidden_summaries)
    evaluation["router_training"] = training
    evaluation["split"] = {"train": len(train), "test": len(test),
                           "rule": "alternating over the id-sorted set",
                           "scored_on": "the test half only"}
    return evaluation


# ---------------------------------------------------------------------------
# self-test capabilities (no model)


def synthetic_capabilities(tasks: list[dict]) -> dict:
    """Two complementary capabilities with a known answer, for CPU self-tests."""

    def first_capability(item: dict) -> float:
        # `D` is good on code and rust, bad on math and factual
        return 1.0 + 2.0 * (item["domain"] in ("math", "factual"))

    def second_capability(item: dict) -> float:
        # `M` is the mirror image
        return 1.0 + 2.0 * (item["domain"] in ("code", "rust"))

    return {"D": first_capability, "M": second_capability}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["self-test", "checkpoint"],
                        default="self-test")
    parser.add_argument("--out", required=True)
    parser.add_argument("--contamination", default=None,
                        help="the contamination receipt that clears this set")
    parser.add_argument("--model-size", default="0.6b", choices=["0.6b", "1.7b"])
    parser.add_argument("--hf-root", default=os.environ.get("HF_HOME", "/hf-cache"))
    for _arm in ("d", "m"):
        parser.add_argument(f"--arm-{_arm}", default=None)
        parser.add_argument(f"--config-{_arm}", default=None)
        parser.add_argument(f"--admission-{_arm}", default=None)
    args = parser.parse_args(argv)

    if args.mode == "checkpoint":
        if not torch.cuda.is_available():
            raise SystemExit("checkpoint mode requires the GPU court (--gpu); the "
                             "frozen 0.6B core does not fit the CPU court's memory "
                             "limit. Use --mode self-test for the CPU harness check.")
        import harness

        device = torch.device("cuda")
        tokenizer = harness.load_tokenizer(args.hf_root)
        tasks = list(fresh_tasks.TASKS)
        models, provenance = {}, {}
        for name in ("D", "M"):
            checkpoint = getattr(args, f"arm_{name.lower()}")
            config_path = getattr(args, f"config_{name.lower()}")
            admission_path = getattr(args, f"admission_{name.lower()}")
            if not (checkpoint and config_path and admission_path):
                raise SystemExit(f"--arm-{name.lower()} needs --config-{name.lower()} "
                                 f"and --admission-{name.lower()}")
            models[name] = load_arm(args.model_size, checkpoint, config_path,
                                    admission_path, args.hf_root, device)
            provenance[name] = {"checkpoint": checkpoint, "config": config_path,
                                "admission": admission_path}
        measured = score_capabilities(models, tokenizer, tasks, device)
        positions = {item["id"]: index for index, item in enumerate(tasks)}

        def lookup(name):
            return lambda item: measured[name][positions[item["id"]]]

        evaluation = run(tasks, {"D": lookup("D"), "M": lookup("M")})
        receipt_extra = {"capability_source": provenance,
                         "per_item_loss": {name: [round(value, 6) for value in values]
                                           for name, values in measured.items()}}
    else:
        tasks = list(fresh_tasks.TASKS)
        evaluation = run(tasks, synthetic_capabilities(tasks))
        receipt_extra = {}

    receipt = {
        "schema": SCHEMA,
        "stamp": stamp_now(),
        "revision": os.environ.get("SELVEDGE_REVISION", "unknown"),
        "created_utc": utc_now(),
        "mode": args.mode,
        "capabilities": (
            "synthetic: D = 1 + 2*[domain in {math, factual}], "
            "M = 1 + 2*[domain in {code, rust}] (lower is better)"
            if args.mode == "self-test" else
            "per-item mean NLL of the item's reference continuation under the "
            "trained T2 D and M checkpoints (lower is better)"),
        "task_set": fresh_tasks.manifest(),
        "contamination_receipt": args.contamination,
        "method": {
            "metric": "mean per-item loss of the selected capability (lower is better)",
            "paired_bootstrap": {"resamples": BOOTSTRAP_RESAMPLES,
                                 "seed": BOOTSTRAP_SEED, "kind": "percentile"},
            "rule": f"fixed-rule picks the first capability when max(code marker, "
                    f"rust keyword) score >= {FIXED_RULE_THRESHOLD}",
            "router": "logistic head on standardised prefix features, trained on "
                      "the alternating training half against the measured outcome",
        },
        "evaluation": evaluation,
        **receipt_extra,
    }
    write_json_atomic(os.path.join(REPO, args.out), receipt)
    print(json.dumps({
        "mode": args.mode,
        "selectors": {name: {"mean_loss": round(row["mean_loss"], 6),
                             "oracle_gap": round(row["oracle_gap"], 6),
                             "accuracy": round(row["accuracy"], 6),
                             "learned_parameters": row["cost"]["learned_parameters"]}
                      for name, row in evaluation["selectors"].items()},
        "out": args.out,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
