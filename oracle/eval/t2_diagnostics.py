#!/usr/bin/env python3
"""P12.2 — diagnose the `H*` router target from its labelled calibration slice.

`H*`'s router is trained on the listwise softmax of per-expert leave-one-out
utilities measured with the **full bank** present (T2.5). This tool measures
three things that decide whether that target asks the router the right question,
using only the label artifact and the router weights — no GPU, no new forward
passes:

1. **Deployed budget.** The router selects `k = 4` of `m = 8`; the labels were
   measured with all 8 present. From the labels alone the *deployed-budget*
   utility `L(S) - L(S \\ {e})` is not recoverable, so the tool computes what is
   recoverable and labels the rest `blocked`: (a) the router's additive capture
   `sum_{e in S_router} u_e` against the best top-4 by `u`; (b) whether the
   router's chosen set ever contains an expert with a *negative* marginal value
   ("least harmful omission"); (c) the exact command that produces the missing
   deployed-budget labels.
2. **Per-expert standardisation.** The registered target softmaxes
   `(u - mean_e) / std_e`. That removes the absolute utility magnitude — and with
   it the per-expert offset that the deployed decision actually cares about. The
   tool reports the KL between the absolute target and the standardised one, the
   per-expert means and spreads, and how much of the target's structure is a
   single common factor across experts (a first-principal-component share close
   to one means the listwise target is mostly ranking *tokens*, not experts).
3. **Identity versus token features.** The router is `z_e(t) = w_e . h_t`, which
   splits exactly into an expert-identity constant `w_e . mean_t(h)` and a
   token-varying part. The tool reports the variance share of the constant, how
   often the identity winner is the router's top-1, how often the router's top-4
   equals the utility's top-4, and the routing margin against the label noise —
   i.e. whether experts whose utility differs by less than the noise are
   distinguishable in principle.

Everything is deterministic (fixed bootstrap seed). Missing inputs fail closed.

Usage (CPU):

    scripts/court oracle -- python3 oracle/eval/t2_diagnostics.py \
        --labels results/artifacts/train/t2/Hstar/utility-labels.pt \
        --checkpoint results/artifacts/train/t2/Hstar/step-000064.pt \
        --out results/t2/diagnostics-$(date -u +%Y%m%dT%H%M%SZ).json

If the label artifact is absent, nothing is measured: the tool prints what is
missing and exits non-zero rather than reporting an empty diagnostic.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.dirname(HERE)
REPO = os.path.dirname(ORACLE)
SCHEMA = "selvedge/p12.2-diagnostics/v1"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261115


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
# loading


def load_labels(path: str) -> dict:
    """Load the label artifact without faulting 3.8 GiB of features at once."""
    sidecar = path + ".sha256"
    if os.path.exists(sidecar):
        recorded = json.load(open(sidecar, encoding="utf-8"))
        if recorded["sha256"] != sha256_file(path):
            raise SystemExit(f"labels {path}: sha256 does not match the sidecar")
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (RuntimeError, ValueError):
        # Older artifacts are not zipfile-backed; fall back to a plain load.
        return torch.load(path, map_location="cpu", weights_only=False)


ROUTER_KEY = re.compile(r"^model\.layers\.(\d+)\..*\.router\.weight$")


def routers_from_state(state: dict) -> dict[int, torch.Tensor]:
    routers = {}
    for name, tensor in state.items():
        match = ROUTER_KEY.match(name)
        if match:
            routers[int(match.group(1))] = tensor.detach().to(torch.float32)
    return routers


def load_routers(checkpoint: str) -> dict[int, torch.Tensor]:
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    except (RuntimeError, ValueError):
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "model" in state:
        state = state["model"]
    routers = routers_from_state(state)
    if not routers:
        raise SystemExit(f"{checkpoint}: no *.router.weight tensors")
    return routers


# ---------------------------------------------------------------------------
# statistics


def _softmax(values: torch.Tensor) -> torch.Tensor:
    return torch.softmax(values.double(), dim=-1)


def _kl(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first = first.clamp_min(1e-300)
    second = second.clamp_min(1e-300)
    return (first * (first.log() - second.log())).sum(dim=-1)


def _standardise(utilities: torch.Tensor) -> torch.Tensor:
    mean = utilities.mean(dim=0, keepdim=True)
    std = utilities.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (utilities - mean) / std


def bootstrap_mean_ci(values: torch.Tensor, resamples: int = BOOTSTRAP_RESAMPLES,
                      seed: int = BOOTSTRAP_SEED, alpha: float = 0.05) -> dict:
    generator = torch.Generator().manual_seed(seed)
    count = values.numel()
    index = torch.randint(0, count, (resamples, count), generator=generator)
    means = values.double()[index].mean(dim=1)
    lower = torch.quantile(means, alpha / 2).item()
    upper = torch.quantile(means, 1 - alpha / 2).item()
    return {"mean": float(values.double().mean()), "low": lower, "high": upper,
            "resamples": resamples, "seed": seed}


def analyse_layer(features: torch.Tensor, utilities: torch.Tensor,
                  router: torch.Tensor, k: int = 4, tau: float = 1.0) -> dict:
    """The three diagnostics for one layer.

    `features` is `[N, hidden]` (the pre-expert hidden state), `utilities` is
    `[N, m]` (`u_e(t) = L_{-e} - L_{+all}`) and `router` is `[m, hidden]`.
    """
    features = features.detach().to(torch.float32)
    utilities = utilities.detach().to(torch.float32)
    router = router.detach().to(torch.float32)
    if features.shape[0] != utilities.shape[0]:
        raise ValueError("features and utilities disagree on positions")
    if router.shape != (utilities.shape[1], features.shape[1]):
        raise ValueError("router shape does not match (experts, hidden)")
    positions, experts = utilities.shape

    # -- (2) per-expert standardisation: what the registered target removes.
    mean = utilities.mean(dim=0)
    std = utilities.std(dim=0, unbiased=False).clamp_min(1e-6)
    standardised = (utilities - mean) / std
    absolute_target = _softmax(utilities / tau)
    relative_target = _softmax(standardised / tau)
    kl_absolute_vs_relative = _kl(absolute_target, relative_target)
    identity_target = _softmax(mean.reshape(1, -1).expand(positions, experts) / tau)
    identity_share = (
        (utilities.argmax(dim=-1) == mean.argmax()).double().mean())

    # common factor across experts: if one component dominates, the target is
    # mostly ranking tokens by a shared "how well does this token fit at all".
    centered = utilities - utilities.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered.double())
    energy = singular.pow(2)
    first_component_share = float(energy[0] / energy.sum().clamp_min(1e-30)) \
        if float(energy.sum()) > 0 else 0.0
    correlation = torch.corrcoef(utilities.double().T).nan_to_num(nan=0.0)
    off_diagonal = correlation[~torch.eye(experts, dtype=torch.bool)]
    mean_correlation = float(off_diagonal.mean())
    mean_absolute_correlation = float(off_diagonal.abs().mean())

    # -- (3) margin against noise: can the router tell 4th from 5th at all?
    ranked = utilities.sort(dim=-1, descending=True).values
    margin = ranked[:, k - 1] - ranked[:, k] if k < experts else torch.zeros(positions)
    margin_stats = bootstrap_mean_ci(margin)
    noise = float(std.median())
    margin_median = float(margin.median())

    # -- (1) the deployed budget, as far as the labels allow.
    logits = features.double() @ router.double().T
    constant = router.double() @ features.double().mean(dim=0)          # [m]
    varying = logits - constant.reshape(1, -1)
    between = constant.var(unbiased=False)
    within = varying.var(dim=0, unbiased=False).mean()
    identity_bias_share = float(between / (between + within).clamp_min(1e-30))

    top_router = logits.topk(k, dim=-1).indices
    top_utility = utilities.topk(k, dim=-1).indices
    top_identity = constant.topk(k).indices.expand(positions, k)
    router_top1_is_identity = float((logits.argmax(dim=-1) == constant.argmax())
                                    .double().mean())
    overlap_utility = _set_overlap(top_router, top_utility)
    overlap_identity = _set_overlap(top_router, top_identity)

    chosen = utilities.gather(1, top_router).sum(dim=-1)
    best = utilities.gather(1, top_utility).sum(dim=-1)
    least_harmful = utilities.gather(1, top_router).min(dim=-1).values
    added = utilities.gather(1, top_router)

    return {
        "positions": positions,
        "experts": experts,
        "label_mean": [float(item) for item in mean],
        "label_std": [float(item) for item in std],
        "kl_absolute_vs_relative_mean": float(kl_absolute_vs_relative.mean()),
        "kl_absolute_vs_relative_p95": float(torch.quantile(
            kl_absolute_vs_relative, 0.95)),
        "identity_argmax_share": float(identity_share),
        "first_component_energy_share": first_component_share,
        "mean_pairwise_utility_correlation": mean_correlation,
        "mean_absolute_pairwise_utility_correlation": mean_absolute_correlation,
        "routing_margin_median": margin_median,
        "routing_margin_mean_ci": margin_stats,
        "label_noise_median": noise,
        "margin_over_noise": margin_median / noise if noise else None,
        "router_identity_bias_share": identity_bias_share,
        "router_top1_is_identity_winner": router_top1_is_identity,
        "router_top4_overlap_with_utility": overlap_utility,
        "router_top4_overlap_with_identity": overlap_identity,
        "router_additive_gap_mean": float((best - chosen).mean()),
        "router_additive_capture_mean": float((chosen / best.abs().clamp_min(1e-9)).mean()),
        # `u_e < 0` means adding `e` to the full bank *reduces* the loss the wrong
        # way round: it is a harmful expert. The deployed set is an addition, so
        # this is the "least harmful omission" share.
        "router_least_harmful_share": float((least_harmful >= 0).double().mean()),
        "router_chosen_marginal_min_mean": float(least_harmful.mean()),
        "router_chosen_negative_marginal_pair_share": float((added < 0).double().mean()),
        "deployed_budget_utility_available": False,
    }


def _set_overlap(first: torch.Tensor, second: torch.Tensor) -> float:
    """Mean intersection size of two `[N, k]` top-k sets, as a fraction of k."""
    if first.shape != second.shape:
        raise ValueError("top-k sets disagree in shape")
    total = 0.0
    for left, right in zip(first.tolist(), second.tolist()):
        total += len(set(left) & set(right)) / len(left)
    return total / first.shape[0]


def analyse(payload: dict, routers: dict[int, torch.Tensor], k: int = 4,
            tau: float = 1.0) -> dict:
    features = payload["features"]
    utilities = payload["utilities"]
    layers = sorted(set(features) & set(utilities))
    if not layers:
        raise SystemExit("the artifact has no layer present in both features and labels")
    missing = [layer for layer in layers if layer not in routers]
    if missing:
        raise SystemExit(f"no router weight for layers {missing}")

    rows = {}
    for layer in layers:
        rows[str(layer)] = analyse_layer(features[layer], utilities[layer],
                                         routers[layer], k=k, tau=tau)

    def mean_of(key):
        values = [row[key] for row in rows.values() if row[key] is not None]
        return {"mean": sum(values) / len(values), "min": min(values),
                "max": max(values)} if values else None

    summary = {key: mean_of(key) for key in (
        "kl_absolute_vs_relative_mean", "identity_argmax_share",
        "first_component_energy_share", "mean_pairwise_utility_correlation",
        "mean_absolute_pairwise_utility_correlation",
        "routing_margin_median", "label_noise_median", "margin_over_noise",
        "router_identity_bias_share", "router_top1_is_identity_winner",
        "router_top4_overlap_with_utility", "router_top4_overlap_with_identity",
        "router_additive_gap_mean", "router_least_harmful_share",
        "router_chosen_marginal_min_mean",
        "router_chosen_negative_marginal_pair_share",
    )}
    return {"layers": len(layers), "k": k, "tau": tau,
            "per_layer": rows, "summary": summary}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="step checkpoint the labels were measured from; the "
                             "router weights are read from it")
    parser.add_argument("--routers", default=None,
                        help="a .pt of {layer: [experts, hidden]} extracted earlier")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    if not os.path.exists(args.labels):
        print(f"t2_diagnostics: label artifact {args.labels} is absent. H*'s Phase U "
              "has not run, so there is nothing to diagnose. Produce it with "
              "`oracle/train/loop.py --arm H* --warmup-steps 64 ... "
              "--labels-out <path>` and re-run.", file=sys.stderr)
        return 66

    payload = load_labels(args.labels)
    if args.routers:
        routers = torch.load(args.routers, map_location="cpu", weights_only=False)
        routers = {int(key) if not isinstance(key, int) else key: value
                   for key, value in routers.items()}
        router_source = {"path": args.routers, "sha256": sha256_file(args.routers)}
    elif args.checkpoint:
        routers = load_routers(args.checkpoint)
        router_source = {"path": args.checkpoint,
                         "sha256": sha256_file(args.checkpoint),
                         "keys": "*.router.weight, all matches"}
    else:
        raise SystemExit("one of --checkpoint or --routers is required")

    result = analyse(payload, routers, k=args.k, tau=args.tau)
    manifest = payload.get("manifest", {})
    receipt = {
        "schema": SCHEMA,
        "stamp": stamp_now(),
        "revision": os.environ.get("SELVEDGE_REVISION", "unknown"),
        "created_utc": utc_now(),
        "arm": manifest.get("arm"),
        "labels": {"path": args.labels, "sha256": sha256_file(args.labels),
                   "manifest": manifest},
        "router": router_source,
        "method": {
            "utility": "u_e(t) = L(with all m) - L(without e), full-bank reference",
            "target": "softmax((u - mean_e) / std_e) at tau",
            "router_split": "z_e(t) = w_e . mean_t(h)  +  w_e . (h_t - mean_t(h))",
            "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
                          "kind": "percentile, per-layer, over calibration positions"},
            "noise": "per-expert label standard deviation over calibration positions",
        },
        "result": result,
        "blocked": [
            {"what": "leave-one-out in the deployed top-k set versus the full bank",
             "why": "u_e is measured with all m present; the deployed-budget utility "
                    "needs new passes with the router's own top-k mask",
             "how": "scripts/court --gpu oracle -- python3 oracle/train/loop.py "
                    "--arm H* --profile t2 --warmup-steps 64 --steps 65 "
                    "--calibration-windows 32 --labels-out "
                    "results/artifacts/train/t2/Hstar/deployed-labels.pt "
                    "(with reference_set=deployed once utility.py exposes it), then "
                    "re-run t2_diagnostics on both artifacts and compare the ranks"},
        ],
    }
    write_json_atomic(os.path.join(REPO, args.out), receipt)
    print(json.dumps({
        "layers": result["layers"],
        "summary": {key: (round(value["mean"], 6) if value else None)
                    for key, value in result["summary"].items()},
        "out": args.out,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
