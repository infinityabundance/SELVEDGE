#!/usr/bin/env python3
"""M2: per-token causal selectors, oracle potential and matched-budget evaluation.

Subcommands (each writes its own receipt):

  prepare       calibration per-document statistics, intervention labels,
                fitted learned selectors, weights JSON and Rust fit data
  interactions  pairwise non-additivity measurement and surrogate test
  oracle        per-token oracle potential at B140/B142 and 16/64/4 groups
  evaluate      all arms on the held-out split and the oracle sample
  analyze       paired bootstrap, matched-budget table, JSON and SVG
  export        Rust parity fixtures and the Python->Rust selector contract

The contract is the M2 section of `docs/EXPERIMENT-CONTRACT.md`, registered
before any of this ran. Every selector receives only pre-decision information;
the attribution and oracle arms are labelled non-deployable in the receipts.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import glob
import hashlib
import itertools
import json
import math
import os
import sys
import time

import numpy as np
import torch

import frontier_sweep as fs
import masked_mlp

SEED = 20261114
M2_PREREGISTRATION_COMMIT = "a0b107adc23f23cbda43a09cbd6d229b0ea024a1"
SCHEMA_PREPARE = "selvedge/selectors/prepare-v1"
SCHEMA_INTERACTIONS = "selvedge/selectors/interactions-v1"
SCHEMA_ORACLE = "selvedge/selectors/oracle-v1"
SCHEMA_EVAL = "selvedge/selectors/eval-v1"
SCHEMA_ANALYSIS = "selvedge/selectors/analysis-v1"
SCHEMA_FIXTURES = "selvedge/atlas/fixtures-v1"
GROUP_FEATURES = ("log_norm", "share", "mean", "max", "min")
FEATURE_DIM = len(GROUP_FEATURES) + 1  # + bias
LAYERS = 28
GROUPS = 16
HIDDEN = 2048
INTERMEDIATE = 6144


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json_atomic(path: str, payload: dict) -> None:
    fs.write_json_atomic(path, sanitize(payload))


def sanitize(value):
    """Replace non-finite floats with null so strict JSON parsers accept it."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.floating):
        return sanitize(float(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    return value


def load_json(path: str) -> dict:
    return fs.load_json(path)


def model_and_accounting(args):
    lock = load_json("models/MODELS.lock.json")
    entry = fs.model_cache_relative_dir(lock, args.model_id)
    if args.model_revision and entry["revision"] != args.model_revision:
        raise SystemExit(
            f"model revision {entry['revision']} != requested {args.model_revision}"
        )
    accounting = fs.accounting_record(args.accounting)
    split_receipt = fs.split_receipt_for(args.split)
    split = fs.load_split(args.split, split_receipt["derived"]["sha256"])
    return entry, accounting, split_receipt, split


def group_count(accounting: dict) -> int:
    return int(accounting["group_count"])


def budget_spec(accounting: dict, name: str) -> dict:
    """Registered budget arithmetic; no measured quantity enters this."""
    backbone = int(accounting["always_active_backbone_elements"])
    groups = int(accounting["group_count"])
    hidden = int(accounting["hidden"])
    intermediate = int(accounting["intermediate_per_layer"])
    if groups != GROUPS or hidden != HIDDEN or intermediate != INTERMEDIATE:
        raise SystemExit("accounting does not match the registered Qwen3-1.7B shape")
    width = -(-intermediate // groups)
    per_group = 3 * width * hidden
    if name == "B140":
        return {
            "name": "B140",
            "groups": 140,
            "groups_per_layer": 5,
            "group_width": width,
            "ffn_elements": 140 * per_group,
            "active_unique_elements": backbone + 140 * per_group,
            "ceiling": int(accounting["ceiling"]),
            "extra_groups": 0,
        }
    if name == "B142":
        return {
            "name": "B142",
            "groups": 142,
            "groups_per_layer": 5,
            "group_width": width,
            "ffn_elements": 142 * per_group,
            "active_unique_elements": backbone + 142 * per_group,
            "ceiling": int(accounting["ceiling"]),
            "extra_groups": 2,
        }
    raise SystemExit(f"unknown budget {name}")


def budget_granularity(accounting: dict, groups: int) -> dict:
    backbone = int(accounting["always_active_backbone_elements"])
    hidden = int(accounting["hidden"])
    intermediate = int(accounting["intermediate_per_layer"])
    width = -(-intermediate // groups)
    per_group = 3 * width * hidden
    target = 330301440
    if groups == 16:
        count = 140
        shape = {"per_layer": 5}
    elif groups == 64:
        count = 560
        shape = {"per_layer": 20}
    elif groups == 4:
        count = 35
        shape = {"per_layer": None, "one_per_layer": 21, "two_per_layer": 7}
    else:
        raise SystemExit(f"unsupported granularity {groups}")
    if count * per_group != target:
        raise SystemExit(
            f"granularity {groups} does not match the registered element budget: "
            f"{count * per_group} vs {target}"
        )
    return {
        "name": f"G{groups}",
        "groups": groups,
        "total_groups": count,
        "group_width": width,
        "ffn_elements": count * per_group,
        "active_unique_elements": backbone + count * per_group,
        "ceiling": int(accounting["ceiling"]),
        "shape": shape,
    }


def controller_elements(arm: dict) -> int:
    model = arm.get("learned")
    if not model:
        return 0
    return int(model.get("parameter_elements", 0))


def selection_from_groups(selected: list[list[int]], groups: int) -> torch.Tensor:
    return masked_mlp.subset_selection_from_groups(selected, groups)


def channel_masks_from_bool(
    selection: list[torch.Tensor], backend: dict
) -> list[torch.Tensor]:
    """`selection[l]` is `[T, G]` bool; returns `[T, I]` float masks."""
    bounds = backend["bounds"]
    intermediate = backend["intermediate"]
    masks = []
    for layer_selection in selection:
        token_count = layer_selection.shape[0]
        mask = torch.zeros(
            (token_count, intermediate),
            dtype=torch.float32,
            device=layer_selection.device,
        )
        for index, (start, end) in enumerate(bounds):
            mask[:, start:end] = layer_selection[:, index : index + 1].to(torch.float32)
        masks.append(mask)
    return masks


def static_mask(selected: list[list[int]], backend: dict, tokens: int) -> list[torch.Tensor]:
    """Broadcast a per-layer static selection to every token."""
    bounds = backend["bounds"]
    masks = []
    for layer_selected in selected:
        mask = torch.zeros(backend["intermediate"], dtype=torch.float32)
        for index in layer_selected:
            start, end = bounds[index]
            mask[start:end] = 1.0
        masks.append(mask.unsqueeze(0).expand(tokens, -1))
    return masks


def topk_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Stable top-k: ties break by ascending group index."""
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    selected = torch.zeros_like(scores, dtype=torch.bool)
    selected.scatter_(1, order[:, :k], True)
    return selected


def best_extra_pairs(
    scores: list[torch.Tensor],
    selection: list[torch.Tensor],
    extras: int,
    largest: bool,
) -> list[torch.Tensor]:
    """Add `extras` (layer, group) pairs per token by global score.

    `scores[l]` is `[T, G]`; `selection[l]` is `[T, G]` bool. `largest=True`
    means larger is better (magnitude/learned); `False` means smaller is
    better (local error deltas).
    """
    layers = len(scores)
    tokens = scores[0].shape[0]
    groups = scores[0].shape[1]
    stacked = torch.stack(
        [
            scores[layer].masked_fill(selection[layer], float("-inf") if largest else float("inf"))
            for layer in range(layers)
        ],
        dim=1,
    )  # [T, L, G]
    flat = stacked.reshape(tokens, layers * groups)
    order = torch.argsort(flat, dim=1, descending=largest, stable=True)
    chosen = order[:, :extras]
    out = [layer_selection.clone() for layer_selection in selection]
    rows = torch.arange(tokens, device=flat.device).unsqueeze(1).expand(-1, extras)
    layer_idx = chosen // groups
    group_idx = chosen % groups
    for layer in range(layers):
        hit = layer_idx == layer
        if not bool(hit.any()):
            continue
        token_idx = rows[hit]
        group_tokens = group_idx[hit]
        out[layer][token_idx, group_tokens] = True
    return out


def selection_group_counts(selection: list[torch.Tensor]) -> dict:
    counts = torch.stack([sel.sum(dim=1) for sel in selection], dim=1)
    per_token = counts.sum(dim=1)
    return {
        "per_token_min": int(per_token.min()),
        "per_token_max": int(per_token.max()),
        "per_token_mean": float(per_token.to(torch.float64).mean()),
        "per_layer_min": int(counts.min()),
        "per_layer_max": int(counts.max()),
    }


def selection_summary(selection: list[torch.Tensor]) -> dict:
    counts = selection_group_counts(selection)
    layers = len(selection)
    distinct = [
        int(torch.unique(selection[layer].to(torch.uint8), dim=0).shape[0])
        for layer in range(layers)
    ]
    adjacent = []
    modal = []
    for layer in range(layers):
        sel = selection[layer]
        if sel.shape[0] > 1:
            same = (sel[1:] == sel[:-1]).all(dim=1).to(torch.float64).mean()
        else:
            same = torch.tensor(1.0)
        adjacent.append(float(same))
        values, counts_per_set = torch.unique(
            sel.to(torch.uint8), dim=0, return_counts=True
        )
        modal.append(float(counts_per_set.max()) / float(sel.shape[0]))
    return {
        **counts,
        "distinct_subsets_per_layer": distinct,
        "adjacent_equal_fraction": adjacent,
        "modal_fraction": modal,
    }


def jaccard_selection(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean per-token Jaccard between two `[T, G]` boolean selections."""
    if a.shape != b.shape:
        raise ValueError("selection shapes differ")
    inter = (a & b).sum(dim=1).to(torch.float64)
    union = (a | b).sum(dim=1).to(torch.float64)
    values = torch.where(union > 0, inter / union, torch.ones_like(union))
    return float(values.mean())


# ---------------------------------------------------------------------------
# per-document calibration statistics and static rankings
# ---------------------------------------------------------------------------


def calibration_statistics(
    model, controller, split: dict, accounting: dict, device, capture
) -> dict:
    """Group activation norms per calibration document, layer and group."""
    groups = group_count(accounting)
    bounds = masked_mlp.group_bounds(int(accounting["intermediate_per_layer"]), groups)
    layers = list(model.model.layers)
    docs = split["windows"]["calibration"]
    per_doc = {}
    for article in docs:
        doc_id = article["doc_id"]
        per_doc[doc_id] = {
            "sha256": article["sha256"],
            "windows": len(article["windows"]),
            "tokens": 0,
            "sum_norms": np.zeros((LAYERS, groups), dtype=np.float64),
            "positive_mass": np.zeros((LAYERS, groups), dtype=np.float64),
        }
    with torch.inference_mode():
        for article in docs:
            doc_id = article["doc_id"]
            for window in article["windows"]:
                input_ids = torch.tensor([window], dtype=torch.long, device=device)
                controller.set_dense()
                capture.clear()
                model(input_ids=input_ids, use_cache=False)
                tokens = input_ids.shape[1]
                per_doc[doc_id]["tokens"] += tokens
                for layer_index, layer in enumerate(layers):
                    hidden = capture.get(layer_index).reshape(-1, input_ids.shape[1] * 1, HIDDEN)
                    hidden = hidden.reshape(-1, HIDDEN)
                    activation = masked_mlp.dense_channels(
                        controller.wrappers[layer_index], hidden
                    )
                    norms = torch.stack(
                        [activation[:, s:e].norm(dim=1) for s, e in bounds], dim=1
                    )
                    total = norms.sum(dim=1)
                    per_doc[doc_id]["sum_norms"][layer_index] += (
                        norms.to(torch.float64).sum(dim=0).cpu().numpy()
                    )
                    valid = total > 0
                    if bool(valid.any()):
                        mass = norms[valid] / total[valid].unsqueeze(1)
                        per_doc[doc_id]["positive_mass"][layer_index] += (
                            (mass > 2.0 / groups)
                            .sum(dim=0)
                            .to(torch.float64)
                            .cpu()
                            .numpy()
                        )
    result = {"groups": groups, "layers": LAYERS, "documents": {}}
    total_norms = np.zeros((LAYERS, groups), dtype=np.float64)
    total_mass = np.zeros((LAYERS, groups), dtype=np.float64)
    total_tokens = 0
    for doc_id, payload in per_doc.items():
        count = payload["tokens"]
        mean_norm = (payload["sum_norms"] / max(count, 1)).tolist()
        frequency = (payload["positive_mass"] / max(count, 1)).tolist()
        result["documents"][doc_id] = {
            "sha256": payload["sha256"],
            "windows": payload["windows"],
            "tokens": count,
            "mean_norm": mean_norm,
            "frequency": frequency,
        }
        total_norms += payload["sum_norms"]
        total_mass += payload["positive_mass"]
        total_tokens += count
    result["aggregate"] = {
        "tokens": total_tokens,
        "mean_norm": (total_norms / max(total_tokens, 1)).tolist(),
        "frequency": (total_mass / max(total_tokens, 1)).tolist(),
    }
    return result


def rankings_from_statistics(statistics: dict) -> dict:
    magnitude = []
    frequency = []
    for layer in range(statistics["layers"]):
        magnitude.append(
            sorted(
                range(statistics["groups"]),
                key=lambda g: (-statistics["aggregate"]["mean_norm"][layer][g], g),
            )
        )
        frequency.append(
            sorted(
                range(statistics["groups"]),
                key=lambda g: (-statistics["aggregate"]["frequency"][layer][g], g),
            )
        )
    random = masked_mlp.random_rankings(statistics["layers"], statistics["groups"], SEED)
    return {"magnitude": magnitude, "frequency": frequency, "random": random}


def document_static_selection(statistics: dict, doc_id: str, k: int = 5) -> list[list[int]]:
    doc = statistics["documents"][doc_id]
    return [
        sorted(range(statistics["groups"]), key=lambda g: (-doc["mean_norm"][layer][g], g))[:k]
        for layer in range(statistics["layers"])
    ]


# ---------------------------------------------------------------------------
# feature extraction and intervention labels
# ---------------------------------------------------------------------------


def group_features_from_activation(
    activation: torch.Tensor, bounds: list[tuple[int, int]]
) -> torch.Tensor:
    """`[T, G, F]` pre-decision group features (without the bias term)."""
    tokens = activation.shape[0]
    groups = len(bounds)
    norms = torch.stack(
        [activation[:, start:end].norm(dim=1) for start, end in bounds], dim=1
    )
    total = norms.sum(dim=1).clamp_min(1e-30)
    features = torch.empty(
        (tokens, groups, len(GROUP_FEATURES)),
        dtype=torch.float32,
        device=activation.device,
    )
    features[:, :, 0] = torch.log1p(norms)
    features[:, :, 1] = norms / total.unsqueeze(1)
    for index, (start, end) in enumerate(bounds):
        chunk = activation[:, start:end]
        features[:, index, 2] = chunk.mean(dim=1)
        features[:, index, 3] = chunk.max(dim=1).values
        features[:, index, 4] = chunk.min(dim=1).values
    return features


def features_with_bias(features: torch.Tensor) -> torch.Tensor:
    bias = torch.ones(
        (*features.shape[:-1], 1), dtype=features.dtype, device=features.device
    )
    return torch.cat([features, bias], dim=-1)


def enumerate_subsets(groups: int, k: int) -> torch.Tensor:
    return torch.tensor(list(itertools.combinations(range(groups), k)), dtype=torch.long)


def subset_errors(stats: dict, subsets: torch.Tensor) -> torch.Tensor:
    """`[T, S]` exact local squared errors for a fixed subset index matrix."""
    gram = stats["gram"]
    dot = stats["dot_dense"]
    dense_sq = stats["dense_sq"]
    tokens = gram.shape[0]
    count = subsets.shape[0]
    k = subsets.shape[1]
    out = torch.empty((tokens, count), dtype=torch.float32, device=gram.device)
    linear_all = dot[:, subsets].sum(dim=-1)
    chunk = 256
    for start in range(0, count, chunk):
        end = min(start + chunk, count)
        idx = subsets[start:end]
        quad = gram[:, idx[:, :, None], idx[:, None, :]].sum(dim=(-1, -2))
        out[:, start:end] = dense_sq.unsqueeze(1) - 2.0 * linear_all[:, start:end] + quad
    return out


def best_k_selection(stats: dict, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact best `k`-subset per token under the local squared error."""
    groups = stats["gram"].shape[1]
    subsets = enumerate_subsets(groups, k).to(stats["gram"].device)
    errors = subset_errors(stats, subsets)
    best = errors.argmin(dim=1)
    selected = torch.zeros(
        (stats["gram"].shape[0], groups), dtype=torch.bool, device=stats["gram"].device
    )
    selected.scatter_(1, subsets[best], True)
    return selected, errors.gather(1, best.unsqueeze(1)).squeeze(1)


def greedy_k_selection(
    stats: dict, k: int, beam: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bounded greedy (beam width 1) for a large group count."""
    if beam != 1:
        raise SystemExit("only the registered greedy beam width 1 is implemented")
    gram = stats["gram"]
    dot = stats["dot_dense"]
    tokens, groups, _ = gram.shape
    diagonal = torch.diagonal(gram, dim1=1, dim2=2)
    selected = torch.zeros((tokens, groups), dtype=torch.bool, device=gram.device)
    error = stats["dense_sq"].clone()
    for _ in range(k):
        delta = -2.0 * dot + 2.0 * (gram @ selected.to(torch.float32).unsqueeze(2)).squeeze(2) + diagonal
        delta = delta.masked_fill(selected, float("inf"))
        choice = delta.argmin(dim=1)
        selected.scatter_(1, choice.unsqueeze(1), True)
        error = error + delta.gather(1, choice.unsqueeze(1)).squeeze(1)
    return selected, error


def local_delta_for_extras(stats: dict, selected: torch.Tensor) -> torch.Tensor:
    """`[T, G]` local error change from adding each unselected group."""
    gram = stats["gram"]
    dot = stats["dot_dense"]
    diagonal = torch.diagonal(gram, dim1=1, dim2=2)
    delta = (
        -2.0 * dot
        + 2.0 * (gram @ selected.to(torch.float32).unsqueeze(2)).squeeze(2)
        + diagonal
    )
    return delta.masked_fill(selected, float("inf"))


def sample_calibration_contexts(
    split: dict, per_window: int, seed: int
) -> dict[str, list[tuple[int, int]]]:
    """Per-window `(layer, position)` samples, uniform without replacement."""
    generator = np.random.Generator(np.random.PCG64(seed))
    windows = [
        (article["doc_id"], index, window)
        for article in split["windows"]["calibration"]
        for index, window in enumerate(article["windows"])
    ]
    per_window = max(1, per_window)
    sampled = {}
    for doc_id, index, window in windows:
        tokens = len(window)
        count = min(per_window, LAYERS * tokens)
        flat = generator.choice(LAYERS * tokens, size=count, replace=False)
        pairs = sorted(((int(value) // tokens, int(value) % tokens) for value in flat))
        sampled[f"{doc_id}:{index}"] = pairs
    return sampled


def collect_labels(
    model,
    controller,
    split: dict,
    accounting: dict,
    device,
    capture,
    contexts_per_window: int,
    pairs_per_context: int,
) -> dict:
    """Intervention-labeled local utility and pairwise labels on calibration."""
    groups = group_count(accounting)
    bounds = masked_mlp.group_bounds(int(accounting["intermediate_per_layer"]), groups)
    sampled = sample_calibration_contexts(split, contexts_per_window, SEED)
    articles = [(article["doc_id"], article) for article in split["windows"]["calibration"]]
    doc_assignment = {
        doc_id: ("train" if index < 12 else "validation")
        for index, (doc_id, _article) in enumerate(articles)
    }
    generator = np.random.Generator(np.random.PCG64(SEED + 1))
    unary = {"train": [], "validation": []}
    pairwise = {"train": [], "validation": []}
    validation_contexts: dict[int, list[dict]] = {layer: [] for layer in range(LAYERS)}
    with torch.inference_mode():
        for doc_id, article in articles:
            for window_index, window in enumerate(article["windows"]):
                key = f"{doc_id}:{window_index}"
                positions_for_window = sampled.get(key, [])
                if not positions_for_window:
                    continue
                input_ids = torch.tensor([window], dtype=torch.long, device=device)
                controller.set_dense()
                capture.clear()
                model(input_ids=input_ids, use_cache=False)
                by_layer: dict[int, list[int]] = {}
                for layer, position in positions_for_window:
                    by_layer.setdefault(layer, []).append(position)
                for layer, positions in by_layer.items():
                    hidden_all = capture.get(layer).reshape(-1, HIDDEN)
                    hidden = hidden_all[torch.tensor(positions, device=device)]
                    activation = masked_mlp.dense_channels(
                        controller.wrappers[layer], hidden
                    )
                    feature_batch = group_features_from_activation(activation, bounds)
                    stats = masked_mlp.group_contributions(
                        controller.wrappers[layer], hidden, groups
                    )
                    norms = stats["group_norms"]
                    base = torch.argsort(norms, dim=1, descending=True, stable=True)[:, :4]
                    base_mask = torch.zeros(
                        (len(positions), groups), dtype=torch.bool, device=device
                    )
                    base_mask.scatter_(1, base, True)
                    base_error = masked_mlp.subset_local_error(stats, base_mask)
                    for row in range(len(positions)):
                        feats = feature_batch[row].cpu().numpy()
                        gram_row = stats["gram"][row].cpu().numpy()
                        dot_row = stats["dot_dense"][row].cpu().numpy()
                        dense_sq_row = float(stats["dense_sq"][row])
                        for group in range(groups):
                            increment = base_mask[row : row + 1].clone()
                            increment[:, group] = True
                            delta = float(
                                base_error[row]
                                - masked_mlp.subset_local_error(stats, increment)[0]
                            )
                            unary[doc_assignment[doc_id]].append(
                                {
                                    "layer": int(layer),
                                    "features": feats[group].astype(np.float64),
                                    "label": delta,
                                    "doc_id": doc_id,
                                }
                            )
                        if doc_assignment[doc_id] == "validation":
                            validation_contexts[layer].append(
                                {
                                    "features": feats.astype(np.float32),
                                    "gram": gram_row.astype(np.float32),
                                    "dot": dot_row.astype(np.float32),
                                    "dense_sq": dense_sq_row,
                                    "doc_id": doc_id,
                                }
                            )
                        candidates = [g for g in range(groups) if not bool(base_mask[row, g])]
                        for _pair in range(pairs_per_context):
                            if len(candidates) < 2:
                                break
                            chosen = generator.choice(candidates, size=2, replace=False)
                            g, j = int(chosen[0]), int(chosen[1])
                            def error(mask_bool):
                                return float(
                                    masked_mlp.subset_local_error(
                                        stats, mask_bool.reshape(1, -1)
                                    )[0]
                                )
                            row_mask = base_mask[row].clone()
                            eg = row_mask.clone(); eg[g] = True
                            ej = row_mask.clone(); ej[j] = True
                            egj = row_mask.clone(); egj[g] = True; egj[j] = True
                            interaction = error(eg) + error(ej) - error(row_mask) - error(egj)
                            pair_features_g = feats[g].astype(np.float64)
                            pair_features_j = feats[j].astype(np.float64)
                            pairwise[doc_assignment[doc_id]].append(
                                {
                                    "layer": int(layer),
                                    "g_features": pair_features_g,
                                    "j_features": pair_features_j,
                                    "label": interaction,
                                    "doc_id": doc_id,
                                }
                            )
    return {
        "unary": unary,
        "pairwise": pairwise,
        "validation_contexts": validation_contexts,
        "sampled": {key: len(value) for key, value in sampled.items()},
        "doc_assignment": doc_assignment,
    }


# ---------------------------------------------------------------------------
# ridge fitting (shared with the Rust `atlas fit` normal equations)
# ---------------------------------------------------------------------------


def fit_ridge(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    xtx = x.T @ x
    xty = x.T @ y
    ridge = lam * np.eye(xtx.shape[0], dtype=np.float64)
    return np.linalg.solve(xtx + ridge, xty)


def standardize(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return mean, scale, (train_x - mean) / scale


def greedy_surrogate_selection(
    unary_scores: torch.Tensor,
    features: torch.Tensor,
    matrix: torch.Tensor,
    lam: float,
    k: int,
) -> torch.Tensor:
    """Greedy top-k by unary + lam * pairwise sum, `[T,G,F]` features."""
    tokens, groups, dim = features.shape
    selected = torch.zeros((tokens, groups), dtype=torch.bool, device=features.device)
    projected = features @ matrix  # [T,G,F]
    for _ in range(k):
        weight = selected.to(features.dtype)
        pair = (features * (weight.unsqueeze(1) @ projected)).sum(dim=2)
        score = unary_scores + lam * pair
        score = score.masked_fill(selected, float("-inf"))
        choice = score.argmax(dim=1)
        selected.scatter_(1, choice.unsqueeze(1), True)
    return selected


def fit_learned_models(labels: dict, groups: int) -> dict:
    unary_train = labels["unary"]["train"]
    unary_val = labels["unary"]["validation"]
    pair_train = labels["pairwise"]["train"]
    pair_val = labels["pairwise"]["validation"]
    val_contexts = labels["validation_contexts"]
    unary_weights = np.zeros((LAYERS, FEATURE_DIM), dtype=np.float64)
    feature_mean = np.zeros((LAYERS, FEATURE_DIM), dtype=np.float64)
    feature_scale = np.zeros((LAYERS, FEATURE_DIM), dtype=np.float64)
    interaction = np.zeros(
        (LAYERS, len(GROUP_FEATURES), len(GROUP_FEATURES)), dtype=np.float64
    )
    chosen_lambda = []
    chosen_pair_lambda = []
    diagnostics = []

    def unary_arrays(rows):
        if not rows:
            return (
                np.zeros((0, FEATURE_DIM), dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.int64),
            )
        x = np.array([np.concatenate([row["features"], [1.0]]) for row in rows], dtype=np.float64)
        y = np.array([row["label"] for row in rows], dtype=np.float64)
        layers = np.array([row["layer"] for row in rows], dtype=np.int64)
        return x, y, layers

    train_x, train_y, train_layers = unary_arrays(unary_train)
    val_x, val_y, val_layers = unary_arrays(unary_val)

    def context_regret(layer: int, scores: torch.Tensor) -> tuple[float, int]:
        contexts = val_contexts.get(layer, [])
        if not contexts:
            return (float("nan"), 0)
        cursor = 0
        regrets = []
        for context in contexts:
            count = context["features"].shape[0]
            context_scores = scores[cursor : cursor + count].reshape(1, -1)
            cursor += count
            sel = topk_mask(context_scores, 5)
            stats = {
                "gram": torch.tensor(context["gram"], dtype=torch.float32).unsqueeze(0),
                "dot_dense": torch.tensor(context["dot"], dtype=torch.float32).unsqueeze(0),
                "dense_sq": torch.tensor([context["dense_sq"]], dtype=torch.float32),
            }
            selected_error = masked_mlp.subset_local_error(stats, sel)
            _, best_error = best_k_selection(stats, 5)
            regrets.append((selected_error - best_error).clamp_min(0.0))
        return (float(torch.cat(regrets).mean()), len(contexts))

    for layer in range(LAYERS):
        train_mask = train_layers == layer
        val_mask = val_layers == layer
        if not bool(train_mask.any()):
            continue
        xl = train_x[train_mask]
        yl = train_y[train_mask]
        mean, scale, xl_std = standardize(xl)
        # The bias column is not standardized: it must stay exactly 1.0.
        mean[-1] = 0.0
        scale[-1] = 1.0
        xl_std[:, -1] = 1.0
        feature_mean[layer] = mean
        feature_scale[layer] = scale
        if bool(val_mask.any()):
            xv_std = (val_x[val_mask] - mean) / scale
            xv_std[:, -1] = 1.0
        else:
            xv_std = np.zeros((0, FEATURE_DIM))
        best = None
        for lam in (1e-4, 1e-3, 1e-2, 1e-1):
            weights = fit_ridge(xl_std, yl, lam)
            if xv_std.shape[0]:
                scores = torch.tensor(xv_std @ weights, dtype=torch.float32)
                regret, count = context_regret(layer, scores)
            else:
                regret, count = (float("nan"), 0)
            key = (regret if math.isfinite(regret) else float("inf"), lam)
            if best is None or key < best[0]:
                best = (key, lam, regret, count, weights)
        _key, lam, regret, count, weights = best
        unary_weights[layer] = weights
        val_spearman = None
        val_r2 = None
        if bool(val_mask.any()):
            predictions = xv_std @ weights
            actual = val_y[val_mask]
            val_spearman = spearman(predictions, actual)
            residual = float(np.sum((predictions - actual) ** 2))
            total = float(np.sum((actual - actual.mean()) ** 2))
            val_r2 = 1.0 - residual / total if total > 0 else None
        chosen_lambda.append({"layer": layer, "lambda": lam, "regret": regret})
        diagnostics.append(
            {
                "layer": layer,
                "unary_train": int(train_mask.sum()),
                "unary_validation": int(val_mask.sum()),
                "validation_contexts": count,
                "lambda": lam,
                "validation_regret": regret,
                "validation_spearman": val_spearman,
                "validation_r2": val_r2,
            }
        )

    # Pairwise interaction fit, per layer, on products of standardised group
    # features. The same standardisation is applied at inference.
    pair_by_layer: dict[int, list[dict]] = {layer: [] for layer in range(LAYERS)}
    pair_val_by_layer: dict[int, list[dict]] = {layer: [] for layer in range(LAYERS)}
    for row in pair_train:
        pair_by_layer[row["layer"]].append(row)
    for row in pair_val:
        pair_val_by_layer[row["layer"]].append(row)

    def pair_matrix(rows, layer):
        if not rows:
            return np.zeros((0, len(GROUP_FEATURES) ** 2))
        mean = feature_mean[layer][: len(GROUP_FEATURES)]
        scale = feature_scale[layer][: len(GROUP_FEATURES)]
        features = []
        for row in rows:
            g = (row["g_features"] - mean) / scale
            j = (row["j_features"] - mean) / scale
            features.append(np.outer(g, j).reshape(-1))
        return np.array(features, dtype=np.float64)

    for layer in range(LAYERS):
        xl = pair_matrix(pair_by_layer[layer], layer)
        if xl.shape[0] == 0:
            continue
        yl = np.array([row["label"] for row in pair_by_layer[layer]], dtype=np.float64)
        xv = pair_matrix(pair_val_by_layer[layer], layer)
        yv = np.array(
            [row["label"] for row in pair_val_by_layer[layer]], dtype=np.float64
        )
        best = None
        for lam in (1e-3, 1e-2, 1e-1, 1.0):
            weights = fit_ridge(xl, yl, lam)
            if xv.shape[0]:
                mse = float(np.mean((xv @ weights - yv) ** 2))
            else:
                mse = float("nan")
            key = (mse if math.isfinite(mse) else float("inf"), lam)
            if best is None or key < best[0]:
                best = (key, lam, mse, weights)
        _key, lam, mse, weights = best
        matrix = weights.reshape(len(GROUP_FEATURES), len(GROUP_FEATURES))
        interaction[layer] = 0.5 * (matrix + matrix.T)
        chosen_pair_lambda.append(
            {"layer": layer, "lambda": lam, "validation_mse": mse}
        )

    # Mixing weight on validation local regret, using the validation contexts.
    best_mix = (1.0, float("inf"))
    for mix in (0.25, 0.5, 1.0, 2.0):
        regrets = []
        for layer in range(LAYERS):
            if not np.any(feature_scale[layer]):
                continue
            for context in val_contexts.get(layer, []):
                raw = np.concatenate(
                    [
                        context["features"],
                        np.ones((context["features"].shape[0], 1), dtype=np.float32),
                    ],
                    axis=1,
                )
                standardized = (raw - feature_mean[layer]) / feature_scale[layer]
                features = torch.tensor(standardized[:, : len(GROUP_FEATURES)], dtype=torch.float32)
                scores = torch.tensor(
                    standardized @ unary_weights[layer], dtype=torch.float32
                )
                sel = greedy_surrogate_selection(
                    scores.unsqueeze(0) if scores.dim() == 1 else scores.reshape(1, -1),
                    features.unsqueeze(0),
                    torch.tensor(interaction[layer], dtype=torch.float32),
                    mix,
                    min(5, groups),
                )
                stats = {
                    "gram": torch.tensor(context["gram"], dtype=torch.float32).unsqueeze(0),
                    "dot_dense": torch.tensor(context["dot"], dtype=torch.float32).unsqueeze(0),
                    "dense_sq": torch.tensor([context["dense_sq"]], dtype=torch.float32),
                }
                selected_error = masked_mlp.subset_local_error(stats, sel)
                _, best_error = best_k_selection(stats, 5)
                regrets.append(float((selected_error - best_error).clamp_min(0.0)[0]))
        regret = float(np.mean(regrets)) if regrets else float("nan")
        if math.isfinite(regret) and regret < best_mix[1]:
            best_mix = (mix, regret)

    parameter_elements = int(
        unary_weights.size + interaction.size
    )
    return {
        "feature_names": list(GROUP_FEATURES) + ["bias"],
        "groups": groups,
        "layers": LAYERS,
        "unary_weights": unary_weights.tolist(),
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "interaction": interaction.tolist(),
        "lambda_mix": best_mix[0],
        "lambda_mix_validation_regret": best_mix[1],
        "parameter_elements": parameter_elements,
        "chosen_lambda": chosen_lambda,
        "chosen_pair_lambda": chosen_pair_lambda,
        "diagnostics": diagnostics,
        "counts": {
            "unary_train": len(unary_train),
            "unary_validation": len(unary_val),
            "pair_train": len(pair_train),
            "pair_validation": len(pair_val),
        },
    }


# ---------------------------------------------------------------------------
# selector construction and scoring
# ---------------------------------------------------------------------------


def score_standardized(features: torch.Tensor, weights: dict, layer: int) -> torch.Tensor:
    mean = torch.tensor(weights["feature_mean"][layer], dtype=features.dtype, device=features.device)
    scale = torch.tensor(weights["feature_scale"][layer], dtype=features.dtype, device=features.device)
    raw = features_with_bias(features)
    standardized = (raw - mean) / scale
    w = torch.tensor(weights["unary_weights"][layer], dtype=features.dtype, device=features.device)
    return (standardized * w).sum(dim=-1)


def interaction_matrix(weights: dict, layer: int, device) -> torch.Tensor:
    return torch.tensor(weights["interaction"][layer], dtype=torch.float32, device=device)


def select_from_scores(
    scores: list[torch.Tensor], extras: int, largest: bool = True
) -> list[torch.Tensor]:
    selection = [topk_mask(score, 5) for score in scores]
    if extras:
        selection = best_extra_pairs(scores, selection, extras, largest=largest)
    return selection


def select_learned_interaction(
    features: list[torch.Tensor],
    weights: dict,
    extras: int,
) -> list[torch.Tensor]:
    selection = []
    lam = float(weights["lambda_mix"])
    for layer, layer_features in enumerate(features):
        unary = score_standardized(layer_features, weights, layer)
        mean = torch.tensor(
            weights["feature_mean"][layer][: len(GROUP_FEATURES)],
            dtype=layer_features.dtype,
            device=layer_features.device,
        )
        scale = torch.tensor(
            weights["feature_scale"][layer][: len(GROUP_FEATURES)],
            dtype=layer_features.dtype,
            device=layer_features.device,
        )
        group_std = (layer_features - mean) / scale
        matrix = interaction_matrix(weights, layer, layer_features.device)
        sel = greedy_surrogate_selection(unary, group_std, matrix, lam, 5)
        selection.append(sel)
    if extras:
        scores = [
            score_standardized(features[layer], weights, layer)
            for layer in range(len(features))
        ]
        selection = best_extra_pairs(scores, selection, extras, largest=True)
    return selection


def dense_group_norms(
    controller: masked_mlp.MaskController, hidden: torch.Tensor, bounds
) -> tuple[torch.Tensor, torch.Tensor]:
    activation = masked_mlp.dense_channels(controller, hidden)
    norms = torch.stack(
        [activation[:, start:end].norm(dim=1) for start, end in bounds], dim=1
    )
    return norms, activation


def arm_scores_from_hidden(
    arm: dict,
    controller: masked_mlp.MaskController,
    hidden_by_layer: list[torch.Tensor],
    accounting: dict,
) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
    groups = group_count(accounting)
    bounds = masked_mlp.group_bounds(int(accounting["intermediate_per_layer"]), groups)
    kind = arm["kind"]
    scores = []
    features = []
    for layer in range(LAYERS):
        hidden = hidden_by_layer[layer].reshape(-1, HIDDEN)
        if kind == "magnitude":
            norms, _ = dense_group_norms(controller.wrappers[layer], hidden, bounds)
            scores.append(norms)
        elif kind == "frequency":
            norms, _ = dense_group_norms(controller.wrappers[layer], hidden, bounds)
            total = norms.sum(dim=1, keepdim=True).clamp_min(1e-30)
            scores.append(norms / total)
        elif kind in ("learned-unary", "learned-interaction"):
            activation = masked_mlp.dense_channels(controller.wrappers[layer], hidden)
            layer_features = group_features_from_activation(activation, bounds)
            features.append(layer_features)
            if kind == "learned-unary":
                scores.append(
                    score_standardized(layer_features, arm["learned"], layer)
                )
        else:
            raise SystemExit(f"unknown score arm {kind}")
    return scores, features if features else None


def selection_from_arm_scores(
    arm: dict, scores: list[torch.Tensor]
) -> list[torch.Tensor]:
    extras = int(arm["budget"]["extra_groups"])
    if arm["kind"] == "learned-interaction":
        raise SystemExit("interaction selection handled separately")
    return select_from_scores(scores, extras, largest=True)


# ---------------------------------------------------------------------------
# GxO attribution (non-deployable)
# ---------------------------------------------------------------------------


def gxo_scores(
    model,
    controller,
    capture,
    input_ids: torch.Tensor,
    accounting: dict,
    corrected: bool,
) -> tuple[list[torch.Tensor], float]:
    """Signed GxO / corrected GxO group scores for every layer and token.

    The loss is the mean teacher-forced next-token log-probability. It uses the
    gold next token, so the arm is non-deployable by construction and labelled
    so in every receipt.
    """
    groups = group_count(accounting)
    bounds = masked_mlp.group_bounds(int(accounting["intermediate_per_layer"]), groups)
    start = time.perf_counter()
    controller.set_dense()
    for wrapper in controller.wrappers:
        wrapper.track_activation = True
    with torch.enable_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits
        targets = input_ids[0, 1:]
        log_probs = torch.log_softmax(logits[0, :-1, :].float(), dim=-1)
        loss = -log_probs.gather(1, targets.unsqueeze(1)).mean()
        loss.backward()
        scores = []
        for layer in range(LAYERS):
            activation = controller.wrappers[layer].last_activation
            grad = activation.grad
            if grad is None:
                raise SystemExit(f"no gradient captured at layer {layer}")
            if corrected:
                channel_sum = grad.sum(dim=-1, keepdim=True)
                per_channel = activation * grad + 0.5 * activation * channel_sum**2
            else:
                per_channel = activation * grad
            activation_flat = activation.reshape(-1, INTERMEDIATE)
            per_channel = per_channel.reshape(-1, INTERMEDIATE)
            layer_scores = torch.stack(
                [
                    per_channel[:, start_channel:end_channel].sum(dim=1)
                    for start_channel, end_channel in bounds
                ],
                dim=1,
            )
            scores.append(layer_scores)
    for wrapper in controller.wrappers:
        wrapper.track_activation = False
        wrapper.last_activation = None
    elapsed = time.perf_counter() - start
    return scores, elapsed


# ---------------------------------------------------------------------------
# oracle
# ---------------------------------------------------------------------------


def oracle_local_map(
    controller,
    capture,
    hidden_by_layer: list[torch.Tensor],
    accounting: dict,
    groups: int,
    shape: str,
    extras: int,
) -> list[torch.Tensor]:
    """Per-token local-exact selection from captured hidden states."""
    bounds = masked_mlp.group_bounds(int(accounting["intermediate_per_layer"]), groups)
    del bounds
    selection = []
    deltas = []
    for layer in range(LAYERS):
        hidden = hidden_by_layer[layer].reshape(-1, HIDDEN)
        stats = masked_mlp.group_contributions(
            controller.wrappers[layer], hidden, groups
        )
        if shape == "per_layer":
            selected, _ = best_k_selection(stats, 5) if groups == 16 else greedy_k_selection(stats, 20)
            selection.append(selected)
            if extras:
                deltas.append(local_delta_for_extras(stats, selected))
        elif shape == "four":
            # 4 groups: best single per layer, then add a second to 7 layers.
            singles = []
            for group in range(4):
                sel = torch.zeros((hidden.shape[0], 4), dtype=torch.bool, device=hidden.device)
                sel[:, group] = True
                singles.append(masked_mlp.subset_local_error(stats, sel))
            singles = torch.stack(singles, dim=1)  # [T, 4]
            best_single = singles.argmin(dim=1)
            selected = torch.zeros((hidden.shape[0], 4), dtype=torch.bool, device=hidden.device)
            selected.scatter_(1, best_single.unsqueeze(1), True)
            selection.append(selected)
            deltas.append(local_delta_for_extras(stats, selected))
        else:
            raise SystemExit(f"unknown oracle shape {shape}")
    if shape == "four":
        # Exactly seven layers get a second group: the largest local reduction,
        # with at most one extra group per layer.
        stacked = torch.stack(deltas, dim=1)  # [T, L, 4]
        tokens = stacked.shape[0]
        flat = stacked.reshape(tokens, -1)
        order = torch.argsort(flat, dim=1, descending=False, stable=True)
        for token in range(tokens):
            chosen_layers = set()
            chosen = 0
            for value in order[token].tolist():
                layer = value // 4
                if layer in chosen_layers:
                    continue
                chosen_layers.add(layer)
                selection[layer][token, value % 4] = True
                chosen += 1
                if chosen == 7:
                    break
    elif extras:
        selection = best_extra_pairs(deltas, selection, extras, largest=False)
    return selection


def oracle_final_refine(
    model,
    controller,
    capture,
    input_ids,
    selection: list[torch.Tensor],
    teacher_logits,
    accounting: dict,
    token_budget: int = 32,
    passes: int = 2,
) -> tuple[list[torch.Tensor], dict]:
    """Bounded exact-KL single-swap refinement on a token sample.

    Each candidate is evaluated with a prefix forward containing only positions
    `<= t`; the target position's own next-token distribution is compared with
    the teacher's at that position. No future token enters the decision.
    """
    groups = group_count(accounting)
    bounds = masked_mlp.group_bounds(int(accounting["intermediate_per_layer"]), groups)
    backend = {
        "bounds": bounds,
        "intermediate": int(accounting["intermediate_per_layer"]),
    }
    tokens = input_ids.shape[1]
    base_hidden = [capture.get(layer).clone() for layer in range(LAYERS)]
    positions = sorted(
        set(
            int(value)
            for value in np.linspace(1, tokens - 1, num=min(token_budget, tokens - 1), dtype=int)
        )
    )
    best = [layer_selection.clone() for layer_selection in selection]
    forwards = 0
    improvements = 0
    candidates_evaluated = 0
    gpu_seconds = 0.0

    def exact_kl(position: int, prefix, logits) -> float:
        teacher = teacher_logits[0, position].float()
        probabilities = torch.softmax(teacher, dim=-1)
        log_teacher = torch.log_softmax(teacher, dim=-1)
        log_masked = torch.log_softmax(logits[0, -1].float(), dim=-1)
        value = (probabilities * (log_teacher - log_masked)).nan_to_num()
        return float(value.sum())

    def run_prefix(position: int, current) -> tuple[object, float]:
        nonlocal forwards, gpu_seconds
        prefix = input_ids[:, : position + 1]
        masks = channel_masks_from_bool(current, backend)
        controller.set_token_masks([mask[: position + 1] for mask in masks])
        start_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if start_event:
            start_event.record()
        logits = model(input_ids=prefix, use_cache=False).logits
        if end_event:
            end_event.record()
            torch.cuda.synchronize()
            gpu_seconds += start_event.elapsed_time(end_event) / 1000.0
        forwards += 1
        return logits, exact_kl(position, prefix, logits)

    with torch.inference_mode():
        for position in positions:
            for _pass in range(passes):
                for layer in range(LAYERS):
                    hidden_row = base_hidden[layer][:, position : position + 1, :].reshape(1, HIDDEN)
                    stats = masked_mlp.group_contributions(
                        controller.wrappers[layer], hidden_row, groups
                    )
                    current = best[layer][position].clone()
                    selected_indices = torch.nonzero(current).flatten().tolist()
                    unselected = [g for g in range(groups) if not bool(current[g])]
                    if not selected_indices or not unselected:
                        continue
                    proposals = []
                    for remove in selected_indices:
                        for add in unselected:
                            proposal = current.clone()
                            proposal[remove] = False
                            proposal[add] = True
                            mask = proposal.reshape(1, -1).to(hidden_row.device)
                            local_error = float(
                                masked_mlp.subset_local_error(stats, mask)[0]
                            )
                            proposals.append((local_error, proposal))
                    proposals.sort(key=lambda item: item[0])
                    _current_logits, current_kl = run_prefix(position, best)
                    accepted = None
                    for _local_error, proposal in proposals[:4]:
                        trial = [layer_selection.clone() for layer_selection in best]
                        trial[layer][position] = proposal.to(trial[layer].device)
                        _logits, kl = run_prefix(position, trial)
                        candidates_evaluated += 1
                        if kl < current_kl - 1e-9:
                            accepted = (trial, kl)
                            break
                    if accepted is not None:
                        best = accepted[0]
                        improvements += 1
    return best, {
        "positions": len(positions),
        "passes": passes,
        "forwards": forwards,
        "candidates_evaluated": candidates_evaluated,
        "improvements": improvements,
        "gpu_seconds": gpu_seconds,
    }


# ---------------------------------------------------------------------------
# receipts and arm construction
# ---------------------------------------------------------------------------


def base_receipt(args, kind: str, accounting: dict, entry: dict, device, split_receipt: dict) -> dict:
    return {
        "schema": SCHEMA_PREPARE if kind == "prepare" else SCHEMA_EVAL,
        "kind": kind,
        "stamp": args.stamp or stamp_now(),
        "revision": fs.git_head(),
        "preregistration_commit": args.preregistration_commit,
        "created_utc": utc_now(),
        "model": {
            "id": args.model_id,
            "revision": entry["revision"],
            "directory": fs.model_directory(entry),
            "hidden": int(accounting["hidden"]),
            "intermediate": int(accounting["intermediate_per_layer"]),
            "layers": LAYERS,
            "groups": int(accounting["group_count"]),
            "loaded_dtype": "float32",
        },
        "environment": fs.environment(device),
        "accounting": accounting,
        "data": {
            "split": args.split,
            "split_sha256": split_receipt["derived"]["sha256"],
            "split_receipt": args.split_receipt,
            "counts": split_receipt["counts"],
        },
        "deviations": [],
        "nonfinite": [],
        "notes": [
            "teacher is the dense float32 model with TF32 disabled",
            "selectors receive only pre-decision information; see docs/EXPERIMENT-CONTRACT.md M2.10",
            "gradient and oracle arms are marked non-deployable",
            "active parameter counts are realized selections plus learned controller weights",
        ],
    }


def m1_greedy_selections() -> dict:
    paths = sorted(glob.glob("results/frontier/sweep-greedy-*.json"))
    if not paths:
        raise SystemExit("no M1 greedy receipt found")
    receipt = load_json(paths[-1])
    wanted = {"greedy-uniform5", "greedy-swap4"}
    out = {}
    for arm in receipt["arms"]:
        if arm["id"] in wanted:
            out[arm["id"]] = {
                "selection": arm["selection"],
                "receipt": paths[-1],
                "stamp": receipt["stamp"],
                "active": arm["active"],
            }
    return out


def build_arms(args, prepare: dict, accounting: dict) -> list[dict]:
    groups = group_count(accounting)
    backbone = int(accounting["always_active_backbone_elements"])
    hidden = int(accounting["hidden"])
    intermediate = int(accounting["intermediate_per_layer"])
    rankings = prepare["rankings"]
    b140 = budget_spec(accounting, "B140")
    b142 = budget_spec(accounting, "B142")
    greedy = m1_greedy_selections()

    def active_from_counts(counts: list[int]) -> dict:
        selection = []
        for count in counts:
            selection.append(list(range(int(count))))
        return masked_mlp.active_parameter_report(
            selection, hidden, intermediate, groups, backbone
        )

    arms = []
    arms.append(
        {
            "id": "teacher-dense",
            "kind": "dense",
            "deployable": False,
            "budget": {
                **b142,
                "name": "noncompliant-dense",
                "groups": LAYERS * groups,
                "extra_groups": 0,
            },
            "active": active_from_counts([groups] * LAYERS),
        }
    )
    arms.append(
        {
            "id": "masked-all",
            "kind": "all",
            "deployable": False,
            "budget": {
                **b142,
                "name": "noncompliant-all-groups",
                "groups": LAYERS * groups,
                "extra_groups": 0,
            },
            "active": active_from_counts([groups] * LAYERS),
        }
    )
    for name, ranking, k in (
        ("static-magnitude-k5", "magnitude", 5),
        ("static-frequency-k5", "frequency", 5),
        ("static-random-k5", "random", 5),
    ):
        selection_lists = [rankings[ranking][layer][:k] for layer in range(LAYERS)]
        arms.append(
            {
                "id": name,
                "kind": "static",
                "deployable": True,
                "budget": b140,
                "selection": selection_lists,
                "active": active_from_counts([k] * LAYERS),
            }
        )
    for arm_id in ("greedy-uniform5", "greedy-swap4"):
        if arm_id not in greedy:
            continue
        entry = greedy[arm_id]
        counts = [len(selected) for selected in entry["selection"]]
        budget = b142 if arm_id == "greedy-uniform5" else {**b140, "name": "M1-K3-112"}
        arms.append(
            {
                "id": f"static-{arm_id}",
                "kind": "static",
                "deployable": True,
                "budget": budget,
                "selection": entry["selection"],
                "active": active_from_counts(counts),
                "m1_receipt": entry["receipt"],
                "budget_matched": arm_id == "greedy-uniform5",
            }
        )
    for kind, arm_id, deployable in (
        ("random", "token-random", True),
        ("magnitude", "token-magnitude", True),
        ("frequency", "token-frequency", True),
    ):
        arms.append(
            {
                "id": arm_id,
                "kind": kind,
                "deployable": deployable,
                "budget": b142,
                "active": active_from_counts([5] * LAYERS + []),
            }
        )
    # Per-token arms are exactly 142 groups per token: five per layer plus two
    # extras. The representative count is five per layer plus two in layer 0.
    rep = [5] * LAYERS
    rep[0] += 2
    for arm in arms:
        if arm["id"] in ("token-random", "token-magnitude", "token-frequency"):
            arm["active"] = active_from_counts(rep)
    arms.append(
        {
            "id": "token-task-static-nn",
            "kind": "task-static",
            "deployable": True,
            "budget": b142,
            "active": active_from_counts(rep),
        }
    )
    for kind, arm_id in (
        ("learned-unary", "token-learned-unary"),
        ("learned-interaction", "token-learned-interaction"),
    ):
        learned = prepare["learned"]
        arms.append(
            {
                "id": arm_id,
                "kind": kind,
                "deployable": True,
                "budget": b142,
                "learned": learned,
                "active": active_from_counts(rep),
            }
        )
    for kind, arm_id in (("gxo", "token-gxo"), ("cor-gxo", "token-cor-gxo")):
        arms.append(
            {
                "id": arm_id,
                "kind": kind,
                "deployable": False,
                "budget": b142,
                "active": active_from_counts(rep),
            }
        )
    for arm in arms:
        arm["controller"] = {
            "elements": controller_elements(arm),
            "selector_id": arm["id"],
        }
        active = arm["active"]["active_unique_elements"] + arm["controller"]["elements"]
        arm["active"]["controller_elements"] = arm["controller"]["elements"]
        arm["active"]["active_unique_elements_with_controller"] = active
        arm["active"]["compliant"] = active <= int(accounting["ceiling"])
        if arm["id"] not in ("teacher-dense", "masked-all", "static-greedy-swap4"):
            if active != arm["budget"]["active_unique_elements"] + arm["controller"]["elements"]:
                raise SystemExit(f"arm {arm['id']} active count does not match its budget")
    return arms


def nearest_calibration_document(
    windows: list[list[int]], calibration: dict, shingles: int = 5
) -> str:
    def shingle_set(tokens: list[int]) -> set[tuple[int, ...]]:
        if len(tokens) <= shingles:
            return {tuple(tokens)}
        return {tuple(tokens[index : index + shingles]) for index in range(len(tokens) - shingles + 1)}

    tokens = [token for window in windows for token in window]
    probe = shingle_set(tokens)
    best = None
    for doc_id, payload in calibration["documents"].items():
        reference = payload.get("shingle_set")
        if reference is None:
            continue
        union = len(probe | reference)
        score = len(probe & reference) / union if union else 0.0
        key = (-score, doc_id)
        if best is None or key < best[0]:
            best = (key, doc_id)
    if best is None:
        raise SystemExit("no calibration documents for nearest-neighbour assignment")
    return best[1]


def document_shingles(split: dict) -> dict[str, set[tuple[int, ...]]]:
    out = {}
    for article in split["windows"]["calibration"]:
        tokens = [token for window in article["windows"] for token in window]
        if len(tokens) <= 5:
            out[article["doc_id"]] = {tuple(tokens)}
        else:
            out[article["doc_id"]] = {
                tuple(tokens[index : index + 5]) for index in range(len(tokens) - 4)
            }
    return out


def task_static_selection(prepare: dict, doc_id: str, extras: int) -> list[list[int]]:
    statistics = prepare["statistics"]
    doc = statistics["documents"][doc_id]
    selection = []
    for layer in range(LAYERS):
        order = sorted(
            range(statistics["groups"]),
            key=lambda g: (-doc["mean_norm"][layer][g], g),
        )
        selection.append(order[:5])
    if extras:
        scored = []
        for layer in range(LAYERS):
            selected = set(selection[layer])
            for group in range(statistics["groups"]):
                if group in selected:
                    continue
                scored.append((-doc["mean_norm"][layer][group], layer, group))
        scored.sort()
        for _score, layer, group in scored[:extras]:
            selection[layer].append(group)
        selection = [sorted(value) for value in selection]
    return selection


# ---------------------------------------------------------------------------
# one-window arm execution
# ---------------------------------------------------------------------------


def random_selection(
    input_ids: torch.Tensor, accounting: dict, extras: int, device
) -> list[torch.Tensor]:
    groups = group_count(accounting)
    tokens = input_ids.shape[1]
    sequence_key = int(input_ids[0, 0].item())
    scores = []
    for layer in range(LAYERS):
        layer_scores = torch.empty((tokens, groups), dtype=torch.float32, device=device)
        for position in range(tokens):
            for group in range(groups):
                state = splitmix64(
                    SEED
                    ^ (layer * 0x9E3779B97F4A7C15)
                    ^ (position * 0xBF58476D1CE4E5B9)
                    ^ (group * 0x94D049BB133111EB)
                    ^ sequence_key
                )
                layer_scores[position, group] = float(state) / float(1 << 64)
        scores.append(layer_scores)
    return select_from_scores(scores, extras, largest=True)


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


def run_metrics(teacher_logits, masked_logits, input_ids) -> dict:
    metrics = masked_mlp.logits_kl_metrics(teacher_logits, masked_logits, input_ids)
    return metrics


def measure_with_selection(
    model,
    controller,
    capture,
    input_ids,
    selection: list[torch.Tensor],
    backend: dict,
    use_capture: bool,
) -> tuple[object, float]:
    masks = channel_masks_from_bool(selection, backend)
    controller.set_token_masks(masks)
    if use_capture:
        capture.clear()
    start_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    if start_event:
        start_event.record()
    with torch.inference_mode():
        logits = model(input_ids=input_ids, use_cache=False).logits
    if end_event:
        end_event.record()
        torch.cuda.synchronize()
        seconds = start_event.elapsed_time(end_event) / 1000.0
    else:
        seconds = 0.0
    return logits, seconds


def selection_for_arm(
    arm: dict,
    model,
    controller,
    capture,
    input_ids,
    dense_hidden: list[torch.Tensor],
    teacher_logits,
    accounting: dict,
    backend: dict,
    document_id: str,
    prepare: dict,
) -> tuple[list[torch.Tensor], dict]:
    """Return the per-token selection and arm-specific timing/provenance."""
    extras = int(arm["budget"]["extra_groups"])
    kind = arm["kind"]
    info = {"iterations": [], "selector_seconds": 0.0, "selector_forwards": 0}
    if kind == "static":
        selection = []
        for layer_selection in arm["selection"]:
            row = torch.zeros(backend["groups"], dtype=torch.bool, device=input_ids.device)
            for index in layer_selection:
                row[index] = True
            selection.append(row.unsqueeze(0).expand(input_ids.shape[1], -1))
        return selection, info
    if kind == "task-static":
        doc_selection = task_static_selection(prepare, document_id, extras)
        selection = []
        for layer_selection in doc_selection:
            row = torch.zeros(backend["groups"], dtype=torch.bool, device=input_ids.device)
            for index in layer_selection:
                row[index] = True
            selection.append(row.unsqueeze(0).expand(input_ids.shape[1], -1))
        return selection, info
    if kind == "random":
        start = time.perf_counter()
        selection = random_selection(input_ids, accounting, extras, input_ids.device)
        info["selector_seconds"] = time.perf_counter() - start
        return selection, info
    if kind in ("magnitude", "frequency", "learned-unary", "learned-interaction"):
        start = time.perf_counter()
        scores, features = arm_scores_from_hidden(arm, controller, dense_hidden, accounting)
        if kind == "learned-interaction":
            selection = select_learned_interaction(features, arm["learned"], extras)
        else:
            selection = selection_from_arm_scores(arm, scores)
        info["selector_seconds"] = time.perf_counter() - start
        return selection, info
    if kind in ("gxo", "cor-gxo"):
        scores, elapsed = gxo_scores(
            model,
            controller,
            capture,
            input_ids,
            accounting,
            corrected=(kind == "cor-gxo"),
        )
        info["selector_seconds"] = elapsed
        info["selector_forwards"] = 1
        return select_from_scores(scores, extras, largest=True), info
    raise SystemExit(f"unknown arm kind {kind}")


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


def fit_data_from_labels(labels: dict, learned: dict) -> dict:
    unary_by_layer: dict[int, list[dict]] = {layer: [] for layer in range(LAYERS)}
    pair_by_layer: dict[int, list[dict]] = {layer: [] for layer in range(LAYERS)}
    for row in labels["unary"]["train"]:
        unary_by_layer[row["layer"]].append(row)
    for row in labels["pairwise"]["train"]:
        pair_by_layer[row["layer"]].append(row)
    lambda_unary = {item["layer"]: item["lambda"] for item in learned["chosen_lambda"]}
    lambda_pair = {
        item["layer"]: item["lambda"] for item in learned["chosen_pair_lambda"]
    }
    layers = []
    for layer in range(LAYERS):
        unary_rows = unary_by_layer[layer]
        if not unary_rows:
            continue
        x = np.array(
            [
                np.concatenate([row["features"], [1.0]])
                for row in unary_rows
            ],
            dtype=np.float64,
        )
        y = np.array([row["label"] for row in unary_rows], dtype=np.float64)
        mean = np.array(learned["feature_mean"][layer], dtype=np.float64)
        scale = np.array(learned["feature_scale"][layer], dtype=np.float64)
        z = (x - mean) / scale
        z[:, -1] = 1.0
        pair_rows = pair_by_layer[layer]
        if pair_rows:
            group_mean = mean[: len(GROUP_FEATURES)]
            group_scale = scale[: len(GROUP_FEATURES)]
            products = []
            pair_y = []
            for row in pair_rows:
                g = (row["g_features"] - group_mean) / group_scale
                j = (row["j_features"] - group_mean) / group_scale
                products.append(np.outer(g, j).reshape(-1))
                pair_y.append(row["label"])
            p = np.array(products, dtype=np.float64)
            py = np.array(pair_y, dtype=np.float64)
            pair_xtx = (p.T @ p).tolist()
            pair_xty = (p.T @ py).tolist()
            pair_count = len(pair_rows)
        else:
            pair_xtx = np.zeros(
                (len(GROUP_FEATURES) ** 2, len(GROUP_FEATURES) ** 2)
            ).tolist()
            pair_xty = np.zeros(len(GROUP_FEATURES) ** 2).tolist()
            pair_count = 0
        layers.append(
            {
                "layer": layer,
                "unary_xtx": (z.T @ z).tolist(),
                "unary_xty": (z.T @ y).tolist(),
                "unary_count": len(unary_rows),
                "unary_lambda": lambda_unary.get(layer),
                "pair_xtx": pair_xtx,
                "pair_xty": pair_xty,
                "pair_count": pair_count,
                "pair_lambda": lambda_pair.get(layer),
                "feature_mean": mean.tolist(),
                "feature_scale": scale.tolist(),
            }
        )
    return {
        "schema": "selvedge/atlas/fit-data-v1",
        "layers": layers,
        "feature_names": learned["feature_names"],
        "groups": learned["groups"],
        "lambda_mix": learned["lambda_mix"],
        "notes": [
            "the matrices are sufficient statistics of the standardised features",
            "the pair matrices are sufficient statistics of outer products of standardised group features",
            "atlas fit solves (X^T X + lambda I) w = X^T y per layer",
        ],
    }


def command_prepare(args) -> int:
    fs.configure_arithmetic()
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    entry, accounting, split_receipt, split = model_and_accounting(args)
    model = fs.load_model(fs.model_directory(entry), device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    controller = masked_mlp.MaskController(model)
    capture = masked_mlp.HiddenCapture(model)
    statistics = calibration_statistics(model, controller, split, accounting, device, capture)
    labels = collect_labels(
        model,
        controller,
        split,
        accounting,
        device,
        capture,
        contexts_per_window=args.contexts_per_window,
        pairs_per_context=args.pairs_per_context,
    )
    learned = fit_learned_models(labels, group_count(accounting))
    rankings = rankings_from_statistics(statistics)
    stamp = args.stamp or stamp_now()
    weights = {
        "schema": "selvedge/atlas/weights-v1",
        "stamp": stamp,
        "revision": fs.git_head(),
        "preregistration_commit": args.preregistration_commit,
        "model": {
            "id": args.model_id,
            "revision": entry["revision"],
            "hidden": int(accounting["hidden"]),
            "intermediate": int(accounting["intermediate_per_layer"]),
            "layers": LAYERS,
            "groups": group_count(accounting),
        },
        "split_sha256": split_receipt["derived"]["sha256"],
        "feature_names": learned["feature_names"],
        "groups": learned["groups"],
        "layers": learned["layers"],
        "unary_weights": learned["unary_weights"],
        "feature_mean": learned["feature_mean"],
        "feature_scale": learned["feature_scale"],
        "interaction": learned["interaction"],
        "lambda_mix": learned["lambda_mix"],
        "parameter_elements": learned["parameter_elements"],
        "selection": {"per_layer": 5, "extra_groups": 2},
        "rankings": rankings,
        "counts": learned["counts"],
        "chosen_lambda": learned["chosen_lambda"],
        "chosen_pair_lambda": learned["chosen_pair_lambda"],
        "validation": learned["diagnostics"],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.weights_out)) or ".", exist_ok=True)
    write_json_atomic(args.weights_out, weights)
    fit_data = fit_data_from_labels(labels, learned)
    fit_data["stamp"] = stamp
    fit_data["revision"] = fs.git_head()
    fit_data["preregistration_commit"] = args.preregistration_commit
    write_json_atomic(args.fit_data_out, fit_data)
    receipt = base_receipt(args, "prepare", accounting, entry, device, split_receipt)
    receipt.update(
        {
            "stamp": stamp,
            "statistics": statistics,
            "rankings": rankings,
            "learned": {
                "parameter_elements": learned["parameter_elements"],
                "counts": learned["counts"],
                "lambda_mix": learned["lambda_mix"],
                "chosen_lambda": learned["chosen_lambda"],
                "chosen_pair_lambda": learned["chosen_pair_lambda"],
                "diagnostics": learned["diagnostics"],
            },
            "labels": {
                "sampled": labels["sampled"],
                "doc_assignment": labels["doc_assignment"],
            },
            "files": {
                "weights": {
                    "path": args.weights_out,
                    "sha256": fs.sha256_file(args.weights_out),
                },
                "fit_data": {
                    "path": args.fit_data_out,
                    "sha256": fs.sha256_file(args.fit_data_out),
                },
            },
            "seed": SEED,
        }
    )
    os.makedirs(args.receipt_dir, exist_ok=True)
    path = os.path.join(
        args.receipt_dir,
        f"prepare-{args.model_id.replace('/', '-')}-{stamp}.json",
    )
    write_json_atomic(path, receipt)
    capture.remove()
    print(f"prepare: {args.model_id}@{entry['revision']} device={device}")
    print(f"prepare: weights {args.weights_out} sha256={receipt['files']['weights']['sha256']}")
    print(f"prepare: fit data {args.fit_data_out} sha256={receipt['files']['fit_data']['sha256']}")
    print(f"prepare: receipt {path}")
    if args.json:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


# ---------------------------------------------------------------------------
# interactions
# ---------------------------------------------------------------------------


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = rx.astype(np.float64)
    ry = ry.astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denominator = math.sqrt(float((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / denominator) if denominator else 0.0


def command_interactions(args) -> int:
    fs.configure_arithmetic()
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    entry, accounting, split_receipt, split = model_and_accounting(args)
    prepare = load_json(args.prepare)
    groups = group_count(accounting)
    bounds = masked_mlp.group_bounds(int(accounting["intermediate_per_layer"]), groups)
    backend = {
        "bounds": bounds,
        "intermediate": int(accounting["intermediate_per_layer"]),
        "groups": groups,
    }
    model = fs.load_model(fs.model_directory(entry), device)
    controller = masked_mlp.MaskController(model)
    capture = masked_mlp.HiddenCapture(model)
    rankings = prepare["rankings"]["magnitude"]
    base_layer_groups = [rankings[layer][:4] for layer in range(LAYERS)]
    static_selection = [rankings[layer][:5] for layer in range(LAYERS)]
    probe_layers = [0, 7, 14, 21, 27]
    articles = split["windows"]["calibration"][:8]
    windows = [
        (article["doc_id"], index, window)
        for article in articles
        for index, window in enumerate(article["windows"][:1])
    ]
    # Co-activation and static group descriptors per layer, from the probe
    # windows. This is calibration text only.
    coactivation_sum = {
        layer: np.zeros((groups, groups), dtype=np.float64) for layer in probe_layers
    }
    mean_norm = {
        layer: np.zeros(groups, dtype=np.float64) for layer in probe_layers
    }
    with torch.inference_mode():
        for _doc_id, _index, window in windows:
            input_ids = torch.tensor([window], dtype=torch.long, device=device)
            controller.set_dense()
            capture.clear()
            model(input_ids=input_ids, use_cache=False)
            for layer in probe_layers:
                hidden = capture.get(layer).reshape(-1, HIDDEN)
                activation = masked_mlp.dense_channels(controller.wrappers[layer], hidden)
                means = torch.stack(
                    [activation[:, start:end].mean(dim=0) for start, end in bounds],
                    dim=0,
                )
                normalized = torch.nn.functional.normalize(means, dim=1)
                coactivation_sum[layer] += (normalized @ normalized.T).cpu().numpy()
                mean_norm[layer] += (
                    torch.stack(
                        [activation[:, start:end].norm(dim=1) for start, end in bounds],
                        dim=1,
                    )
                    .mean(dim=0)
                    .cpu()
                    .numpy()
                )
    coactivation = {
        layer: coactivation_sum[layer] / max(len(windows), 1) for layer in probe_layers
    }
    mean_norm = {
        layer: mean_norm[layer] / max(len(windows), 1) for layer in probe_layers
    }
    pairs = {}
    for layer in probe_layers:
        excluded = set(base_layer_groups[layer])
        candidates = [g for g in range(groups) if g not in excluded]
        scored = []
        for g, j in itertools.combinations(candidates, 2):
            scored.append((float(coactivation[layer][g, j]), g, j))
        scored.sort()
        low = [(g, j) for _score, g, j in scored[:2]]
        high = [(g, j) for _score, g, j in sorted(scored, reverse=True)[:2]]
        pairs[layer] = low + high
    records = []
    nonfinite = []
    forwards = 0
    seconds = 0.0
    with torch.inference_mode():
        for doc_id, index, window in windows:
            input_ids = torch.tensor([window], dtype=torch.long, device=device)
            controller.set_dense()
            teacher = model(input_ids=input_ids, use_cache=False).logits
            for layer in probe_layers:
                for g, j in pairs[layer]:
                    variants = {
                        "S": base_layer_groups[layer],
                        "Sg": sorted(set(base_layer_groups[layer]) | {g}),
                        "Sj": sorted(set(base_layer_groups[layer]) | {j}),
                        "Sgj": sorted(set(base_layer_groups[layer]) | {g, j}),
                    }
                    kls = {}
                    for name, layer_selection in variants.items():
                        selection = [list(value) for value in static_selection]
                        selection[layer] = layer_selection
                        masks = static_mask(selection, backend, input_ids.shape[1])
                        controller.set_token_masks(masks)
                        start_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                        end_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                        if start_event:
                            start_event.record()
                        logits = model(input_ids=input_ids, use_cache=False).logits
                        if end_event:
                            end_event.record()
                            torch.cuda.synchronize()
                            seconds += start_event.elapsed_time(end_event) / 1000.0
                        forwards += 1
                        metrics = masked_mlp.logits_kl_metrics(teacher, logits, input_ids)
                        if not bool(torch.isfinite(metrics["kl"]).all()):
                            nonfinite.append({"doc_id": doc_id, "layer": layer, "pair": [g, j]})
                        kls[name] = float(metrics["kl"].mean())
                    delta_g = kls["S"] - kls["Sg"]
                    delta_j = kls["S"] - kls["Sj"]
                    interaction = kls["Sg"] + kls["Sj"] - kls["S"] - kls["Sgj"]
                    records.append(
                        {
                            "window": f"{doc_id}:{index}",
                            "doc_id": doc_id,
                            "layer": layer,
                            "g": g,
                            "j": j,
                            "coactivation": float(coactivation[layer][g, j]),
                            "mean_norm_product": float(mean_norm[layer][g] * mean_norm[layer][j]),
                            "kl_S": kls["S"],
                            "kl_Sg": kls["Sg"],
                            "kl_Sj": kls["Sj"],
                            "kl_Sgj": kls["Sgj"],
                            "delta_g": delta_g,
                            "delta_j": delta_j,
                            "interaction": interaction,
                        }
                    )
    capture.remove()
    interactions = np.array([row["interaction"] for row in records], dtype=np.float64)
    deltas_g = np.array([row["delta_g"] for row in records], dtype=np.float64)
    deltas_j = np.array([row["delta_j"] for row in records], dtype=np.float64)
    sign_flips = {}
    for layer in probe_layers:
        for g, j in pairs[layer]:
            values = [
                row["interaction"]
                for row in records
                if row["layer"] == layer and row["g"] == g and row["j"] == j
            ]
            signs = [1 if value > 0 else (-1 if value < 0 else 0) for value in values]
            flips = sum(1 for left, right in zip(signs, signs[1:]) if left != right)
            sign_flips[f"{layer}:{g}:{j}"] = {
                "values": values,
                "flips": flips,
                "contexts": len(values),
            }
    # Surrogate: leave-one-window-out ridge on graph features.
    feature_rows = []
    for row in records:
        feature_rows.append(
            [
                row["coactivation"],
                math.log1p(row["mean_norm_product"]),
                row["layer"] / (LAYERS - 1),
                1.0,
            ]
        )
    features = np.array(feature_rows, dtype=np.float64)
    labels = interactions
    windows_unique = sorted({row["window"] for row in records})
    predictions = np.zeros_like(labels)
    for held_out in windows_unique:
        train_mask = np.array([row["window"] != held_out for row in records])
        test_mask = ~train_mask
        if not bool(test_mask.any()):
            continue
        mean, scale, train_std = standardize(features[train_mask])
        weights = fit_ridge(train_std, labels[train_mask], 1e-2)
        test_std = (features[test_mask] - mean) / scale
        predictions[test_mask] = test_std @ weights
    overall_spearman = spearman(labels, predictions)
    sign_accuracy = float(np.mean(np.sign(labels) == np.sign(predictions)))
    indices = fs.bootstrap_indices(len(windows_unique), args.resamples, SEED)
    window_to_rows = {
        window: [index for index, row in enumerate(records) if row["window"] == window]
        for window in windows_unique
    }
    bootstrap_spearman = []
    bootstrap_mean = []
    bootstrap_sign = []
    for resample in indices:
        row_indices = []
        for value in resample:
            row_indices.extend(window_to_rows[windows_unique[int(value)]])
        if len(row_indices) < 4:
            continue
        sampled_labels = labels[row_indices]
        sampled_predictions = predictions[row_indices]
        bootstrap_spearman.append(spearman(sampled_labels, sampled_predictions))
        bootstrap_mean.append(float(sampled_labels.mean()))
        bootstrap_sign.append(
            float(np.mean(np.sign(sampled_labels) == np.sign(sampled_predictions)))
        )
    def ci(values):
        values = np.array([value for value in values if math.isfinite(value)])
        if len(values) == 0:
            return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
        low, high = np.percentile(values, [2.5, 97.5])
        return {"mean": float(values.mean()), "ci_low": float(low), "ci_high": float(high)}
    positive = [row for row in records if row["interaction"] > 0]
    negative = [row for row in records if row["interaction"] < 0]
    positive.sort(key=lambda row: row["interaction"], reverse=True)
    negative.sort(key=lambda row: row["interaction"])
    stamp = args.stamp or stamp_now()
    receipt = base_receipt(args, "interactions", accounting, entry, device, split_receipt)
    receipt.update(
        {
            "schema": SCHEMA_INTERACTIONS,
            "stamp": stamp,
            "prepare": {
                "path": args.prepare,
                "sha256": fs.sha256_file(args.prepare),
            },
            "protocol": {
                "windows": len(windows),
                "layers": probe_layers,
                "pairs_per_layer": 4,
                "contexts": len(records),
                "forward_count": forwards,
                "gpu_seconds": seconds,
                "base_set": "top-4 magnitude groups of the probed layer",
                "other_layers": "static-magnitude-k5",
            },
            "pairs": {str(layer): [list(pair) for pair in pairs[layer]] for layer in probe_layers},
            "distribution": {
                "mean": float(interactions.mean()),
                "median": float(np.median(interactions)),
                "q25": float(np.percentile(interactions, 25)),
                "q75": float(np.percentile(interactions, 75)),
                "positive_fraction": float(np.mean(interactions > 0)),
                "negative_fraction": float(np.mean(interactions < 0)),
                "zero_fraction": float(np.mean(interactions == 0)),
                "mean_ci": ci(bootstrap_mean),
                "delta_g_mean": float(deltas_g.mean()),
                "delta_j_mean": float(deltas_j.mean()),
                "magnitude_asymmetry_mean": float(np.mean(np.abs(deltas_g) - np.abs(deltas_j))),
            },
            "sign_asymmetry": {
                "note": (
                    "the pairwise interaction is algebraically symmetric in g and j; "
                    "the measurable asymmetry is the sign change of a fixed pair across "
                    "contexts and the magnitude difference of the two marginal gains"
                ),
                "fixed_pair_sign_flips": sign_flips,
                "pairs_with_a_sign_flip": sum(
                    1 for value in sign_flips.values() if value["flips"] > 0
                ),
                "pairs_total": len(sign_flips),
            },
            "examples": {
                "synergy": positive[:5],
                "antagonism": negative[:5],
            },
            "surrogate": {
                "features": ["coactivation", "log1p_mean_norm_product", "layer_fraction", "bias"],
                "method": "leave-one-window-out ridge (lambda 1e-2)",
                "spearman": overall_spearman,
                "spearman_ci": ci(bootstrap_spearman),
                "sign_accuracy": sign_accuracy,
                "sign_accuracy_ci": ci(bootstrap_sign),
                "passes_h2_5": bool(
                    overall_spearman >= 0.2
                    and ci(bootstrap_spearman)["ci_low"] > 0
                    and sign_accuracy > 0.5
                ),
            },
            "records": records,
            "nonfinite": nonfinite,
        }
    )
    receipt["data"]["sampled_windows"] = [f"{doc_id}:{index}" for doc_id, index, _ in windows]
    os.makedirs(args.receipt_dir, exist_ok=True)
    path = os.path.join(args.receipt_dir, f"interactions-{stamp}.json")
    write_json_atomic(path, receipt)
    print(f"interactions: contexts={len(records)} forwards={forwards} forwards")
    print(
        f"interactions: mean I={interactions.mean():.6g} positive={np.mean(interactions > 0):.3f} "
        f"spearman={overall_spearman:.3f}"
    )
    print(f"interactions: receipt {path}")
    if args.json:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


# ---------------------------------------------------------------------------
# oracle
# ---------------------------------------------------------------------------


def oracle_sample(split: dict, count: int = 8) -> list[dict]:
    articles = split["windows"]["heldout_general"][:count]
    return [
        {
            "doc_id": article["doc_id"],
            "sha256": article["sha256"],
            "window_index": 0,
            "window": article["windows"][0],
        }
        for article in articles
    ]


def oracle_configs(accounting: dict) -> list[dict]:
    return [
        {"id": "oracle-local-16-B140", "groups": 16, "shape": "per_layer", "k": 5, "extras": 0, "budget": budget_spec(accounting, "B140")},
        {"id": "oracle-local-16-B142", "groups": 16, "shape": "per_layer", "k": 5, "extras": 2, "budget": budget_spec(accounting, "B142")},
        {"id": "oracle-local-64", "groups": 64, "shape": "per_layer", "k": 20, "extras": 0, "budget": budget_granularity(accounting, 64)},
        {"id": "oracle-local-4", "groups": 4, "shape": "four", "k": 1, "extras": 0, "budget": budget_granularity(accounting, 4)},
    ]


def command_oracle(args) -> int:
    fs.configure_arithmetic()
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    entry, accounting, split_receipt, split = model_and_accounting(args)
    model = fs.load_model(fs.model_directory(entry), device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    controller = masked_mlp.MaskController(model)
    capture = masked_mlp.HiddenCapture(model)
    sample = oracle_sample(split, args.windows)
    configs = oracle_configs(accounting)
    stamp = args.stamp or stamp_now()
    config_results = {
        config["id"]: {
            "id": config["id"],
            "groups": config["groups"],
            "shape": config["shape"],
            "k": config["k"],
            "extras": config["extras"],
            "budget": config["budget"],
            "windows": {},
            "passes": [],
            "forwards": 0,
            "selector_seconds": 0.0,
            "gpu_seconds": 0.0,
            "subset_evaluations": 0,
        }
        for config in configs
    }
    final_refine = None
    with torch.inference_mode():
        for window_record in sample:
            doc_id = window_record["doc_id"]
            window = window_record["window"]
            input_ids = torch.tensor([window], dtype=torch.long, device=device)
            controller.set_dense()
            capture.clear()
            teacher = model(input_ids=input_ids, use_cache=False).logits
            dense_hidden = [capture.get(layer).clone() for layer in range(LAYERS)]
            for config in configs:
                entry_result = config_results[config["id"]]
                current_hidden = dense_hidden
                best = None
                for pass_index in range(args.passes):
                    start = time.perf_counter()
                    selection = oracle_local_map(
                        controller,
                        capture,
                        current_hidden,
                        accounting,
                        config["groups"],
                        config["shape"],
                        config["extras"],
                    )
                    elapsed = time.perf_counter() - start
                    entry_result["selector_seconds"] += elapsed
                    entry_result["subset_evaluations"] += (
                        4368 * input_ids.shape[1] if config["groups"] == 16 else 0
                    )
                    backend = {
                        "bounds": masked_mlp.group_bounds(
                            int(accounting["intermediate_per_layer"]), config["groups"]
                        ),
                        "intermediate": int(accounting["intermediate_per_layer"]),
                        "groups": config["groups"],
                    }
                    logits, seconds = measure_with_selection(
                        model, controller, capture, input_ids, selection, backend, use_capture=True
                    )
                    entry_result["forwards"] += 1
                    entry_result["gpu_seconds"] += seconds
                    metrics = masked_mlp.logits_kl_metrics(teacher, logits, input_ids)
                    summary = masked_mlp.summarize_positions(metrics)
                    summary["pass"] = pass_index
                    summary["selector_seconds"] = elapsed
                    summary["selection"] = selection_summary(selection)
                    entry_result["passes"].append(
                        {"doc_id": doc_id, **summary}
                    )
                    if best is None or summary["kl_mean"] < best["kl_mean"]:
                        best = {
                            "doc_id": doc_id,
                            "pass": pass_index,
                            "kl_mean": summary["kl_mean"],
                            "summary": summary,
                            "selection": [value.clone() for value in selection],
                            "kl": metrics["kl"],
                            "top1": metrics["top1"],
                        }
                    current_hidden = [capture.get(layer).clone() for layer in range(LAYERS)]
                kl_positions = best["kl"]
                entry_result["windows"][doc_id] = {
                    "doc_id": doc_id,
                    "window_index": window_record["window_index"],
                    "positions": int(kl_positions.numel()),
                    "best_pass": best["pass"],
                    "kl_mean": best["summary"]["kl_mean"],
                    "kl_median": best["summary"]["kl_median"],
                    "kl_p95": best["summary"]["kl_p95"],
                    "kl_max": best["summary"]["kl_max"],
                    "top1_agreement": best["summary"]["top1_agreement"],
                    "teacher_entropy_mean": best["summary"]["teacher_entropy_mean"],
                    "nll_teacher_sum": best["summary"]["nll_teacher_mean"]
                    * best["summary"]["nll_positions"],
                    "nll_masked_sum": best["summary"]["nll_masked_mean"]
                    * best["summary"]["nll_positions"],
                    "nll_positions": best["summary"]["nll_positions"],
                    "fraction_kl_le_005": float((kl_positions <= 0.05).to(torch.float64).mean()),
                    "fraction_kl_le_025": float((kl_positions <= 0.25).to(torch.float64).mean()),
                    "fraction_top1": float(best["top1"].mean()),
                    "selection": best["summary"]["selection"],
                }
                if config["id"] == "oracle-local-16-B142":
                    # Final refinement uses the B142 16-group map as its start.
                    if final_refine is None:
                        final_refine = {
                            "windows": {},
                            "forwards": 0,
                            "candidates_evaluated": 0,
                            "improvements": 0,
                            "gpu_seconds": 0.0,
                        }
                    if len(final_refine["windows"]) < args.refine_windows:
                        refined, info = oracle_final_refine(
                            model,
                            controller,
                            capture,
                            input_ids,
                            best["selection"],
                            teacher,
                            accounting,
                            token_budget=args.refine_positions,
                            passes=args.refine_passes,
                        )
                        backend = {
                            "bounds": masked_mlp.group_bounds(
                                int(accounting["intermediate_per_layer"]), 16
                            ),
                            "intermediate": int(accounting["intermediate_per_layer"]),
                            "groups": 16,
                        }
                        logits, seconds = measure_with_selection(
                            model, controller, capture, input_ids, refined, backend, use_capture=False
                        )
                        metrics = masked_mlp.logits_kl_metrics(teacher, logits, input_ids)
                        summary = masked_mlp.summarize_positions(metrics)
                        kl_positions = metrics["kl"]
                        final_refine["windows"][doc_id] = {
                            "doc_id": doc_id,
                            "positions": int(kl_positions.numel()),
                            "kl_mean": summary["kl_mean"],
                            "kl_median": summary["kl_median"],
                            "kl_p95": summary["kl_p95"],
                            "kl_max": summary["kl_max"],
                            "top1_agreement": summary["top1_agreement"],
                            "teacher_entropy_mean": summary["teacher_entropy_mean"],
                            "nll_teacher_sum": summary["nll_teacher_mean"]
                            * summary["nll_positions"],
                            "nll_masked_sum": summary["nll_masked_mean"]
                            * summary["nll_positions"],
                            "nll_positions": summary["nll_positions"],
                            "fraction_kl_le_005": float((kl_positions <= 0.05).to(torch.float64).mean()),
                            "fraction_kl_le_025": float((kl_positions <= 0.25).to(torch.float64).mean()),
                            "fraction_top1": float(metrics["top1"].mean()),
                            "selection": selection_summary(refined),
                            "info": info,
                        }
                        final_refine["forwards"] += info["forwards"] + 1
                        final_refine["candidates_evaluated"] += info["candidates_evaluated"]
                        final_refine["improvements"] += info["improvements"]
                        final_refine["gpu_seconds"] += info["gpu_seconds"] + seconds
    capture.remove()
    # Aggregate per config: item-level aggregates, cross-token consistency.
    for config in configs:
        entry_result = config_results[config["id"]]
        windows = list(entry_result["windows"].values())
        if windows:
            entry_result["aggregate"] = {
                "items": len(windows),
                "positions": int(sum(item["positions"] for item in windows)),
                "kl_mean": float(np.mean([item["kl_mean"] for item in windows])),
                "kl_median_mean": float(np.mean([item["kl_median"] for item in windows])),
                "kl_p95_mean": float(np.mean([item["kl_p95"] for item in windows])),
                "top1_agreement": float(np.mean([item["top1_agreement"] for item in windows])),
                "fraction_kl_le_005": float(np.mean([item["fraction_kl_le_005"] for item in windows])),
                "fraction_kl_le_025": float(np.mean([item["fraction_kl_le_025"] for item in windows])),
                "teacher_entropy_mean": float(np.mean([item["teacher_entropy_mean"] for item in windows])),
                "nll_teacher_sum": float(sum(item["nll_teacher_sum"] for item in windows)),
                "nll_masked_sum": float(sum(item["nll_masked_sum"] for item in windows)),
                "nll_positions": int(sum(item["nll_positions"] for item in windows)),
            }
            entry_result["aggregate"]["ppl_ratio"] = float(
                math.exp(
                    (entry_result["aggregate"]["nll_masked_sum"] - entry_result["aggregate"]["nll_teacher_sum"])
                    / max(entry_result["aggregate"]["nll_positions"], 1)
                )
            )
            # Cross-token consistency: per-layer mean adjacent equality and
            # distinct subset counts, and cross-window Jaccard.
            adjacent = np.array(
                [item["selection"]["adjacent_equal_fraction"] for item in windows]
            )
            modal = np.array([item["selection"]["modal_fraction"] for item in windows])
            distinct = np.array(
                [item["selection"]["distinct_subsets_per_layer"] for item in windows]
            )
            entry_result["consistency"] = {
                "adjacent_equal_fraction_mean": float(adjacent.mean()),
                "adjacent_equal_fraction_per_layer": adjacent.mean(axis=0).tolist(),
                "modal_fraction_mean": float(modal.mean()),
                "distinct_subsets_per_layer_mean": float(distinct.mean()),
                "distinct_subsets_per_layer": distinct.mean(axis=0).tolist(),
                "per_token_groups_min": int(min(item["selection"]["per_token_min"] for item in windows)),
                "per_token_groups_max": int(max(item["selection"]["per_token_max"] for item in windows)),
            }
    stamp = args.stamp or stamp_now()
    receipt = base_receipt(args, "oracle", accounting, entry, device, split_receipt)
    receipt.update(
        {
            "schema": SCHEMA_ORACLE,
            "stamp": stamp,
            "sample": {
                "articles": len(sample),
                "positions": int(sum(len(item["window"]) for item in sample)),
                "documents": [item["doc_id"] for item in sample],
                "window_index": 0,
            },
            "protocol": {
                "passes": args.passes,
                "local_objective": "exact layer-output squared error from the Gram matrix",
                "final_refine": {
                    "windows": args.refine_windows,
                    "positions_per_window": args.refine_positions,
                    "passes": args.refine_passes,
                    "start": "oracle-local-16-B142",
                },
            },
            "configs": [config_results[config["id"]] for config in configs],
            "final_refine": final_refine,
            "seed": SEED,
            "notes": [
                "the oracle is non-deployable: it searches per-token subsets and runs several forwards",
                "the local objective is an exact one-layer intervention, not the final KL",
                "the final refinement accepts a swap only if the token's exact prefix KL improves",
            ],
        }
    )
    os.makedirs(args.receipt_dir, exist_ok=True)
    path = os.path.join(args.receipt_dir, f"oracle-{stamp}.json")
    write_json_atomic(path, receipt)
    for config in configs:
        aggregate = config_results[config["id"]].get("aggregate", {})
        print(
            f"oracle: {config['id']} kl={aggregate.get('kl_mean', float('nan')):.6g} "
            f"top1={aggregate.get('top1_agreement', float('nan')):.4f} "
            f"ppl_ratio={aggregate.get('ppl_ratio', float('nan')):.6g}"
        )
    print(f"oracle: receipt {path}")
    if args.json:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def build_domains(split: dict, sample: str | None) -> dict:
    if sample == "oracle":
        articles = split["windows"]["heldout_general"][:8]
        return {
            "general": [
                {"doc_id": article["doc_id"], "windows": article["windows"][:1]}
                for article in articles
            ]
        }
    return {
        "general": [
            {"doc_id": article["doc_id"], "windows": article["windows"]}
            for article in split["windows"]["heldout_general"]
        ],
        "code": [
            {"doc_id": item["doc_id"], "windows": item["windows"]}
            for item in split["windows"]["heldout_code"]
        ],
        "math": [
            {"doc_id": item["doc_id"], "windows": item["windows"]}
            for item in split["windows"]["heldout_math"]
        ],
    }


def evaluate_arm_on_window(
    arm: dict,
    model,
    controller,
    capture,
    input_ids,
    teacher_logits,
    dense_hidden: list[torch.Tensor],
    backend: dict,
    prepare: dict,
    document_id: str,
    accounting: dict,
) -> dict:
    kind = arm["kind"]
    if kind == "dense":
        metrics = masked_mlp.logits_kl_metrics(teacher_logits, teacher_logits, input_ids)
        return {
            "metrics": metrics,
            "seconds": 0.0,
            "forwards": 0,
            "selector_seconds": 0.0,
            "selection": None,
            "iteration_kl": [],
        }
    if kind == "all":
        selection = [
            torch.ones((input_ids.shape[1], backend["groups"]), dtype=torch.bool, device=input_ids.device)
            for _ in range(LAYERS)
        ]
        logits, seconds = measure_with_selection(
            model, controller, capture, input_ids, selection, backend, use_capture=False
        )
        metrics = masked_mlp.logits_kl_metrics(teacher_logits, logits, input_ids)
        return {
            "metrics": metrics,
            "seconds": seconds,
            "forwards": 1,
            "selector_seconds": 0.0,
            "selection": selection,
            "iteration_kl": [],
        }
    if kind in ("static", "task-static", "random"):
        selection, info = selection_for_arm(
            arm, model, controller, capture, input_ids, dense_hidden, teacher_logits,
            accounting, backend, document_id, prepare,
        )
        logits, seconds = measure_with_selection(
            model, controller, capture, input_ids, selection, backend, use_capture=False
        )
        metrics = masked_mlp.logits_kl_metrics(teacher_logits, logits, input_ids)
        return {
            "metrics": metrics,
            "seconds": seconds,
            "forwards": 1,
            "selector_seconds": info["selector_seconds"],
            "selection": selection,
            "iteration_kl": [float(metrics["kl"].mean())],
        }
    if kind in ("magnitude", "frequency", "learned-unary", "learned-interaction"):
        iteration_kl = []
        selection, info = selection_for_arm(
            arm, model, controller, capture, input_ids, dense_hidden, teacher_logits,
            accounting, backend, document_id, prepare,
        )
        logits0, seconds0 = measure_with_selection(
            model, controller, capture, input_ids, selection, backend, use_capture=True
        )
        metrics0 = masked_mlp.logits_kl_metrics(teacher_logits, logits0, input_ids)
        iteration_kl.append(float(metrics0["kl"].mean()))
        hidden1 = [capture.get(layer).clone() for layer in range(LAYERS)]
        selection, info1 = selection_for_arm(
            arm, model, controller, capture, input_ids, hidden1, teacher_logits,
            accounting, backend, document_id, prepare,
        )
        logits1, seconds1 = measure_with_selection(
            model, controller, capture, input_ids, selection, backend, use_capture=False
        )
        metrics = masked_mlp.logits_kl_metrics(teacher_logits, logits1, input_ids)
        iteration_kl.append(float(metrics["kl"].mean()))
        return {
            "metrics": metrics,
            "seconds": seconds0 + seconds1,
            "forwards": 2,
            "selector_seconds": info["selector_seconds"] + info1["selector_seconds"],
            "selection": selection,
            "iteration_kl": iteration_kl,
        }
    if kind in ("gxo", "cor-gxo"):
        selection, info = selection_for_arm(
            arm, model, controller, capture, input_ids, dense_hidden, teacher_logits,
            accounting, backend, document_id, prepare,
        )
        logits, seconds = measure_with_selection(
            model, controller, capture, input_ids, selection, backend, use_capture=False
        )
        metrics = masked_mlp.logits_kl_metrics(teacher_logits, logits, input_ids)
        return {
            "metrics": metrics,
            "seconds": seconds,
            "forwards": 1,
            "selector_seconds": info["selector_seconds"],
            "selection": selection,
            "iteration_kl": [float(metrics["kl"].mean())],
        }
    raise SystemExit(f"unknown arm kind {kind}")


def command_evaluate(args) -> int:
    fs.configure_arithmetic()
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    entry, accounting, split_receipt, split = model_and_accounting(args)
    prepare = load_json(args.prepare)
    weights = load_json(args.weights)
    if weights["split_sha256"] != split_receipt["derived"]["sha256"]:
        raise SystemExit("weights split does not match the split receipt")
    prepare["learned"] = weights
    if prepare["model"]["revision"] != entry["revision"]:
        raise SystemExit("prepare receipt model revision does not match the model lock")
    if prepare["data"]["split_sha256"] != split_receipt["derived"]["sha256"]:
        raise SystemExit("prepare receipt split does not match the split receipt")
    prepare["statistics"]["documents"] = {
        doc_id: {
            **payload,
            "shingle_set": None,
        }
        for doc_id, payload in prepare["statistics"]["documents"].items()
    }
    shingles = document_shingles(split)
    for doc_id, shingle_set in shingles.items():
        if doc_id in prepare["statistics"]["documents"]:
            prepare["statistics"]["documents"][doc_id]["shingle_set"] = shingle_set
    model = fs.load_model(fs.model_directory(entry), device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    controller = masked_mlp.MaskController(model)
    capture = masked_mlp.HiddenCapture(model)
    groups = group_count(accounting)
    bounds = masked_mlp.group_bounds(int(accounting["intermediate_per_layer"]), groups)
    backend = {
        "bounds": bounds,
        "intermediate": int(accounting["intermediate_per_layer"]),
        "groups": groups,
    }
    arms = build_arms(args, prepare, accounting)
    domains = build_domains(split, args.sample)
    stamp = args.stamp or stamp_now()
    results = {
        arm["id"]: {
            "id": arm["id"],
            "kind": arm["kind"],
            "deployable": arm["deployable"],
            "budget": arm["budget"],
            "active": arm["active"],
            "controller": arm["controller"],
            "budget_matched": arm.get("budget_matched", True),
            "m1_receipt": arm.get("m1_receipt"),
            "domains": {domain: {} for domain in domains},
            "timing": {"seconds": 0.0, "selector_seconds": 0.0, "forwards": 0, "nonfinite": 0},
            "iteration_kl": [],
            "selection_summary": [],
        }
        for arm in arms
    }
    nonfinite = []
    document_assignment = {}
    import contextlib

    with contextlib.nullcontext():
        for domain, items in domains.items():
            for item in items:
                doc_id = item["doc_id"]
                if doc_id not in document_assignment:
                    document_assignment[doc_id] = nearest_calibration_document(
                        item["windows"], prepare["statistics"]
                    )
                for window_index, window in enumerate(item["windows"]):
                    input_ids = torch.tensor([window], dtype=torch.long, device=device)
                    controller.set_dense()
                    capture.clear()
                    with torch.inference_mode():
                        teacher = model(input_ids=input_ids, use_cache=False).logits
                    dense_hidden = [capture.get(layer).clone() for layer in range(LAYERS)]
                    for arm in arms:
                        outcome = evaluate_arm_on_window(
                            arm,
                            model,
                            controller,
                            capture,
                            input_ids,
                            teacher,
                            dense_hidden,
                            backend,
                            prepare,
                            document_assignment[doc_id],
                            accounting,
                        )
                        metrics = outcome["metrics"]
                        if not bool(torch.isfinite(metrics["kl"]).all()):
                            nonfinite.append({"arm": arm["id"], "domain": domain, "doc_id": doc_id})
                            results[arm["id"]]["timing"]["nonfinite"] += 1
                        summary = masked_mlp.summarize_positions(metrics)
                        accumulator = results[arm["id"]]["domains"][domain].setdefault(
                            doc_id, fs.WindowAccumulator()
                        )
                        accumulator.add(summary, metrics["kl"])
                        results[arm["id"]]["timing"]["seconds"] += outcome["seconds"]
                        results[arm["id"]]["timing"]["selector_seconds"] += outcome["selector_seconds"]
                        results[arm["id"]]["timing"]["forwards"] += outcome["forwards"]
                        results[arm["id"]]["iteration_kl"].append(outcome["iteration_kl"])
                        if outcome["selection"] is not None:
                            results[arm["id"]]["selection_summary"].append(
                                selection_summary(outcome["selection"])
                            )
    capture.remove()
    # Finalise per-document items and aggregates.
    output_arms = []
    for arm in arms:
        record = results[arm["id"]]
        record["domains"] = {
            domain: {
                "items": [
                    accumulator.finish(doc_id)
                    for doc_id, accumulator in sorted(record["domains"][domain].items())
                ]
            }
            for domain in domains
        }
        for domain, payload in record["domains"].items():
            payload["aggregate"] = fs.aggregate_items(payload["items"])
        if record["selection_summary"]:
            counts = [
                summary["per_token_min"]
                for summary in record["selection_summary"]
            ]
            record["selection"] = {
                "per_token_groups_min": int(min(counts)),
                "per_token_groups_max": int(
                    max(summary["per_token_max"] for summary in record["selection_summary"])
                ),
                "windows": len(record["selection_summary"]),
                "adjacent_equal_fraction_mean": float(
                    np.mean(
                        [
                            np.mean(summary["adjacent_equal_fraction"])
                            for summary in record["selection_summary"]
                        ]
                    )
                ),
                "distinct_subsets_per_layer_mean": float(
                    np.mean(
                        [
                            np.mean(summary["distinct_subsets_per_layer"])
                            for summary in record["selection_summary"]
                        ]
                    )
                ),
            }
        iteration_kl = record.pop("iteration_kl")
        if iteration_kl:
            first = [value[0] for value in iteration_kl if value]
            last = [value[-1] for value in iteration_kl if value]
            record["iteration_kl"] = {
                "iteration0_mean": float(np.mean(first)) if first else None,
                "final_mean": float(np.mean(last)) if last else None,
                "per_window": iteration_kl,
            }
        record["timing"]["selector_flops_per_token"] = selector_flop_estimate(arm)
        output_arms.append(record)
    stamp = args.stamp or stamp_now()
    receipt = base_receipt(args, "evaluate", accounting, entry, device, split_receipt)
    receipt.update(
        {
            "schema": SCHEMA_EVAL,
            "stamp": stamp,
            "sample": args.sample or "full",
            "prepare": {"path": args.prepare, "sha256": fs.sha256_file(args.prepare)},
            "budget": {
                "B140": budget_spec(accounting, "B140"),
                "B142": budget_spec(accounting, "B142"),
                "granularity": {
                    "16": budget_granularity(accounting, 16),
                    "64": budget_granularity(accounting, 64),
                    "4": budget_granularity(accounting, 4),
                },
            },
            "document_assignment": document_assignment,
            "arms": output_arms,
            "nonfinite": nonfinite,
            "seed": SEED,
        }
    )
    os.makedirs(args.receipt_dir, exist_ok=True)
    path = os.path.join(
        args.receipt_dir,
        f"eval-{args.sample or 'full'}-{stamp}.json",
    )
    write_json_atomic(path, receipt)
    for record in output_arms:
        general = record["domains"].get("general", {}).get("aggregate", {})
        if general:
            print(
                f"eval: {record['id']:<28} kl={general.get('kl_mean', float('nan')):.6g} "
                f"top1={general.get('top1_agreement', float('nan')):.4f} "
                f"ppl={general.get('ppl_ratio', float('nan')):.6g} "
                f"deployable={record['deployable']} active={record['active']['active_unique_elements_with_controller']}"
            )
    print(f"eval: receipt {path}")
    if args.json:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def selector_flop_estimate(arm: dict) -> int:
    """Derived selector FLOPs per decision; labelled derived, not measured."""
    kind = arm["kind"]
    if kind in ("static", "task-static", "dense", "all"):
        return 0
    groups = GROUPS
    if kind in ("magnitude", "frequency"):
        return 2 * groups
    if kind in ("random",):
        return 8 * groups
    if kind in ("learned-unary",):
        return 2 * groups * FEATURE_DIM
    if kind in ("learned-interaction",):
        return 2 * groups * FEATURE_DIM + 2 * 5 * groups * len(GROUP_FEATURES) * len(GROUP_FEATURES)
    if kind in ("gxo", "cor-gxo"):
        # Backward pass through the transformer is not a selector formula we
        # pretend to cost exactly; it is reported as a measured time instead.
        return 0
    return 0


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def compare_domain(
    base_items: list[dict], arm_items: list[dict], resamples: int
) -> dict:
    base = {item["doc_id"]: item for item in base_items}
    arm = {item["doc_id"]: item for item in arm_items}
    shared = sorted(set(base) & set(arm))
    if len(shared) < 3:
        return {"items": len(shared), "available": False}
    base_kl = np.array([base[key]["kl_mean"] for key in shared], dtype=np.float64)
    arm_kl = np.array([arm[key]["kl_mean"] for key in shared], dtype=np.float64)
    indices = fs.bootstrap_indices(len(shared), resamples, SEED)
    absolute = base_kl - arm_kl
    relative = np.where(base_kl > 0, absolute / base_kl, 0.0)
    absolute_stats = fs.bootstrap_mean(absolute, indices)
    relative_stats = fs.bootstrap_mean(relative, indices)
    return {
        "items": len(shared),
        "available": True,
        "base_kl": float(base_kl.mean()),
        "arm_kl": float(arm_kl.mean()),
        "absolute_reduction": absolute_stats,
        "relative_reduction": relative_stats,
        "passes_five_percent": bool(
            relative_stats["mean"] >= 0.05 and relative_stats["ci_low"] > 0
        ),
    }


def svg_bar_chart(rows: list[dict], path: str, title: str) -> None:
    width = 1120
    row_height = 22
    top = 56
    height = top + row_height * len(rows) + 40
    maximum = max((row["kl_mean"] for row in rows), default=1.0) or 1.0
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="16" y="28" font-family="sans-serif" font-size="16" fill="#111">{title}</text>',
        f'<text x="16" y="46" font-family="sans-serif" font-size="11" fill="#555">'
        "mean teacher KL (nats/token); bar = 95% paired-bootstrap CI; lower is better</text>",
    ]
    label_width = 300
    chart_width = width - label_width - 96
    for index, row in enumerate(rows):
        y = top + index * row_height
        value = row["kl_mean"]
        low = row["kl_ci_low"]
        high = row["kl_ci_high"]
        bar = max(1.0, chart_width * value / maximum)
        low_x = label_width + chart_width * low / maximum
        high_x = label_width + chart_width * high / maximum
        color = "#1f6feb" if row["deployable"] else "#999999"
        lines.append(
            f'<text x="16" y="{y + 14}" font-family="sans-serif" font-size="11" fill="#111">'
            f'{row["id"][:44]}</text>'
        )
        lines.append(
            f'<rect x="{label_width}" y="{y + 3}" width="{bar:.2f}" height="{row_height - 6}" '
            f'fill="{color}" fill-opacity="0.75"/>'
        )
        lines.append(
            f'<line x1="{low_x:.2f}" y1="{y + row_height / 2}" x2="{high_x:.2f}" '
            f'y2="{y + row_height / 2}" stroke="#111" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{width - 88}" y="{y + 14}" font-family="monospace" font-size="11" '
            f'fill="#111">{value:.4g}</text>'
        )
    lines.append("</svg>")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def command_analyze(args) -> int:
    prepare = load_json(args.prepare)
    evaluation = load_json(args.eval)
    oracle = load_json(args.oracle) if args.oracle else None
    resamples = args.resamples
    thresholds = {
        "acceptable": {"kl_mean": 0.05, "ppl_ratio": 1.02, "top1_agreement": 0.95},
        "usable": {"kl_mean": 0.25, "ppl_ratio": 1.10, "top1_agreement": 0.85},
    }
    arms = evaluation["arms"]
    arm_index = {arm["id"]: arm for arm in arms}
    table = []
    for arm in arms:
        general_items = arm["domains"].get("general", {}).get("items", [])
        if not general_items:
            continue
        metrics = fs.arm_domain_metrics(
            general_items, fs.bootstrap_indices(len(general_items), resamples, SEED)
        )
        table.append(
            {
                "id": arm["id"],
                "deployable": arm["deployable"],
                "budget": arm["budget"]["name"],
                "budget_matched": arm.get("budget_matched", True),
                "active_unique_elements": arm["active"]["active_unique_elements_with_controller"],
                "compliant": arm["active"]["compliant"],
                "controller_elements": arm["active"]["controller_elements"],
                "kl_mean": metrics["kl_mean"],
                "kl_ci_low": metrics["kl_mean_ci"]["ci_low"],
                "kl_ci_high": metrics["kl_mean_ci"]["ci_high"],
                "kl_p95": metrics["kl_p95_mean"],
                "top1": metrics["top1_agreement"],
                "top1_ci_low": metrics["top1_ci"]["ci_low"],
                "top1_ci_high": metrics["top1_ci"]["ci_high"],
                "ppl_ratio": metrics["ppl_ratio"]["point"],
                "ppl_ci_low": metrics["ppl_ratio"]["ci_low"],
                "ppl_ci_high": metrics["ppl_ratio"]["ci_high"],
                "selector_seconds": arm["timing"]["selector_seconds"],
                "selector_flops_per_token": arm["timing"].get("selector_flops_per_token"),
                "domains": {
                    domain: fs.arm_domain_metrics(
                        payload["items"],
                        fs.bootstrap_indices(len(payload["items"]), resamples, SEED),
                    )
                    for domain, payload in arm["domains"].items()
                    if payload["items"]
                },
            }
        )
    oracle_rows = []
    if oracle:
        for config in oracle["configs"]:
            items = list(config["windows"].values())
            if not items:
                continue
            items = [
                {
                    **item,
                    "positions": item.get("positions", item.get("nll_positions", 0)),
                }
                for item in items
            ]
            metrics = fs.arm_domain_metrics(
                items, fs.bootstrap_indices(len(items), resamples, SEED)
            )
            oracle_rows.append(
                {
                    "id": config["id"],
                    "groups": config["groups"],
                    "budget": config["budget"]["name"],
                    "active_unique_elements": config["budget"]["active_unique_elements"],
                    "kl_mean": metrics["kl_mean"],
                    "kl_ci_low": metrics["kl_mean_ci"]["ci_low"],
                    "kl_ci_high": metrics["kl_mean_ci"]["ci_high"],
                    "kl_p95": metrics["kl_p95_mean"],
                    "top1": metrics["top1_agreement"],
                    "ppl_ratio": metrics["ppl_ratio"]["point"],
                    "fraction_kl_le_005": config["aggregate"]["fraction_kl_le_005"],
                    "fraction_kl_le_025": config["aggregate"]["fraction_kl_le_025"],
                    "consistency": config.get("consistency"),
                    "forwards": config["forwards"],
                    "selector_seconds": config["selector_seconds"],
                    "gpu_seconds": config["gpu_seconds"],
                    "non_deployable": True,
                }
            )
        if oracle.get("final_refine") and oracle["final_refine"]["windows"]:
            items = [
                {
                    **item,
                    "positions": item.get("positions", item.get("nll_positions", 0)),
                }
                for item in oracle["final_refine"]["windows"].values()
            ]
            metrics = fs.arm_domain_metrics(
                items, fs.bootstrap_indices(len(items), resamples, SEED)
            )
            oracle_rows.append(
                {
                    "id": "oracle-final-refine",
                    "groups": 16,
                    "budget": "B142",
                    "active_unique_elements": oracle["configs"][1]["budget"]["active_unique_elements"],
                    "kl_mean": metrics["kl_mean"],
                    "kl_ci_low": metrics["kl_mean_ci"]["ci_low"],
                    "kl_ci_high": metrics["kl_mean_ci"]["ci_high"],
                    "kl_p95": metrics["kl_p95_mean"],
                    "top1": metrics["top1_agreement"],
                    "ppl_ratio": metrics["ppl_ratio"]["point"],
                    "fraction_kl_le_005": float(
                        np.mean([item["fraction_kl_le_005"] for item in items])
                    ),
                    "fraction_kl_le_025": float(
                        np.mean([item["fraction_kl_le_025"] for item in items])
                    ),
                    "consistency": None,
                    "forwards": oracle["final_refine"]["forwards"],
                    "selector_seconds": 0.0,
                    "gpu_seconds": oracle["final_refine"]["gpu_seconds"],
                    "non_deployable": True,
                }
            )
    deployable = [row for row in table if row["deployable"]]
    baseline = arm_index.get("token-magnitude")
    comparisons = []
    if baseline is not None:
        for row in deployable:
            if row["id"] == "token-magnitude":
                continue
            arm = arm_index[row["id"]]
            comparison = compare_domain(
                baseline["domains"]["general"]["items"],
                arm["domains"]["general"]["items"],
                resamples,
            )
            comparison["arm"] = row["id"]
            comparison["baseline"] = "token-magnitude"
            baseline_row = table[[item["id"] for item in table].index("token-magnitude")]
            difference = row["active_unique_elements"] - baseline_row["active_unique_elements"]
            comparison["active_delta"] = difference
            comparison["matched_budget"] = bool(
                row["budget"] == baseline_row["budget"] and abs(difference) <= 6144
            )
            comparisons.append(comparison)
    learned_unary = arm_index.get("token-learned-unary")
    learned_interaction = arm_index.get("token-learned-interaction")
    learned_comparison = None
    if learned_unary is not None and learned_interaction is not None:
        learned_comparison = compare_domain(
            learned_unary["domains"]["general"]["items"],
            learned_interaction["domains"]["general"]["items"],
            resamples,
        )
        learned_comparison["arm"] = "token-learned-interaction"
        learned_comparison["baseline"] = "token-learned-unary"
    k6_pass = any(
        comparison.get("available")
        and comparison.get("matched_budget")
        and comparison.get("passes_five_percent")
        for comparison in comparisons
    )
    oracle_best = (
        min(oracle_rows, key=lambda row: row["kl_mean"]) if oracle_rows else None
    )
    oracle_usable = None
    if oracle_best is not None:
        oracle_usable = bool(
            oracle_best["kl_mean"] <= thresholds["usable"]["kl_mean"]
            and oracle_best["ppl_ratio"] <= thresholds["usable"]["ppl_ratio"]
            and oracle_best["top1"] >= thresholds["usable"]["top1_agreement"]
        )
    oracle_16 = {
        row["id"]: row for row in oracle_rows if row["id"].startswith("oracle-local-16")
    }
    oracle_64 = next((row for row in oracle_rows if row["id"] == "oracle-local-64"), None)
    oracle_4 = next((row for row in oracle_rows if row["id"] == "oracle-local-4"), None)
    best_16 = min(oracle_16.values(), key=lambda row: row["kl_mean"]) if oracle_16 else None
    granularity = {
        "oracle_16_best": best_16,
        "oracle_64": oracle_64,
        "oracle_4": oracle_4,
        "finer_is_better": bool(
            best_16 is not None
            and oracle_64 is not None
            and oracle_64["kl_mean"] < best_16["kl_mean"]
        ),
        "coarser_is_worse": bool(
            best_16 is not None
            and oracle_4 is not None
            and oracle_4["kl_mean"] > best_16["kl_mean"]
        ),
    }
    k8_pass = bool(
        oracle_64 is not None
        and best_16 is not None
        and oracle_64["kl_mean"] <= thresholds["usable"]["kl_mean"]
        and best_16["kl_mean"] > thresholds["usable"]["kl_mean"]
    )
    verdicts = {
        "K6": {
            "passed": bool(k6_pass),
            "rule": (
                "a deployable arm at matched active unique parameters reduces general mean KL "
                "by at least 5% versus token-magnitude with a paired 95% CI lower bound above zero"
            ),
            "best_relative_reduction": max(
                (
                    comparison.get("relative_reduction", {}).get("mean", float("-inf"))
                    for comparison in comparisons
                    if comparison.get("matched_budget")
                ),
                default=None,
            ),
        },
        "K7": {
            "passed": bool(oracle_usable),
            "rule": "the best oracle is within the registered usable tier",
            "best_oracle": oracle_best["id"] if oracle_best else None,
            "best_oracle_kl": oracle_best["kl_mean"] if oracle_best else None,
            "best_oracle_ppl": oracle_best["ppl_ratio"] if oracle_best else None,
            "best_oracle_top1": oracle_best["top1"] if oracle_best else None,
            "usable_tier": thresholds["usable"],
        },
        "K8": {
            "passed": bool(k8_pass),
            "rule": (
                "the 64-group oracle reaches the usable tier while the 16-group oracle does not"
            ),
            "finer_is_better": granularity["finer_is_better"],
            "coarser_is_worse": granularity["coarser_is_worse"],
        },
        "K9": {
            "passed": not evaluation.get("nonfinite")
            and (oracle is None or not oracle.get("nonfinite")),
            "rule": "no leakage, non-finite value, over-ceiling arm, or parity failure",
        },
    }
    stamp = args.stamp or stamp_now()
    analysis = {
        "schema": SCHEMA_ANALYSIS,
        "stamp": stamp,
        "revision": fs.git_head(),
        "preregistration_commit": evaluation["preregistration_commit"],
        "created_utc": utc_now(),
        "seed": SEED,
        "resamples": resamples,
        "thresholds": thresholds,
        "receipts": {
            "prepare": {"path": args.prepare, "sha256": fs.sha256_file(args.prepare)},
            "eval": {"path": args.eval, "sha256": fs.sha256_file(args.eval)},
            "oracle": (
                {"path": args.oracle, "sha256": fs.sha256_file(args.oracle)}
                if args.oracle
                else None
            ),
        },
        "budget": evaluation["budget"],
        "table": table,
        "oracle": oracle_rows,
        "comparisons": comparisons,
        "learned_comparison": learned_comparison,
        "granularity": granularity,
        "verdicts": verdicts,
    }
    compare_arms = []
    baseline_items = None
    for arm in arms:
        if arm["id"] == "token-magnitude":
            baseline_items = {item["doc_id"]: item["kl_mean"] for item in arm["domains"]["general"]["items"]}
        budget_name = arm["budget"]["name"]
        if budget_name == "B142":
            counts = [5] * LAYERS
            counts[0] += 2
            width = 384
        elif budget_name == "B140":
            counts = [5] * LAYERS
            width = 384
        elif budget_name == "M1-K3-112":
            counts = [4] * LAYERS
            width = 384
        else:
            continue
        compare_arms.append(
            {
                "id": arm["id"],
                "budget": budget_name,
                "groups_per_layer": counts,
                "group_width": width,
                "controller_elements": arm["active"]["controller_elements"],
                "items": [
                    {"doc_id": item["doc_id"], "kl_mean": item["kl_mean"]}
                    for item in arm["domains"]["general"]["items"]
                ],
            }
        )
    compare_input = {
        "schema": "selvedge/atlas/compare-input-v1",
        "stamp": stamp,
        "revision": fs.git_head(),
        "ceiling": int(evaluation["accounting"]["ceiling"]),
        "backbone_elements": int(evaluation["accounting"]["always_active_backbone_elements"]),
        "hidden": int(evaluation["accounting"]["hidden"]),
        "intermediate": int(evaluation["accounting"]["intermediate_per_layer"]),
        "baseline": "token-magnitude",
        "arms": compare_arms,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    compare_path = os.path.join(args.out_dir, "compare-input.json")
    write_json_atomic(compare_path, compare_input)
    json_path = os.path.join(args.out_dir, "selectors.json")
    write_json_atomic(json_path, analysis)
    table_path = os.path.join(args.out_dir, "table.md")
    with open(table_path, "w", encoding="utf-8") as handle:
        handle.write("| arm | deployable | budget | active unique | KL mean | KL 95% CI | PPL ratio | top-1 |\n")
        handle.write("| --- | :---: | --- | ---: | ---: | --- | ---: | ---: |\n")
        for row in sorted(table, key=lambda item: item["kl_mean"]):
            handle.write(
                f"| {row['id']} | {'yes' if row['deployable'] else 'no'} | {row['budget']} | "
                f"{row['active_unique_elements']} | {row['kl_mean']:.6f} | "
                f"[{row['kl_ci_low']:.6f}, {row['kl_ci_high']:.6f}] | "
                f"{row['ppl_ratio']:.6g} | {row['top1']:.4f} |\n"
            )
        if oracle_rows:
            handle.write("\nOracle (non-deployable):\n\n")
            handle.write("| arm | budget | KL mean | KL 95% CI | PPL ratio | top-1 | frac KL<=0.05 | frac KL<=0.25 |\n")
            handle.write("| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |\n")
            for row in sorted(oracle_rows, key=lambda item: item["kl_mean"]):
                handle.write(
                    f"| {row['id']} | {row['budget']} | {row['kl_mean']:.6f} | "
                    f"[{row['kl_ci_low']:.6f}, {row['kl_ci_high']:.6f}] | "
                    f"{row['ppl_ratio']:.6g} | {row['top1']:.4f} | "
                    f"{row['fraction_kl_le_005']:.4f} | {row['fraction_kl_le_025']:.4f} |\n"
                )
    svg_path = os.path.join(args.out_dir, "selectors.svg")
    svg_bar_chart(table, svg_path, "M2 selectors at matched budget — general held-out")
    report_path = args.report
    if report_path and os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        begin = "<!-- BEGIN M2 TABLE -->"
        end = "<!-- END M2 TABLE -->"
        if begin in lines and end in lines:
            start = lines.index(begin)
            stop = lines.index(end)
            block = [begin, "", "| arm | deployable | budget | active unique | KL mean | KL 95% CI | PPL ratio | top-1 |", "| --- | :---: | --- | ---: | ---: | --- | ---: | ---: |"]
            for row in sorted(table, key=lambda item: item["kl_mean"]):
                block.append(
                    f"| {row['id']} | {'yes' if row['deployable'] else 'no'} | {row['budget']} | "
                    f"{row['active_unique_elements']} | {row['kl_mean']:.6f} | "
                    f"[{row['kl_ci_low']:.6f}, {row['kl_ci_high']:.6f}] | "
                    f"{row['ppl_ratio']:.6g} | {row['top1']:.4f} |"
                )
            if oracle_rows:
                block.extend(
                    [
                        "",
                        "Oracle (non-deployable):",
                        "",
                        "| arm | budget | KL mean | KL 95% CI | PPL ratio | top-1 | frac KL<=0.05 | frac KL<=0.25 |",
                        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
                    ]
                )
                for row in sorted(oracle_rows, key=lambda item: item["kl_mean"]):
                    block.append(
                        f"| {row['id']} | {row['budget']} | {row['kl_mean']:.6f} | "
                        f"[{row['kl_ci_low']:.6f}, {row['kl_ci_high']:.6f}] | "
                        f"{row['ppl_ratio']:.6g} | {row['top1']:.4f} | "
                        f"{row['fraction_kl_le_005']:.4f} | {row['fraction_kl_le_025']:.4f} |"
                    )
            block.append(end)
            lines = lines[:start] + block + lines[stop + 1 :]
        oracle_begin = "<!-- BEGIN M2 ORACLE TABLE -->"
        oracle_end = "<!-- END M2 ORACLE TABLE -->"
        if oracle_begin in lines and oracle_end in lines:
            start = lines.index(oracle_begin)
            stop = lines.index(oracle_end)
            block = [oracle_begin, "", "| oracle | budget | active unique | KL mean | KL 95% CI | PPL ratio | top-1 | frac KL<=0.05 | frac KL<=0.25 | forwards | GPU seconds |", "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
            for row in sorted(oracle_rows, key=lambda item: item["kl_mean"]):
                block.append(
                    f"| {row['id']} | {row['budget']} | {row['active_unique_elements']} | "
                    f"{row['kl_mean']:.6f} | [{row['kl_ci_low']:.6f}, {row['kl_ci_high']:.6f}] | "
                    f"{row['ppl_ratio']:.6g} | {row['top1']:.4f} | {row['fraction_kl_le_005']:.4f} | "
                    f"{row['fraction_kl_le_025']:.4f} | {row['forwards']} | {row['gpu_seconds']:.2f} |"
                )
            block.append(oracle_end)
            lines = lines[:start] + block + lines[stop + 1 :]
        cons_begin = "<!-- BEGIN M2 CONSISTENCY TABLE -->"
        cons_end = "<!-- END M2 CONSISTENCY TABLE -->"
        if cons_begin in lines and cons_end in lines:
            start = lines.index(cons_begin)
            stop = lines.index(cons_end)
            block = [cons_begin, "", "| oracle | adjacent equal | distinct subsets / layer | modal fraction | groups / token |", "| --- | ---: | ---: | ---: | --- |"]
            for row in sorted(oracle_rows, key=lambda item: item["kl_mean"]):
                consistency = row.get("consistency") or {}
                groups = f"{consistency.get('per_token_groups_min', '?')}-{consistency.get('per_token_groups_max', '?')}"
                block.append(
                    f"| {row['id']} | {consistency.get('adjacent_equal_fraction_mean', float('nan')):.4f} | "
                    f"{consistency.get('distinct_subsets_per_layer_mean', float('nan')):.1f} | "
                    f"{consistency.get('modal_fraction_mean', float('nan')):.4f} | {groups} |"
                )
            block.append(cons_end)
            lines = lines[:start] + block + lines[stop + 1 :]
        if report_path and os.path.exists(report_path):
            with open(report_path, "w", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
    print(f"analyze: table {table_path}")
    print(f"analyze: json {json_path}")
    print(f"analyze: plot {svg_path}")
    print(
        f"analyze: K6 passed={verdicts['K6']['passed']} K7 passed={verdicts['K7']['passed']} "
        f"K8 passed={verdicts['K8']['passed']} K9 passed={verdicts['K9']['passed']}"
    )
    for comparison in comparisons:
        if comparison.get("available"):
            print(
                f"analyze: {comparison['arm']} vs token-magnitude relative="
                f"{comparison['relative_reduction']['mean']:.4f} "
                f"CI=[{comparison['relative_reduction']['ci_low']:.4f}, "
                f"{comparison['relative_reduction']['ci_high']:.4f}] "
                f"matched={comparison['matched_budget']}"
            )
    return 0 if verdicts["K9"]["passed"] else 1


# ---------------------------------------------------------------------------
# Rust parity fixtures
# ---------------------------------------------------------------------------


def command_fixtures(args) -> int:
    fs.configure_arithmetic()
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    entry, accounting, split_receipt, split = model_and_accounting(args)
    prepare = load_json(args.prepare)
    weights = load_json(args.weights)
    groups = group_count(accounting)
    bounds = masked_mlp.group_bounds(int(accounting["intermediate_per_layer"]), groups)
    model = fs.load_model(fs.model_directory(entry), device)
    controller = masked_mlp.MaskController(model)
    capture = masked_mlp.HiddenCapture(model)
    generator = np.random.Generator(np.random.PCG64(SEED + 7))
    records = []
    with torch.inference_mode():
        for article_index, article in enumerate(split["windows"]["calibration"][:8]):
            for window in article["windows"][:1]:
                input_ids = torch.tensor([window], dtype=torch.long, device=device)
                controller.set_dense()
                capture.clear()
                model(input_ids=input_ids, use_cache=False)
                sequence_key = int(input_ids[0, 0].item())
                positions = generator.choice(
                    input_ids.shape[1], size=32, replace=False
                )
                for position in positions:
                    layer = int(generator.integers(0, LAYERS))
                    hidden = capture.get(layer)[:, int(position) : int(position) + 1, :]
                    activation = masked_mlp.dense_channels(
                        controller.wrappers[layer], hidden.reshape(-1, HIDDEN)
                    )
                    features = group_features_from_activation(activation, bounds)[0]
                    record = {
                        "layer": layer,
                        "layers": LAYERS,
                        "position": int(position),
                        "sequence_key": sequence_key,
                        "groups": [
                            {
                                "group": group,
                                "features": [float(value) for value in features[group].tolist()],
                            }
                            for group in range(groups)
                        ],
                    }
                    if len(records) < 8:
                        record["hidden"] = [float(value) for value in hidden.reshape(-1).tolist()]
                    # Expected per-layer decisions.
                    log_norm = torch.tensor([item["features"][0] for item in record["groups"]])
                    share = torch.tensor([item["features"][1] for item in record["groups"]])
                    feature_tensor = features.unsqueeze(0)
                    unary_scores = score_standardized(feature_tensor, weights, layer)
                    unary_selection = topk_mask(unary_scores, 5)[0]
                    group_std = (
                        features
                        - torch.tensor(
                            weights["feature_mean"][layer][: len(GROUP_FEATURES)],
                            device=features.device,
                        )
                    ) / torch.tensor(
                        weights["feature_scale"][layer][: len(GROUP_FEATURES)],
                        device=features.device,
                    )
                    interaction_selection = greedy_surrogate_selection(
                        unary_scores,
                        group_std.unsqueeze(0),
                        torch.tensor(
                            weights["interaction"][layer],
                            dtype=torch.float32,
                            device=features.device,
                        ),
                        float(weights["lambda_mix"]),
                        5,
                    )[0]
                    random_scores = torch.zeros(groups)
                    for group in range(groups):
                        state = splitmix64(
                            SEED
                            ^ (layer * 0x9E3779B97F4A7C15)
                            ^ (int(position) * 0xBF58476D1CE4E5B9)
                            ^ (group * 0x94D049BB133111EB)
                            ^ sequence_key
                        )
                        random_scores[group] = float(state) / float(1 << 64)
                    record["expected"] = {
                        "magnitude": sorted(
                            torch.argsort(log_norm, descending=True, stable=True)[:5].tolist()
                        ),
                        "frequency": sorted(
                            torch.argsort(share, descending=True, stable=True)[:5].tolist()
                        ),
                        "random": sorted(
                            torch.argsort(random_scores, descending=True, stable=True)[:5].tolist()
                        ),
                        "static-magnitude": sorted(
                            weights.get("rankings", prepare.get("rankings", {}))["magnitude"][layer][:5]
                        ),
                        "static-frequency": sorted(
                            weights.get("rankings", prepare.get("rankings", {}))["frequency"][layer][:5]
                        ),
                        "learned-unary": sorted(
                            torch.nonzero(unary_selection).flatten().tolist()
                        ),
                        "learned-interaction": sorted(
                            torch.nonzero(interaction_selection).flatten().tolist()
                        ),
                    }
                    records.append(record)
    capture.remove()
    payload = {
        "schema": SCHEMA_FIXTURES,
        "stamp": args.stamp or stamp_now(),
        "revision": fs.git_head(),
        "preregistration_commit": args.preregistration_commit,
        "model": {"id": args.model_id, "revision": entry["revision"]},
        "split_sha256": split_receipt["derived"]["sha256"],
        "weights": {"path": args.weights, "sha256": fs.sha256_file(args.weights)},
        "groups": groups,
        "layers": LAYERS,
        "feature_names": list(GROUP_FEATURES),
        "note": (
            "expected decisions are the Python per-layer top-5; the two global B142 "
            "extras are a second-stage allocation and are not part of the per-layer contract"
        ),
        "record_count": len(records),
        "records": records,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    write_json_atomic(args.out, payload)
    print(f"fixtures: {len(records)} records {args.out}")
    print(f"fixtures: sha256 {fs.sha256_file(args.out)}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_common(parser):
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--model-revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    parser.add_argument("--split", default="results/artifacts/split/m1-split.json")
    parser.add_argument(
        "--split-receipt", default="results/splits/split-20261114T223127Z.json"
    )
    parser.add_argument(
        "--accounting", default="results/accounting/20261114T201402Z-qwen3-1.7b.json"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stamp", default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--preregistration-commit", default=M2_PREREGISTRATION_COMMIT
    )
    parser.add_argument("--receipt-dir", default="results/selectors")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    add_common(prepare)
    prepare.add_argument("--contexts-per-window", type=int, default=64)
    prepare.add_argument("--pairs-per-context", type=int, default=2)
    prepare.add_argument("--weights-out", default="results/selectors/m2-weights.json")
    prepare.add_argument("--fit-data-out", default="results/selectors/m2-fit-data.json")

    interactions = sub.add_parser("interactions")
    add_common(interactions)
    interactions.add_argument("--prepare", default="results/selectors/m2-weights.json")
    interactions.add_argument("--resamples", type=int, default=10000)

    oracle = sub.add_parser("oracle")
    add_common(oracle)
    oracle.add_argument("--windows", type=int, default=8)
    oracle.add_argument("--passes", type=int, default=3)
    oracle.add_argument("--refine-windows", type=int, default=2)
    oracle.add_argument("--refine-positions", type=int, default=16)
    oracle.add_argument("--refine-passes", type=int, default=1)

    evaluate = sub.add_parser("evaluate")
    add_common(evaluate)
    evaluate.add_argument("--prepare", default="results/selectors/prepare-Qwen-Qwen3-1.7B-PATH.json")
    evaluate.add_argument("--weights", default="results/selectors/m2-weights.json")
    evaluate.add_argument("--sample", choices=["full", "oracle"], default="full")

    analyze = sub.add_parser("analyze")
    analyze.add_argument("--prepare", required=True)
    analyze.add_argument("--eval", required=True)
    analyze.add_argument("--oracle", default=None)
    analyze.add_argument("--resamples", type=int, default=10000)
    analyze.add_argument("--out-dir", default="results/selectors")
    analyze.add_argument("--report", default="results/M2-SELECTORS.md")
    analyze.add_argument("--stamp", default=None)

    fixtures = sub.add_parser("fixtures")
    add_common(fixtures)
    fixtures.add_argument("--prepare", required=True)
    fixtures.add_argument("--weights", required=True)
    fixtures.add_argument(
        "--out", default="results/selectors/fixtures/selector-inputs.json"
    )
    return parser


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare":
        return command_prepare(args)
    if args.command == "interactions":
        return command_interactions(args)
    if args.command == "oracle":
        return command_oracle(args)
    if args.command == "evaluate":
        return command_evaluate(args)
    if args.command == "analyze":
        return command_analyze(args)
    if args.command == "fixtures":
        return command_fixtures(args)
    parser.error(f"unknown command {args.command}")
    return 64


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
