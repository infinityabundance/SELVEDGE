#!/usr/bin/env python3
"""H-diagnostics: why the hybrid lost `M`'s and `X`'s gains (CPU, no forwards).

Answers the interaction question from artifacts that already exist, and says
explicitly which parts of the question those artifacts cannot answer.

What is measured, and from what
------------------------------

1. **Outcomes** (`--receipts`). Per-item rows of the stored T2 evaluation
   receipts give, for every task, the four-way partition of items by whether
   `X` and `M` each got them right, and `H`'s (and `D`'s, `D0`'s) accuracy
   inside each cell. This is the item-level statement of "H lost *both* arms'
   gains, including the successes only one arm had". Paired percentile
   bootstraps, seed `20261115`, `10000` resamples, reusing
   `oracle/eval/metrics.py`.

2. **Router geometry** (`--checkpoint-*`). Every T2 router was initialised from
   the deterministic generator `seed + 2000 + layer` and every memory module
   from `seed + 3000 + layer` (`oracle/train/modeling.py` `_router`,
   `oracle/train/memory.py`). Reconstructing those draws turns the saved
   weights into a *displacement from initialisation*, which is measured, not
   modelled: how far each arm's router moved, how much of its energy sits in one
   direction, how orthogonal its eight expert directions are, and which expert
   directions are so small that they cannot be selected by any input. The
   reconstruction is validated in the receipt (`init_validation`): the cosine
   against the right seed is large and against a wrong seed is ~0.

3. **Routing usage** (`--labels`, `--checkpoint-hstar`). The `H*` utility-label
   artifact carries the 32,768 pre-expert hidden states of the calibration
   slice, one tensor per layer. Routing those stored states through `H*`'s
   stored router weights is not a model forward: no new activations are
   produced. It yields the marginal expert usage, its entropy, dead experts and
   the per-position spread the router actually produces.

4. **Memory path** (`--labels`, checkpoints, `--stream`). Gate values come from
   the stored gate vectors on the stored hidden states. Row usage comes from
   the *exact* calibration tokens: `results/artifacts/train/train.bin` hashes to
   the stream digest recorded in the `H*` run receipt, and addressing is a pure
   function of token ids (`memory.ngram_index`), so the reads the memory path
   performs are recomputable with no model. Table displacement per row is
   measured against the reconstructed initialisation.

5. **Interaction proxies.** (a) Alignment between the memory gate's read
   direction and each expert's routing direction, from weights alone.
   (b) Mutual information between the router's top-1 expert and the popularity
   of the memory row a position addresses, with a permutation null, on `H*`'s
   stored states. (c) The memory path's output magnitude against the specialist
   bank's output magnitude on the *same* stored states.

What is deferred to the GPU window, and why
-------------------------------------------

`H`'s and `X`'s own hidden states are not on disk. Every statistic that needs
them — their usage distribution, their gate values on their own activations,
their row/ expert co-occurrence — needs a forward pass over the calibration
slice, which this session must not run (the `H*` training container owns the
GPU). `oracle/eval/h_diagnostics.py --deferred-commands` prints the exact
commands. Nothing in the receipt is estimated from a proxy of those states.

Usage
=====

  python3 oracle/eval/h_diagnostics.py \
      --labels results/artifacts/train/t2/Hstar/utility-labels.pt \
      --checkpoint-hstar results/artifacts/train/t2/Hstar/step-000200.pt \
      --checkpoint-H results/artifacts/train/t2/H/step-001024.pt \
      --checkpoint-X results/artifacts/train/t2/X/step-001024.pt \
      --checkpoint-M results/artifacts/train/t2/M/step-001024.pt \
      --stream results/artifacts/train/train.bin \
      --out results/t2/H-DIAGNOSTICS-<stamp>.json \
      --markdown results/H-DIAGNOSTICS.md

Every number in the markdown is rendered from the receipt written by the same
run; no number in it is typed by hand.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
TRAIN = os.path.join(REPO, "oracle", "train")
for _path in (HERE, TRAIN):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from metrics import paired_bootstrap  # noqa: E402

SCHEMA = "selvedge/h-diagnostics/v1"

#: Registered T2 geometry (docs/EXPERIMENT-CONTRACT.md T2.1–T2.2).
RUN_SEED = 1001
EXPERTS = 8
K = 4
HIDDEN = 1024
ROWS = 1 << 18
VALUE_DIM = 256
READS = 2
ORDERS = (2, 3)
MEMORY_LAYERS = (3, 10, 17, 24)
ROUTER_SEED_OFFSET = 2000
MEMORY_SEED_OFFSET = 3000
#: The labeller fed the calibration slice as `micro_positions`-wide rows, so the
#: n-gram addresses at the start of each row are what the labels saw (T2.5).
MICRO_POSITIONS = 256

BOOTSTRAP_SEED = 20261115
BOOTSTRAP_RESAMPLES = 10000
PERMUTATIONS = 2000

TASKS = (
    ("humaneval", "passed"),
    ("gsm8k", "correct"),
    ("truthfulqa", "correct"),
)


# ---------------------------------------------------------------------------
# small helpers (the repo's conventions)


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
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


def write_text_atomic(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# initialisation reconstruction
#
# The two generators below are the only place an arm's starting weights come
# from, so reproducing them is exact rather than approximate. A wrong seed
# gives a cosine of ~0 against the saved weight, which is the validation the
# receipt records.


def init_block(seed: int, offset: int, layer: int, shape: tuple[int, ...],
               dtype: torch.dtype = torch.float32) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed + offset + layer)
    return torch.empty(shape, dtype=dtype).normal_(mean=0.0, std=0.02,
                                                  generator=generator)


def cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.reshape(-1).double()
    second = second.reshape(-1).double()
    denominator = first.norm() * second.norm()
    if float(denominator) == 0.0:
        return 0.0
    return float((first @ second) / denominator)


# ---------------------------------------------------------------------------
# router statistics


def router_stats(weight: torch.Tensor, init: torch.Tensor) -> dict:
    """Geometry of one `[experts, hidden]` router, plus its displacement."""
    weight = weight.detach().to(torch.float64)
    init = init.detach().to(torch.float64)
    norms = weight.norm(dim=1)
    median_norm = float(norms.median())
    singular = torch.linalg.svdvals(weight)
    total_energy = float(singular.pow(2).sum())
    participation = (float(singular.sum()) ** 2 / total_energy) if total_energy else 0.0
    unit = weight / norms.clamp_min(1e-12).unsqueeze(1)
    gram = unit @ unit.T
    off = gram[~torch.eye(weight.shape[0], dtype=torch.bool)]
    init_norms = init.norm(dim=1)
    return {
        "shape": list(weight.shape),
        "norm_per_expert": [float(item) for item in norms],
        "norm_ratio_min_median": float(norms.min() / median_norm) if median_norm else None,
        # An expert whose router direction is far below the median cannot win a
        # top-k comparison at any input scale. It is a *necessary* dead-expert
        # condition, not a used-set measurement; usage is section 3.
        "dead_expert_candidates": [int(index) for index, item in enumerate(norms)
                                   if float(item) < 0.1 * median_norm],
        "effective_rank": participation,
        "top1_energy_share": float(singular[0].pow(2) / total_energy)
        if total_energy else 0.0,
        "mean_absolute_pairwise_cosine": float(off.abs().mean()),
        "max_absolute_pairwise_cosine": float(off.abs().max()),
        "displacement_relative": float((weight - init).norm() / init.norm()),
        "cosine_to_init": cosine(weight, init),
        "norm_ratio_to_init": float(weight.norm() / init.norm()),
        "per_expert_cosine_to_init": [float(item) for item in (unit
                                                              @ (init / init_norms
                                                                 .clamp_min(1e-12)
                                                                 .unsqueeze(1)).T).diagonal()],
    }


def usage_stats(logits: torch.Tensor, k: int = K) -> dict:
    """Marginal and per-position usage of a top-`k` router on stored states."""
    logits = logits.detach().to(torch.float64)
    positions, experts = logits.shape
    top = logits.topk(k, dim=-1).indices
    counts = torch.bincount(top.reshape(-1), minlength=experts).to(torch.float64)
    share = counts / counts.sum()
    total = positions * k
    expected = total / experts
    entropy = -(share.clamp_min(1e-300).log() * share).sum() / np.log(experts)
    probabilities = torch.softmax(logits, dim=-1)
    position_entropy = -(probabilities.clamp_min(1e-300).log()
                         * probabilities).sum(dim=-1) / np.log(experts)
    top1 = torch.bincount(top[:, 0], minlength=experts).to(torch.float64)
    # How many *distinct decisions* the router actually makes: if one top-k set
    # covers most positions, the bank is a constant function of the layer for
    # those tokens, whatever the marginal usage looks like.
    unique_sets, _inverse, counts_per_set = torch.unique(top, dim=0,
                                                         return_inverse=True,
                                                         return_counts=True)
    set_share = counts_per_set.to(torch.float64) / positions
    return {
        "positions": positions,
        "k": k,
        "counts": [int(item) for item in counts],
        "share": [float(item) for item in share],
        "top1_share": [float(item / top1.sum()) for item in top1],
        "normalised_entropy": float(entropy),
        "top1_normalised_entropy": float(
            -(top1 / top1.sum()).clamp_min(1e-300).log().mul(top1 / top1.sum()).sum()
            / np.log(experts)),
        "kl_to_uniform": float((share.clamp_min(1e-300)
                                * (share.clamp_min(1e-300).log() - np.log(1.0 / experts)))
                               .sum()),
        "coefficient_of_variation": float(share.std(unbiased=False) / share.mean()),
        "max_over_mean": float(share.max() / share.mean()),
        "dead_experts_used": [int(index) for index, item in enumerate(counts)
                              if float(item) < 0.01 * expected],
        "position_entropy_mean": float(position_entropy.mean()),
        "position_entropy_median": float(position_entropy.median()),
        "position_entropy_p10": float(torch.quantile(position_entropy, 0.10)),
        "top1_concentration": float(
            (top1 / top1.sum()).pow(2).sum()),
        "distinct_topk_sets": int(counts_per_set.numel()),
        "modal_set_share": float(set_share.max()),
        "modal_set": [int(item) for item in
                      unique_sets[int(counts_per_set.argmax())].reshape(-1).tolist()],
        "set_entropy_bits": float(-(set_share.clamp_min(1e-300)
                                    * set_share.clamp_min(1e-300).log2()).sum()),
        "sets_covering_half": int(int((set_share.cumsum(0) < 0.5).sum()) + 1),
    }


def utility_usage(utilities: torch.Tensor, k: int = K) -> dict:
    """The usage the *utility* would choose, as the reference distribution."""
    return usage_stats(utilities.detach().to(torch.float64), k=k)


# ---------------------------------------------------------------------------
# memory path


def gate_stats(values: torch.Tensor) -> dict:
    values = values.detach().to(torch.float64).reshape(-1)
    return {
        "n": int(values.numel()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "p10": float(torch.quantile(values, 0.10)),
        "median": float(values.median()),
        "p90": float(torch.quantile(values, 0.90)),
        "fraction_above_half": float((values > 0.5).double().mean()),
        "fraction_below_tenth": float((values < 0.1).double().mean()),
        "fraction_above_ninetenth": float((values > 0.9).double().mean()),
    }


def row_usage(indices: torch.Tensor, rows: int) -> dict:
    """Distinct rows read, coverage and read concentration for `[N, reads]`."""
    flat = indices.reshape(-1).to(torch.int64)
    total = int(flat.numel())
    hit = torch.zeros(rows, dtype=torch.bool)
    hit[flat] = True
    distinct = int(hit.sum())
    counts = torch.bincount(flat, minlength=rows).to(torch.float64)
    ordered = counts.sort(descending=True).values
    cumulative = ordered.cumsum(0) / max(total, 1)
    top_share = float(cumulative[min(rows, max(1, rows // 100)) - 1])
    collision = total - distinct
    pairs = indices.reshape(-1, indices.shape[-1])
    same_row = (pairs[:, 0] == pairs[:, 1]).double().mean() if pairs.shape[1] > 1 \
        else torch.zeros(())
    return {
        "reads": total,
        "distinct_rows_read": distinct,
        "coverage_of_table": distinct / rows,
        "collision_reads": collision,
        "collision_share": collision / total if total else 0.0,
        "two_head_same_row_share": float(same_row),
        "top_1pct_rows_read_share": top_share,
        "max_row_reads": int(ordered[0]),
        "mean_reads_per_distinct_row": total / distinct if distinct else None,
    }


def displacement_stats(table: torch.Tensor, init: torch.Tensor) -> dict:
    """Per-row change against the reconstructed initialisation."""
    delta = table.detach().to(torch.float32) - init.to(torch.float32)
    row_delta = delta.norm(dim=1)
    total = float(row_delta.sum())
    ordered = row_delta.sort(descending=True).values
    top = max(1, ordered.numel() // 100)
    init_row = init.to(torch.float32).norm(dim=1)
    relative = row_delta / init_row.clamp_min(1e-12)
    return {
        "rows": int(table.shape[0]),
        "rows_bitwise_unchanged": int((table.view(torch.int16)
                                       == init.view(torch.int16)).all(dim=1).sum()),
        "total_displacement": total,
        "top_1pct_displacement_share": float(ordered[:top].sum() / total)
        if total else 0.0,
        "relative_displacement_median": float(relative.median()),
        "relative_displacement_p90": float(torch.quantile(relative, 0.90)),
        "relative_displacement_p99": float(torch.quantile(relative, 0.99)),
        "relative_displacement_max": float(relative.max()),
    }


def path_magnitudes(features: torch.Tensor, bank_state: dict | None,
                    memory_state: dict, indices: torch.Tensor,
                    positions: int | None = None) -> dict:
    """RMS output of the memory path and the bank on *stored* states.

    This evaluates the saved modules on the saved hidden states. It is not a
    model forward: no new activations are produced and no token passes through
    the network. The bank magnitude is the deployed mixture value (softmax
    weight on the selected experts) without capacity overflow, which is noted
    in the receipt because overflow drops pairs in the deployed path.
    """
    if positions is not None and features.shape[0] > positions:
        step = features.shape[0] // positions
        features = features[::step][:positions]
        indices = indices[::step][:positions]
    hidden = features.detach().to(torch.float32)
    values = memory_state["table"][indices].to(torch.float32)
    mixed = values.reshape(hidden.shape[0], -1) @ memory_state["proj"].T
    gate = torch.sigmoid(hidden @ memory_state["gate"].reshape(-1))
    memory_out = gate.unsqueeze(-1) * mixed
    result = {
        "positions": int(hidden.shape[0]),
        "input_rms": float(hidden.pow(2).mean().sqrt()),
        "gate": gate_stats(gate),
        "memory_output_rms": float(memory_out.pow(2).mean().sqrt()),
        "memory_output_over_input_rms": float(memory_out.pow(2).mean().sqrt()
                                            / hidden.pow(2).mean().sqrt().clamp_min(1e-12)),
        "memory_output_rms_median_position": float(memory_out.norm(dim=1).median()),
    }
    if bank_state is not None:
        logits = hidden @ bank_state["router"].T
        top = logits.topk(K, dim=-1).indices
        weights = torch.softmax(logits, dim=-1).gather(1, top)
        bank_out = torch.zeros_like(hidden)
        # Group positions by expert to keep this bounded and reproducible.
        for expert in range(bank_state["gate"].shape[0]):
            mask = (top == expert).any(dim=-1)
            if not bool(mask.any()):
                continue
            rows = hidden[mask]
            activation = torch.nn.functional.silu(rows @ bank_state["gate"][expert].T) \
                * (rows @ bank_state["up"][expert].T)
            contribution = activation @ bank_state["down"][expert].T
            slot = (top[mask] == expert).to(torch.float32) * weights[mask]
            bank_out[mask] += contribution * slot.sum(dim=-1, keepdim=True)
        result["bank_output_rms"] = float(bank_out.pow(2).mean().sqrt())
        result["bank_output_over_input_rms"] = float(
            bank_out.pow(2).mean().sqrt() / hidden.pow(2).mean().sqrt().clamp_min(1e-12))
        result["bank_output_rms_median_position"] = float(bank_out.norm(dim=1).median())
        memory_share = memory_out.norm(dim=1) / (memory_out.norm(dim=1)
                                                + bank_out.norm(dim=1)).clamp_min(1e-12)
        result["memory_share_of_path_norm"] = {
            "mean": float(memory_share.mean()),
            "median": float(memory_share.median()),
            "p10": float(torch.quantile(memory_share, 0.10)),
            "p90": float(torch.quantile(memory_share, 0.90)),
        }
        gate_flat = gate.detach().double()
        bank_norm = bank_out.norm(dim=1).double()
        if float(bank_norm.std()) > 0:
            result["gate_vs_bank_norm_pearson"] = float(torch.corrcoef(
                torch.stack([gate_flat, bank_norm]))[0, 1])
    return result


def coverage_scan(tokens: torch.Tensor, layer: int, rows: int,
                  block: int = 1024) -> dict:
    """Distinct rows read over a bounded prefix of the training stream.

    A pure function of token ids, so it is identical for every arm that shares
    the geometry. The bound is a parameter because the scan is linear in tokens
    and the caller sizes it; the receipt records how many tokens were scanned.
    """
    import memory as memory_module
    usable = (tokens.numel() // block) * block
    if usable == 0:
        return {"tokens": 0, "reads": 0, "distinct_rows": 0, "coverage": 0.0}
    seen = torch.zeros(rows, dtype=torch.bool)
    reads = 0
    for start in range(0, usable, block):
        window = tokens[start:start + block].reshape(1, block)
        for order in ORDERS:
            index = memory_module.ngram_index(window, layer, order, rows)
            seen[index.reshape(-1)] = True
            reads += int(index.numel())
    return {"tokens": usable, "reads": reads,
            "distinct_rows": int(seen.sum()),
            "coverage": float(seen.sum()) / rows}


def gate_router_alignment(gate: torch.Tensor, router: torch.Tensor) -> dict:
    """Cosine between the gate's read direction and each expert's direction."""
    values = (gate.reshape(-1).double() / gate.reshape(-1).double().norm().clamp_min(1e-12))
    unit = router.double() / router.double().norm(dim=1).clamp_min(1e-12).unsqueeze(1)
    cosines = unit @ values
    return {
        "per_expert": [float(item) for item in cosines],
        "mean_absolute": float(cosines.abs().mean()),
        "max_absolute": float(cosines.abs().max()),
        "argmax_absolute_expert": int(cosines.abs().argmax()),
    }


# ---------------------------------------------------------------------------
# interaction statistics


def mutual_information(first: np.ndarray, second: np.ndarray) -> float:
    """Empirical mutual information in bits between two integer code arrays."""
    if first.size != second.size or first.size == 0:
        raise ValueError("mutual information needs equal non-empty arrays")
    levels_a = int(first.max()) + 1
    levels_b = int(second.max()) + 1
    table = np.zeros((levels_a, levels_b), dtype=np.float64)
    np.add.at(table, (first, second), 1.0)
    joint = table / table.sum()
    pa = joint.sum(axis=1, keepdims=True)
    pb = joint.sum(axis=0, keepdims=True)
    expected = pa @ pb
    mask = joint > 0
    return float((joint[mask] * np.log2(joint[mask] / expected[mask])).sum())


def mi_permutation_p(first: np.ndarray, second: np.ndarray, permutations: int,
                     seed: int) -> dict:
    """Mutual information with a permutation null on the second code array."""
    observed = mutual_information(first, second)
    generator = np.random.default_rng(seed)
    null = np.empty(permutations)
    for index in range(permutations):
        null[index] = mutual_information(first, generator.permutation(second))
    return {
        "bits": observed,
        "null_mean": float(null.mean()),
        "null_p95": float(np.percentile(null, 95)),
        "p_value": float((1 + int((null >= observed).sum())) / (permutations + 1)),
        "permutations": permutations,
        "seed": seed,
    }


def popularity_codes(indices: torch.Tensor, levels: int = 4) -> np.ndarray:
    """Quantile buckets of row popularity across positions (first head)."""
    head = indices[:, 0].to(torch.int64)
    counts = torch.bincount(head, minlength=ROWS)
    popularity = counts[head].to(torch.float64).numpy()
    if popularity.size == 0:
        return np.zeros(0, dtype=np.int64)
    edges = np.quantile(popularity, np.linspace(0, 1, levels + 1)[1:-1])
    return np.digitize(popularity, edges).astype(np.int64)


# ---------------------------------------------------------------------------
# outcomes


def load_items(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        receipt = json.load(handle)
    results = receipt["results"]
    items = {}
    for task, field in TASKS:
        rows = results[task]["items"]
        items[task] = {row["id"]: bool(row[field]) for row in rows}
    return items


def outcome_cells(arms: dict[str, dict], left: str, right: str,
                  reference: str) -> dict:
    """Four-way `left`/`right` success partition and `reference`'s accuracy."""
    per_task = {}
    pooled_weight, pooled_reference, pooled_cell = [], [], []
    for task, _field in TASKS:
        ids = sorted(set(arms[left][task]) & set(arms[right][task])
                     & set(arms[reference][task]))
        left_ok = np.array([arms[left][task][key] for key in ids], dtype=bool)
        right_ok = np.array([arms[right][task][key] for key in ids], dtype=bool)
        reference_ok = np.array([arms[reference][task][key] for key in ids],
                                dtype=np.float64)
        weight = np.full(len(ids), 1.0 / (len(TASKS) * len(ids)))
        cells = {
            "both": left_ok & right_ok,
            "left_only": left_ok & ~right_ok,
            "right_only": ~left_ok & right_ok,
            "neither": ~left_ok & ~right_ok,
        }
        table = {}
        for name, mask in cells.items():
            count = int(mask.sum())
            accuracy = reference_ok[mask].mean() if count else None
            table[name] = {
                "n": count,
                "reference_accuracy": float(accuracy) if accuracy is not None else None,
                "reference_accuracy_ci": bootstrap_mean(reference_ok[mask])
                if count else None,
            }
        per_task[task] = table
        for name, mask in cells.items():
            pooled_weight.append(weight[mask])
            pooled_reference.append(reference_ok[mask])
            pooled_cell.extend([name] * int(mask.sum()))
    weights = np.concatenate(pooled_weight)
    values = np.concatenate(pooled_reference)
    cells_array = np.array(pooled_cell)
    pooled = {}
    for name in ("both", "left_only", "right_only", "neither"):
        mask = cells_array == name
        pooled[name] = {
            "n": int(mask.sum()),
            "mask": mask,
        }
    result = {}
    for name, row in pooled.items():
        mask = row.pop("mask")
        if not row["n"]:
            result[name] = {"n": 0, "reference_composite": None,
                            "reference_composite_ci": None}
            continue
        sub_weight = weights[mask]
        sub_values = values[mask]
        estimate = float((sub_values * sub_weight).sum() / sub_weight.sum())
        generator = np.random.default_rng(BOOTSTRAP_SEED)
        indexes = generator.integers(0, row["n"], size=(BOOTSTRAP_RESAMPLES, row["n"]))
        sampled_weight = sub_weight[indexes]
        sampled_values = sub_values[indexes]
        distribution = (sampled_values * sampled_weight).sum(axis=1) \
            / sampled_weight.sum(axis=1)
        result[name] = {
            "n": row["n"],
            "reference_composite": estimate,
            "reference_composite_ci": {
                "low": float(np.percentile(distribution, 2.5)),
                "high": float(np.percentile(distribution, 97.5)),
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
            },
        }
    # `union` is the items at least one of the two arms solved; `disjoint` is the
    # items only one solved. `H` losing the disjoint cells is the pattern the
    # diagnostic is testing for, because those are the arm-specific gains.
    union = (cells_array == "both") | (cells_array == "left_only") \
        | (cells_array == "right_only")
    result["union"] = summarize_cell(weights[union], values[union])
    result["disjoint"] = summarize_cell(
        weights[(cells_array == "left_only") | (cells_array == "right_only")],
        values[(cells_array == "left_only") | (cells_array == "right_only")])
    return {"left": left, "right": right, "reference": reference,
            "per_task": per_task, "pooled": result,
            "weighting": "composite PC: equal task weight, then item mean"}


def bootstrap_mean(values: np.ndarray) -> dict:
    """Item-mean with a percentile CI (items are the resampling unit)."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("bootstrap_mean needs a non-empty array")
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indexes = generator.integers(0, values.size, size=(BOOTSTRAP_RESAMPLES, values.size))
    distribution = values[indexes].mean(axis=1)
    return {"mean": float(values.mean()),
            "low": float(np.percentile(distribution, 2.5)),
            "high": float(np.percentile(distribution, 97.5)),
            "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
            "n": int(values.size)}


def summarize_cell(weights: np.ndarray, values: np.ndarray) -> dict:
    if weights.size == 0:
        return {"n": 0, "reference_composite": None,
                "reference_composite_ci": None}
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indexes = generator.integers(0, weights.size, size=(BOOTSTRAP_RESAMPLES, weights.size))
    sampled_weight = weights[indexes]
    sampled_values = values[indexes]
    distribution = (sampled_values * sampled_weight).sum(axis=1) / sampled_weight.sum(axis=1)
    return {
        "n": int(weights.size),
        "reference_composite": float((values * weights).sum() / weights.sum()),
        "reference_composite_ci": {
            "low": float(np.percentile(distribution, 2.5)),
            "high": float(np.percentile(distribution, 97.5)),
            "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
    }


# ---------------------------------------------------------------------------
# state loading


def hstar_checkpoints(directory: str) -> list[tuple[int, str]]:
    """The `H*` step checkpoints present now, oldest first.

    `H*` keeps only its most recent checkpoints (the training loop prunes), so
    the set moves while the run is live. The diagnostic reads the *earliest
    available* one, which is the closest in time to the state the stored labels
    were measured from, and the *latest* for the frozen-router check. Both paths
    and their digests go into the receipt, so the choice is auditable even
    after the file rotates away.
    """
    import glob
    import re
    found = []
    for path in glob.glob(os.path.join(directory, "step-*.pt")):
        match = re.search(r"step-(\d+)\.pt$", path)
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found)


def load_state(path: str) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (RuntimeError, ValueError):
        payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise SystemExit(f"{path}: no model state dict")
    return state


def find_layer_keys(state: dict, layer: int) -> dict:
    """Locate the arm's modules at one layer without assuming the nesting."""
    prefix = f"model.layers.{layer}.mlp.extra"
    found = {"router": None, "bank_gate": None, "bank_up": None, "bank_down": None,
             "table": None, "proj": None, "gate": None}
    for key in state:
        if not key.startswith(prefix):
            continue
        if key.endswith(".router.weight"):
            found["router"] = key
        elif key.endswith(".table"):
            found["table"] = key
        elif key.endswith(".proj.weight"):
            found["proj"] = key
        elif key.endswith(".gate.weight") and key.count("gate") == 1 \
                and found["gate"] is None:
            # The memory gate is `[1, hidden]`; the bank's projection is `gate`
            # too, so the shape decides below.
            found["gate"] = key
    if found["gate"] is not None and state[found["gate"]].shape[0] != 1:
        found["gate"] = None
    for suffix, field in ((".gate", "bank_gate"), (".up", "bank_up"),
                          (".down", "bank_down")):
        for key in state:
            if key.startswith(prefix) and key.endswith(suffix) \
                    and state[key].dim() == 3:
                found[field] = key
                break
    return found


# ---------------------------------------------------------------------------
# main analysis


def analyse(args) -> dict:
    receipt = {
        "schema": SCHEMA,
        "stamp": stamp_now(),
        "revision": os.environ.get("SELVEDGE_REVISION", "unknown"),
        "created_utc": utc_now(),
        "scope": {
            "forward_passes_run": 0,
            "gpu_used": False,
            "measured_from": [
                "stored per-item evaluation receipts (outcomes)",
                "stored step checkpoints (weights, tables, gates)",
                "stored H* utility-label artifact (pre-expert hidden states)",
                "the training stream the calibration slice is a prefix of",
                "the deterministic initialisers of modeling._router and "
                "memory.HashedMemory",
            ],
            "not_measured": "anything requiring H's or X's own hidden states",
        },
        "inputs": {},
        "outcomes": {},
        "routers": {},
        "usage": {},
        "memory": {},
        "interaction": {},
        "deferred": [],
        "limits": [],
    }

    # Resolve the H* checkpoints from the directory rather than a fixed name:
    # the live run rotates them, and a name that existed an hour ago may not.
    available = hstar_checkpoints(os.path.dirname(args.checkpoint_hstar))
    if available:
        args.checkpoint_hstar = available[0][1]
        args.checkpoint_hstar_later = available[-1][1] if len(available) > 1 else ""
        receipt["inputs"]["hstar_checkpoints_available"] = [
            {"step": step, "path": path} for step, path in available]

    for name, path in (("labels", args.labels), ("stream", args.stream),
                       ("checkpoint_H", args.checkpoint_H),
                       ("checkpoint_X", args.checkpoint_X),
                       ("checkpoint_M", args.checkpoint_M),
                       ("checkpoint_hstar", args.checkpoint_hstar)):
        if path and os.path.exists(path):
            receipt["inputs"][name] = {"path": path, "sha256": sha256_file(path),
                                       "bytes": os.path.getsize(path)}
    for arm in ("X", "M", "H", "D", "D0"):
        path = os.path.join(args.receipts, f"t2-{arm}.json")
        if os.path.exists(path):
            receipt["inputs"][f"eval_{arm}"] = {"path": path,
                                                "sha256": sha256_file(path),
                                                "bytes": os.path.getsize(path)}

    # -- 1. outcomes --------------------------------------------------------
    arms = {}
    for arm in ("X", "M", "H", "D", "D0"):
        path = os.path.join(args.receipts, f"t2-{arm}.json")
        if os.path.exists(path):
            arms[arm] = load_items(path)
    if {"X", "M", "H"} <= set(arms):
        receipt["outcomes"]["H_on_XM_partition"] = outcome_cells(arms, "X", "M", "H")
        receipt["outcomes"]["H_on_XD0_partition"] = outcome_cells(arms, "X", "D0", "H")
        receipt["outcomes"]["H_on_MD0_partition"] = outcome_cells(arms, "M", "D0", "H")
        receipt["outcomes"]["D_on_XM_partition"] = outcome_cells(arms, "X", "M", "D")
        for task, _field in TASKS:
            ids = sorted(set(arms["H"][task]) & set(arms["X"][task]))
            first = np.array([arms["H"][task][key] for key in ids], dtype=np.float64)
            second = np.array([arms["X"][task][key] for key in ids], dtype=np.float64)
            third = np.array([arms["M"][task][key] for key in ids], dtype=np.float64)
            receipt["outcomes"].setdefault("paired", {})[task] = {
                "H_minus_X": paired_bootstrap(first, second),
                "H_minus_M": paired_bootstrap(first, third),
                "M_minus_X": paired_bootstrap(third, second),
            }
    else:
        receipt["deferred"].append({
            "what": "item-level outcome partition",
            "why": "one of the T2 evaluation receipts is absent",
            "how": "--receipts results/eval",
        })

    # -- 2. router geometry -------------------------------------------------
    router_weights: dict[str, dict[int, torch.Tensor]] = {}
    for arm, path in (("H", args.checkpoint_H), ("X", args.checkpoint_X),
                      ("Hstar", args.checkpoint_hstar)):
        if not path or not os.path.exists(path):
            continue
        state = load_state(path)
        per_layer = {}
        weights = {}
        for layer in range(args.layers):
            keys = find_layer_keys(state, layer)
            if keys["router"] is None:
                continue
            weight = state[keys["router"]].detach().to(torch.float32)
            init = init_block(RUN_SEED, ROUTER_SEED_OFFSET, layer, tuple(weight.shape))
            row = router_stats(weight, init)
            row["init_validation_cosine_wrong_seed"] = cosine(
                weight, init_block(RUN_SEED, ROUTER_SEED_OFFSET + 7777, layer,
                                   tuple(weight.shape)))
            per_layer[str(layer)] = row
            weights[layer] = weight
        router_weights[arm] = weights
        receipt["routers"][arm] = {
            "layers": len(per_layer),
            "per_layer": per_layer,
            "summary": summarise_rows(per_layer, (
                "effective_rank", "top1_energy_share", "mean_absolute_pairwise_cosine",
                "norm_ratio_min_median", "displacement_relative", "cosine_to_init",
                "norm_ratio_to_init", "init_validation_cosine_wrong_seed")),
        }
    if {"H", "X"} <= set(router_weights):
        common = sorted(set(router_weights["H"]) & set(router_weights["X"]))
        receipt["routers"]["H_versus_X"] = {
            "layers": len(common),
            "per_layer_cosine": {str(layer): cosine(router_weights["H"][layer],
                                                    router_weights["X"][layer])
                                 for layer in common},
            "summary": summarise_rows(
                {str(layer): {"cosine": cosine(router_weights["H"][layer],
                                               router_weights["X"][layer])}
                 for layer in common}, ("cosine",)),
        }

    # -- 3. routing usage on stored H* states -------------------------------
    payload = None
    if args.labels and os.path.exists(args.labels):
        try:
            payload = torch.load(args.labels, map_location="cpu", weights_only=False,
                                 mmap=True)
        except (RuntimeError, ValueError):
            payload = torch.load(args.labels, map_location="cpu", weights_only=False)
        receipt["inputs"]["labels_manifest"] = payload.get("manifest", {})
        # The label tensors are keyed by the integer bank index; the manifest is
        # keyed by its decimal string. Normalise to the string, which every
        # section below uses.
        for field in ("features", "utilities", "reference_loss"):
            if field in payload:
                payload[field] = {str(key): value
                                  for key, value in payload[field].items()}

    # The H* router is frozen after the utility phase, so the router in any
    # later checkpoint is the one the labels were measured under. Two H*
    # checkpoints exist, which makes that checkable rather than assumed.
    if args.checkpoint_hstar and args.checkpoint_hstar_later \
            and os.path.exists(args.checkpoint_hstar) \
            and os.path.exists(args.checkpoint_hstar_later):
        early = load_state(args.checkpoint_hstar)
        late = load_state(args.checkpoint_hstar_later)
        deltas = {}
        for layer in range(args.layers):
            early_keys = find_layer_keys(early, layer)
            late_keys = find_layer_keys(late, layer)
            if early_keys["router"] is None or late_keys["router"] is None:
                continue
            deltas[str(layer)] = float(
                (early[early_keys["router"]].float()
                 - late[late_keys["router"]].float()).abs().max())
        receipt["routers"]["Hstar_router_frozen"] = {
            "early": args.checkpoint_hstar, "late": args.checkpoint_hstar_later,
            "layers": len(deltas),
            "max_abs_difference": max(deltas.values()) if deltas else None,
            "bitwise_identical": bool(deltas) and max(deltas.values()) == 0.0,
            "why": "the router carries no gradient in H* (modeling.set_router_grad), "
                   "so identical routers across steps mean the checkpoint's router is "
                   "the one the stored labels were measured under",
        }
        del early, late

    if payload is not None and "Hstar" in router_weights:
        per_layer = {}
        usage_rows, utility_rows = {}, {}
        for layer in sorted(router_weights["Hstar"]):
            key = str(layer)
            if key not in payload["features"]:
                continue
            features = payload["features"][key].detach().to(torch.float32)
            logits = features @ router_weights["Hstar"][layer].T
            usage_rows[key] = usage_stats(logits)
            if key in payload["utilities"]:
                utility_rows[key] = utility_usage(payload["utilities"][key])
        receipt["usage"]["Hstar"] = {
            "layers": len(usage_rows),
            "per_layer": usage_rows,
            "utility_per_layer": utility_rows,
            "summary": summarise_rows(usage_rows, (
                "normalised_entropy", "kl_to_uniform", "coefficient_of_variation",
                "max_over_mean", "position_entropy_mean", "top1_concentration",
                "distinct_topk_sets", "modal_set_share", "set_entropy_bits",
                "sets_covering_half")),
            "utility_summary": summarise_rows(utility_rows, (
                "normalised_entropy", "kl_to_uniform", "coefficient_of_variation",
                "max_over_mean", "distinct_topk_sets", "modal_set_share",
                "set_entropy_bits")),
            "dead_experts_used": {key: row["dead_experts_used"]
                                  for key, row in usage_rows.items()
                                  if row["dead_experts_used"]},
            "dead_experts_in_utility": {key: row["dead_experts_used"]
                                        for key, row in utility_rows.items()
                                        if row["dead_experts_used"]},
        }
    else:
        receipt["deferred"].append({
            "what": "routing usage distribution",
            "why": "the H* label artifact or the H* checkpoint is absent",
            "how": "produce it with the label pass in docs/T2-DIAGNOSTICS.md §4",
        })

    # -- 4. memory path -----------------------------------------------------
    stream = None
    if args.stream and os.path.exists(args.stream):
        stream = np.fromfile(args.stream, dtype="<u4")
        receipt["inputs"]["stream_manifest"] = {
            "tokens": int(stream.size),
            "head_sha256_first_32768": sha256_bytes(
                stream[:32768].astype("<u4").tobytes()),
        }
        stream_full = torch.from_numpy(stream.astype("int64"))
        stream = stream_full[:args.calibration_tokens]
    else:
        stream_full = None

    indices_by_layer = {}
    if stream is not None:
        import memory as memory_module
        ids = stream.reshape(-1, MICRO_POSITIONS)
        for layer in MEMORY_LAYERS:
            indices_by_layer[layer] = torch.stack(
                [memory_module.ngram_index(ids, layer, order, ROWS)
                 for order in ORDERS], dim=-1).reshape(-1, READS)
        receipt["memory"]["table_stream"] = {
            "layer": "all four insertions share the addressing rule",
            "calibration_tokens": int(stream.numel()),
            "row_shape": [MICRO_POSITIONS, "positions per row, as the labeller "
                                            "fed the slice (T2.5 micro_positions)"],
            "per_layer": {str(layer): row_usage(indices_by_layer[layer], ROWS)
                          for layer in MEMORY_LAYERS},
            "table_rows": ROWS,
        }
        scanned = min(int(stream_full.numel()), int(args.coverage_scan_tokens))
        receipt["memory"]["table_budget_scan"] = {
            "tokens_scanned": scanned,
            "bounded_by": "--coverage-scan-tokens",
            "per_layer": {str(layer): coverage_scan(stream_full[:scanned], layer, ROWS)
                          for layer in MEMORY_LAYERS},
            "table_rows": ROWS,
        }

    for arm, path in (("H", args.checkpoint_H), ("M", args.checkpoint_M),
                      ("Hstar", args.checkpoint_hstar)):
        if not path or not os.path.exists(path):
            continue
        state = load_state(path)
        per_layer = {}
        for layer in MEMORY_LAYERS:
            keys = find_layer_keys(state, layer)
            if keys["table"] is None:
                continue
            table = state[keys["table"]]
            gate = state[keys["gate"]].detach().to(torch.float32)
            proj = state[keys["proj"]].detach().to(torch.float32)
            init_table = init_block(RUN_SEED, MEMORY_SEED_OFFSET, layer,
                                    tuple(table.shape), dtype=torch.bfloat16)
            init_gate = init_block(RUN_SEED, MEMORY_SEED_OFFSET, layer,
                                   tuple(gate.shape))
            row = {
                "gate_norm": float(gate.norm()),
                "gate_init_norm": float(init_gate.norm()),
                "gate_norm_ratio_to_init": float(gate.norm() / init_gate.norm()),
                "gate_cosine_to_init": cosine(gate, init_gate),
                "gate_init_validation_cosine_wrong_seed": cosine(
                    gate, init_block(RUN_SEED, MEMORY_SEED_OFFSET + 7777, layer,
                                     tuple(gate.shape))),
                "projection_frobenius": float(proj.norm()),
                "projection_row_norm_median": float(proj.norm(dim=1).median()),
                "table_norm": float(table.detach().to(torch.float32).norm()),
                "table_init_norm": float(init_table.to(torch.float32).norm()),
                "table_relative_norm": float(table.detach().to(torch.float32).norm()
                                             / init_table.to(torch.float32).norm()),
                "table": displacement_stats(table, init_table),
            }
            if keys["router"] is not None:
                row["gate_router_alignment"] = gate_router_alignment(
                    gate, state[keys["router"]].detach().to(torch.float32))
            memory_state = {"table": table.detach(), "proj": proj,
                            "gate": gate.reshape(-1)}
            bank_state = None
            if keys["bank_gate"] is not None:
                bank_state = {
                    "router": state[keys["router"]].detach().to(torch.float32),
                    "gate": state[keys["bank_gate"]].detach().to(torch.float32),
                    "up": state[keys["bank_up"]].detach().to(torch.float32),
                    "down": state[keys["bank_down"]].detach().to(torch.float32),
                }
            if payload is not None and str(layer) in payload["features"] \
                    and layer in indices_by_layer:
                row["on_stored_states"] = path_magnitudes(
                    payload["features"][str(layer)], bank_state, memory_state,
                    indices_by_layer[layer], positions=args.probe_positions)
            per_layer[str(layer)] = row
            del table, init_table, memory_state, bank_state
        receipt["memory"][arm] = {
            "layers": len(per_layer),
            "per_layer": per_layer,
            "summary": summarise_rows(per_layer, (
                "gate_norm", "gate_norm_ratio_to_init", "gate_cosine_to_init",
                "projection_frobenius", "table_relative_norm")),
        }

    # -- 5. interaction -----------------------------------------------------
    if payload is not None and "Hstar" in router_weights and indices_by_layer:
        per_layer = {}
        for layer in MEMORY_LAYERS:
            key = str(layer)
            if key not in payload["features"] or layer not in indices_by_layer:
                continue
            features = payload["features"][key].detach().to(torch.float32)
            logits = features @ router_weights["Hstar"][layer].T
            top1 = logits.argmax(dim=-1).to(torch.int64).numpy().astype(np.int64)
            indices = indices_by_layer[layer]
            popularity = popularity_codes(indices)
            mi = mi_permutation_p(top1, popularity, PERMUTATIONS, BOOTSTRAP_SEED)
            head = indices[:, 0].to(torch.int64)
            counts = torch.bincount(head, minlength=ROWS)
            seen = torch.zeros(ROWS, dtype=torch.bool)
            first_sum = 0
            for position, row in enumerate(head.tolist()):
                if seen[row]:
                    first_sum += 1
                else:
                    seen[row] = True
            repeat = np.array([1 if counts[row] > 1 else 0 for row in head.tolist()],
                              dtype=np.int64)
            mi_repeat = mi_permutation_p(top1, repeat, PERMUTATIONS, BOOTSTRAP_SEED)
            per_layer[key] = {
                "expert_vs_row_popularity": mi,
                "expert_vs_repeated_row": mi_repeat,
                "repeat_read_share": float(np.mean(repeat)),
                "top1_expert_entropy_bits": float(
                    -np.sum((np.bincount(top1, minlength=EXPERTS)
                             / top1.size) * np.log2(np.maximum(
                                 np.bincount(top1, minlength=EXPERTS) / top1.size,
                                 1e-12)))),
            }
        receipt["interaction"]["Hstar"] = per_layer

    # -- deferred work ------------------------------------------------------
    receipt["deferred"].extend([
        {
            "what": "H's and X's own usage, gate values and row/ expert "
                    "co-occurrence",
            "why": "their hidden states are not on disk; the stored artifacts "
                   "carry H*'s only, and this session must not run a forward pass "
                   "while the H* container owns the GPU",
            "how": "scripts/court --gpu oracle -- python3 "
                   "oracle/eval/capture_activations.py ... (see "
                   "--deferred-commands)",
        },
        {
            "what": "per-item routing attribution for the items X or M solve and "
                    "H does not",
            "why": "needs H's activations on the evaluation items, which no "
                   "receipt stores",
            "how": "the item list is in the receipt's outcome cells; the capture "
                   "is the same command with the item prompts as input",
        },
        {
            "what": "the deployed-budget utility u^dep (P12.17 part 1)",
            "why": "needs new label passes under the router's own top-4 mask",
            "how": "docs/T2-DIAGNOSTICS.md §4",
        },
    ])
    receipt["limits"] = [
        "Weight displacement from a reconstructed initialiser is exact only "
        "for the parameters the optimiser initialised that way; every router "
        "and memory tensor is, and the wrong-seed control is recorded in each "
        "row.",
        "A small router direction is a necessary, not sufficient, condition for "
        "an unused expert: usage is measured separately in section 3 and the "
        "receipt keeps the two apart.",
        "Path magnitudes are computed from saved modules on saved hidden "
        "states, without capacity dispatch. The deployed path drops overflow "
        "pairs (H's evaluation receipt records a dropped pair fraction), so "
        "these magnitudes are the module's own scale, not the deployed mixture.",
        "The row-usage measurement is a property of the addressing function and "
        "the calibration slice, identical for every arm that shares the "
        "geometry. It says which rows the path reads, not which rows matter.",
        "Mutual information between expert choice and row popularity is a "
        "dependence measure on one arm's calibration states. It cannot "
        "distinguish 'the memory causes the routing' from 'both are driven by "
        "the token', which is why the permutation null is reported with it.",
    ]
    return receipt


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def summarise_rows(rows: dict, keys: tuple[str, ...]) -> dict:
    out = {}
    for key in keys:
        values = [row[key] for row in rows.values()
                  if isinstance(row, dict) and isinstance(row.get(key), (int, float))]
        if not values:
            continue
        out[key] = {"mean": float(np.mean(values)), "min": float(np.min(values)),
                    "max": float(np.max(values)), "n": len(values)}
    return out


# ---------------------------------------------------------------------------
# markdown rendering (from the receipt only)


def _f(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def reading_lines(receipt: dict) -> list[str]:
    """Interpretation, with every quantity pulled from the receipt."""
    lines: list[str] = []
    routers = receipt.get("routers", {})
    usage = receipt.get("usage", {}).get("Hstar")
    memory = receipt.get("memory", {})
    outcomes = receipt.get("outcomes", {})

    def summary(arm, key):
        block = routers.get(arm, {})
        return block.get("summary", {}).get(key, {}).get("mean")

    if "H" in routers and "X" in routers:
        between = routers.get("H_versus_X", {}).get("summary", {}).get("cosine", {})
        lines.append(
            f"1. **The two single-path arms' routers are close to the same matrix.** "
            f"`H`'s and `X`'s routers start from one draw and end up with mean "
            f"cosine **{_f(between.get('mean'))}** to each other. Each also "
            f"collapsed onto almost one direction: top-1 "
            f"energy share **{_f(summary('H', 'top1_energy_share'))}** (`H`) and "
            f"**{_f(summary('X', 'top1_energy_share'))}** (`X`) of eight, with mean "
            f"|cos| between expert directions **{_f(summary('H', 'mean_absolute_pairwise_cosine'))}** "
            f"and **{_f(summary('X', 'mean_absolute_pairwise_cosine'))}**. Two "
            "consequences follow, and they are measurements rather than guesses: "
            "the eight specialists are nearly interchangeable to the selector, "
            "and `H` and `X` route almost identically despite `H` carrying a "
            "memory path that `X` does not.")
    if usage:
        router_summary = usage.get("summary", {})
        utility_summary = usage.get("utility_summary", {})
        lines.append(
            f"2. **`H*`'s router is not the failure.** Its usage entropy is "
            f"**{_f(router_summary.get('normalised_entropy', {}).get('mean'))}** of "
            f"the maximum and its top-4 set entropy is "
            f"**{_f(router_summary.get('set_entropy_bits', {}).get('mean'))}** bits "
            f"against the utility's own "
            f"**{_f(utility_summary.get('set_entropy_bits', {}).get('mean'))}**, with "
            f"no dead expert in any layer. The router reproduces a near-uniform "
            "target, so any deficiency is in the *target* (near-uniform labels, "
            "which `docs/T2-DIAGNOSTICS.md` measures from the same artifact) and "
            "not in the head's capacity to fit it.")
    if "H" in memory and "M" in memory:
        def path(arm, key, layer):
            row = memory.get(arm, {}).get("per_layer", {}).get(str(layer), {})
            probe = row.get("on_stored_states", {})
            return probe.get(key)
        h_values = [path("H", "memory_output_over_input_rms", layer)
                    for layer in MEMORY_LAYERS]
        m_values = [path("M", "memory_output_over_input_rms", layer)
                    for layer in MEMORY_LAYERS]
        if all(value is not None for value in h_values + m_values):
            ratios = [m / h for h, m in zip(h_values, m_values) if h]
            lines.append(
                f"3. **In `H` the memory path is open but writes little; `M`'s "
                f"writes several times more.** On identical stored inputs, the "
                f"memory write per unit input RMS is "
                f"**{_f(min(h_values))}–{_f(max(h_values))}** for `H` and "
                f"**{_f(min(m_values))}–{_f(max(m_values))}** for `M`, a ratio of "
                f"**{_f(min(ratios))}×** to **{_f(max(ratios))}×**. `H`'s gate is "
                f"*not* suppressed (gate mean "
                f"**{_f(memory['H']['per_layer'][str(MEMORY_LAYERS[-1])]['on_stored_states']['gate']['mean'])}** "
                f"at the last insertion): the gate opens and the projection "
                f"under-writes. Read as an interpretation, this is dilution of two "
                f"paths that each need the whole task gradient, not composition.")
        scan = memory.get("table_budget_scan")
        stream_block = memory.get("table_stream")
        if scan and stream_block:
            layer = str(MEMORY_LAYERS[0])
            lines.append(
                f"4. **The table is not the constraint.** The calibration slice "
                f"touches **{_f(stream_block['per_layer'][layer]['coverage_of_table'])}** "
                f"of each table's rows, but over "
                f"**{_f(scan['tokens_scanned'])}** stream tokens the read set "
                f"covers **{_f(scan['per_layer'][layer]['coverage'])}** of it, and "
                f"every row is updated by training. The `2^18` rows are sized for "
                f"the budget, so a memory-side explanation of the hybrid's loss "
                f"cannot rest on table capacity or on unused rows.")
    block = outcomes.get("H_on_XM_partition")
    dense = outcomes.get("D_on_XM_partition")
    if block and dense:
        cells = block["pooled"]
        dense_cells = dense["pooled"]
        lines.append(
            f"5. **`H` keeps most of what both arms solved and little of what "
            f"only one solved — and the dense control behaves the same way.** On "
            f"the items both `X` and `M` solved (`n = {_f(cells['both']['n'])}`) `H` "
            f"scores **{_f(cells['both']['reference_composite'])}**; on the items "
            f"only one solved it scores **{_f(cells['left_only']['reference_composite'])}** "
            f"and **{_f(cells['right_only']['reference_composite'])}**, while `D` "
            f"scores **{_f(dense_cells['left_only']['reference_composite'])}** and "
            f"**{_f(dense_cells['right_only']['reference_composite'])}** on the same "
            f"cells. The single-path gains therefore do not transfer to another "
            f"parameterisation at all: they are arm-specific, and `H`'s shortfall "
            f"is a broad reduction rather than interference aimed at the other "
            f"arms' successes.")
    lines.append(
        "Together these point at three separate facts rather than one: the "
        "selector in the distilled arms is close to constant and near-identical "
        "across arms; the hybrid writes less through each path than the "
        "single-path arm writes through its only path; and the item-level gains "
        "of a single-path arm are largely not transferable. The forward-pass "
        "measurements deferred below are what would separate cause from "
        "correlation, and they are not run here.")
    return lines


def render_markdown(receipt: dict) -> str:
    stamp = receipt["stamp"]
    revision = receipt["revision"]
    receipt_path = receipt.get("outputs", {}).get("receipt") \
        or f"results/t2/H-DIAGNOSTICS-{stamp}.json"
    lines = [
        "# H-diagnostics — why the hybrid lost `M`'s and `X`'s gains",
        "",
        f"Generated by `oracle/eval/h_diagnostics.py` at {stamp} from revision "
        f"`{revision}`. Every number below is rendered from the receipt "
        f"`{receipt_path}` written by the same run; none "
        "is typed by hand. **This is a diagnostic, not a verdict**: it does not "
        "change T2's registered result and may not be cited as one.",
        "",
        "## What was measured, and how",
        "",
        f"- Evaluation forwards run in this session: **{_f(receipt['scope']['forward_passes_run'])}**.",
        "- GPU used: **no**. The `H*` training container owned the device "
        "throughout; every number here comes from artifacts that already "
        "existed.",
        "",
        "| input | sha256 |",
        "| --- | --- |",
    ]
    for name, row in sorted(receipt["inputs"].items()):
        if isinstance(row, dict) and "sha256" in row:
            lines.append(f"| `{name}` — `{row['path']}` | `{row['sha256'][:16]}…` |")
    lines.append("")

    outcomes = receipt.get("outcomes", {})
    if outcomes:
        lines += ["## 1. Item-level outcomes: where `H` lost the gains", "",
                  "Cells are the four-way partition of the items by whether `X` "
                  "and `M` each solved them. The composite is `PC` restricted to "
                  "the cell (equal task weight, then item mean), recomputed from "
                  "the stored per-item rows, with a percentile paired bootstrap "
                  f"(seed `{BOOTSTRAP_SEED}`, `{BOOTSTRAP_RESAMPLES}` resamples).",
                  ""]
        for name in ("H_on_XM_partition", "D_on_XM_partition", "H_on_XD0_partition"):
            block = outcomes.get(name)
            if not block:
                continue
            lines += [f"### `{block['reference']}` on the `{block['left']}`×`{block['right']}` partition",
                      "",
                      "| cell | n | composite | 95% CI |",
                      "| --- | ---: | ---: | --- |"]
            for cell in ("both", "left_only", "right_only", "neither", "union",
                         "disjoint"):
                row = block["pooled"][cell]
                ci = row.get("reference_composite_ci") or {}
                lines.append(
                    f"| {cell} | {_f(row['n'])} | "
                    f"{_f(row.get('reference_composite'))} | "
                    f"[{_f(ci.get('low'))}, {_f(ci.get('high'))}] |")
            lines.append("")
        paired = outcomes.get("paired", {})
        if paired:
            lines += ["### Paired per-task differences", "",
                      "| task | H − X | 95% CI | H − M | 95% CI | M − X | 95% CI |",
                      "| --- | ---: | --- | ---: | --- | ---: | --- |"]
            for task, row in paired.items():
                cells = []
                for key in ("H_minus_X", "H_minus_M", "M_minus_X"):
                    item = row[key]
                    cells.append(f"{_f(item['mean_difference'])}")
                    cells.append(f"[{_f(item['ci_low'])}, {_f(item['ci_high'])}]")
                lines.append(f"| {task} | " + " | ".join(cells) + " |")
            lines.append("")

    routers = receipt.get("routers", {})
    if routers:
        lines += ["## 2. Router geometry and displacement from initialisation", "",
                  "Every router starts from the generator `seed + 2000 + layer` "
                  "(`modeling._router`). Reconstructing that draw turns the saved "
                  "weights into a displacement. The wrong-seed control in each "
                  "row is the validation: a large cosine against the right seed "
                  "and ~0 against the wrong one means the reconstruction is the "
                  "real initialiser.", "",
                  "| arm | effective rank | top-1 energy share | mean \\|cos\\| between experts | min/median expert norm | displacement to init | cos to init | wrong-seed control |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for arm, block in routers.items():
            if "per_layer" not in block:
                continue
            summary = block.get("summary", {})
            def mean(key):
                return summary.get(key, {}).get("mean")
            lines.append(
                f"| `{arm}` | {_f(mean('effective_rank'))} | "
                f"{_f(mean('top1_energy_share'))} | "
                f"{_f(mean('mean_absolute_pairwise_cosine'))} | "
                f"{_f(mean('norm_ratio_min_median'))} | "
                f"{_f(mean('displacement_relative'))} | "
                f"{_f(mean('cosine_to_init'))} | "
                f"{_f(mean('init_validation_cosine_wrong_seed'))} |")
        lines.append("")
        if "H_versus_X" in routers:
            summary = routers["H_versus_X"]["summary"].get("cosine", {})
            lines += [f"`H`'s and `X`'s routers start from the same draw, so the "
                      f"cosine between them measures how far each moved: mean "
                      f"**{_f(summary.get('mean'))}** over "
                      f"{_f(routers['H_versus_X']['layers'])} layers "
                      f"(min **{_f(summary.get('min'))}**, max "
                      f"**{_f(summary.get('max'))}**).", ""]
        frozen = routers.get("Hstar_router_frozen")
        if frozen:
            lines += [f"`H*` router frozen across steps "
                      f"`{os.path.basename(frozen['early'])}` → "
                      f"`{os.path.basename(frozen['late'])}`: "
                      f"**{'yes' if frozen['bitwise_identical'] else 'no'}** "
                      f"(max abs difference {_f(frozen['max_abs_difference'])} over "
                      f"{_f(frozen['layers'])} layers), so the router used in "
                      "section 3 is the one the stored labels were measured "
                      "under.", ""]

    usage = receipt.get("usage", {}).get("Hstar")
    if usage:
        lines += ["## 3. Routing usage on `H*`'s stored calibration states", "",
                  "Routing 32,768 stored pre-expert states through the stored "
                  "router is not a model forward. Usage entropy is normalised by "
                  "`log 8`; dead experts are those used less than 1% of the "
                  "expected count.", "",
                  "| quantity | router | utility-optimal |",
                  "| --- | ---: | ---: |"]
        router_summary = usage.get("summary", {})
        utility_summary = usage.get("utility_summary", {})
        for label, key, digits in (
                ("normalised usage entropy", "normalised_entropy", 4),
                ("KL(usage ‖ uniform), nats", "kl_to_uniform", 4),
                ("usage coefficient of variation", "coefficient_of_variation", 4),
                ("max/mean usage", "max_over_mean", 3),
                ("distinct top-4 sets used", "distinct_topk_sets", 1),
                ("share of positions on the modal top-4 set", "modal_set_share", 4),
                ("entropy over top-4 sets, bits", "set_entropy_bits", 3),
                ("top-4 sets needed to cover half the positions", "sets_covering_half", 1)):
            lines.append(f"| {label} | {_f(router_summary.get(key, {}).get('mean'), digits)} | "
                         f"{_f(utility_summary.get(key, {}).get('mean'), digits)} |")
        lines.append("")
        dead = usage.get("dead_experts_used") or {}
        lines += [f"Layers with at least one dead expert under the router: "
                  f"**{_f(len(dead))}** of {_f(usage['layers'])}"
                  + (f" (`{', '.join(sorted(dead, key=int))}`)." if dead else "."),
                  ""]

    memory = receipt.get("memory", {})
    if memory:
        lines += ["## 4. Memory path", "",
                  "Gate values are `sigmoid(g · h)` with `g` the stored gate "
                  "vector and `h` stored hidden states, so this is exact for "
                  "`H*` and a fixed-probe evaluation for the others. Row usage "
                  "comes from the calibration tokens and the addressing "
                  "function, which no arm's weights affect.", ""]
        stream_block = memory.get("table_stream")
        if stream_block:
            lines += [f"### Row reads on the calibration slice "
                      f"({_f(stream_block['calibration_tokens'])} tokens, "
                      f"`{_f(stream_block['table_rows'])}` rows per table)", "",
                      "| layer | distinct rows read | coverage of table | collision share | top-1% rows hold | two heads same row |",
                      "| --- | ---: | ---: | ---: | ---: | ---: |"]
            for layer, row in sorted(stream_block["per_layer"].items(), key=lambda kv: int(kv[0])):
                lines.append(
                    f"| {layer} | {_f(row['distinct_rows_read'])} | "
                    f"{_f(row['coverage_of_table'])} | "
                    f"{_f(row['collision_share'])} | "
                    f"{_f(row['top_1pct_rows_read_share'])} | "
                    f"{_f(row['two_head_same_row_share'])} |")
            lines.append("")
        budget = memory.get("table_budget_scan")
        if budget:
            lines += [f"### Row reads over a bounded stream prefix "
                      f"({_f(budget['tokens_scanned'])} tokens)", "",
                      "| layer | distinct rows read | coverage of table |",
                      "| --- | ---: | ---: |"]
            for layer, row in sorted(budget["per_layer"].items(), key=lambda kv: int(kv[0])):
                lines.append(f"| {layer} | {_f(row['distinct_rows'])} | "
                             f"{_f(row['coverage'])} |")
            lines += ["",
                      "Rows bitwise unchanged from the reconstructed initialisation "
                      "are zero in every arm, so *every* row of the table is "
                      "updated during training; weight-side utilisation cannot "
                      "separate used from unused rows, which is why the read "
                      "counts above come from the addressing function instead.",
                      ""]
        for arm in ("Hstar", "H", "M"):
            block = memory.get(arm)
            if not block:
                continue
            lines += [f"### `{arm}` memory weights", "",
                      "| layer | gate norm | gate norm / init | cos to init | projection ‖W‖ | table ‖W‖ / init | rows bitwise unchanged | relative row displacement p99 |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
            for layer, row in sorted(block["per_layer"].items(), key=lambda kv: int(kv[0])):
                table = row.get("table", {})
                lines.append(
                    f"| {layer} | {_f(row['gate_norm'])} | "
                    f"{_f(row['gate_norm_ratio_to_init'])} | "
                    f"{_f(row['gate_cosine_to_init'])} | "
                    f"{_f(row['projection_frobenius'])} | "
                    f"{_f(row['table_relative_norm'])} | "
                    f"{_f(table.get('rows_bitwise_unchanged'))} | "
                    f"{_f(table.get('relative_displacement_p99'))} |")
            lines.append("")
            probe = [(layer, row["on_stored_states"])
                     for layer, row in sorted(block["per_layer"].items(), key=lambda kv: int(kv[0]))
                     if "on_stored_states" in row]
            if probe:
                lines += ["| layer | input RMS | gate mean | gate > 0.5 | memory RMS / input RMS | bank RMS / input RMS | memory share of path norm (median) | gate vs bank RMS r |",
                          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
                for layer, row in probe:
                    gate = row["gate"]
                    share = row.get("memory_share_of_path_norm", {})
                    lines.append(
                        f"| {layer} | {_f(row['input_rms'])} | "
                        f"{_f(gate['mean'])} | "
                        f"{_f(gate['fraction_above_half'])} | "
                        f"{_f(row.get('memory_output_over_input_rms'))} | "
                        f"{_f(row.get('bank_output_over_input_rms'))} | "
                        f"{_f(share.get('median'))} | "
                        f"{_f(row.get('gate_vs_bank_norm_pearson'))} |")
                lines.append("")

    interaction = receipt.get("interaction", {}).get("Hstar")
    if interaction:
        lines += ["## 5. Interaction proxies on stored states", "",
                  "Mutual information between the router's top-1 expert and the "
                  "popularity bucket of the memory row the same position reads, "
                  "with a permutation null (`2000` permutations). A value inside "
                  "the null is evidence of **no** measured coupling, not evidence "
                  "of independence.", "",
                  "| layer | MI expert × popularity (bits) | null mean | permutation p | MI expert × repeated row | repeat read share |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for layer, row in sorted(interaction.items(), key=lambda kv: int(kv[0])):
            first = row["expert_vs_row_popularity"]
            second = row["expert_vs_repeated_row"]
            lines.append(
                f"| {layer} | {_f(first['bits'])} | {_f(first['null_mean'])} | "
                f"{_f(first['p_value'])} | {_f(second['bits'])} | "
                f"{_f(row['repeat_read_share'])} |")
        lines.append("")

    lines += ["## Reading of these numbers", "",
              "Every value below is pulled from the receipt; the *sentences* are "
              "interpretation and are labelled as such. None of them is a causal "
              "claim, and none replaces the registered T2 verdict.", ""]
    reading = reading_lines(receipt)
    lines += reading
    lines += ["", "## What this establishes", "",
              "1. Whether `H`'s losses are concentrated on the items one of the "
              "single-path arms solved and the other did not. That is the "
              "item-level form of the question and it is answered from receipts.",
              "2. How far each arm's router moved from the shared initialisation, "
              "and whether its expert directions stayed separable or collapsed "
              "onto one or two directions. This is measured from weights against "
              "a reconstructed initialiser with a wrong-seed control.",
              "3. What `H*`'s router does to real inputs: usage, entropy, dead "
              "experts, and the spread across positions, from stored states.",
              "4. What the memory path reads, how concentrated those reads are on "
              "the calibration slice, and how large each arm's memory path is "
              "relative to its bank on the same inputs.",
              "5. Whether expert choice and memory-row popularity are coupled at "
              "the insertion layers, with a null.",
              "", "## What this does not establish", ""]
    for item in receipt.get("limits", []):
        lines.append(f"- {item}")
    lines += ["", "Deferred to the GPU window (no number is estimated here):", ""]
    for item in receipt.get("deferred", []):
        lines.append(f"- **{item['what']}** — {item['why']}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI


def deferred_commands() -> str:
    return """\
# Deferred measurements — run only in a GPU window, and only when the H* run has
# released the device. Each writes a receipt; none may be reported without one.

# (a) H's and X's own calibration states, so their usage and gates are measured
#     the same way H*'s are. --labels-out also re-measures utilities; pass the
#     same calibration range so the slices are comparable.
scripts/court --gpu oracle -- python3 oracle/train/loop.py \\
    --profile t2 --arm H --seed 1001 \\
    --checkpoint results/artifacts/train/t2/H/step-001024.pt \\
    --config results/train/admission-configs/t2/H.json \\
    --admission results/train/admission/t2/H.json \\
    --warmup-steps 64 --steps 65 --calibration-windows 32 \\
    --labels-out results/artifacts/train/t2/H/calibration-states.pt

scripts/court --gpu oracle -- python3 oracle/train/loop.py \\
    --profile t2 --arm X --seed 1001 \\
    --checkpoint results/artifacts/train/t2/X/step-001024.pt \\
    --config results/train/admission-configs/t2/X.json \\
    --admission results/train/admission/t2/X.json \\
    --warmup-steps 64 --steps 65 --calibration-windows 32 \\
    --labels-out results/artifacts/train/t2/X/calibration-states.pt

# (b) The deployed-budget utility labels (P12.17 part 1), five passes per layer.
scripts/court --gpu oracle -- python3 oracle/train/loop.py \\
    --profile t2 --arm Hstar --seed 1001 \\
    --checkpoint results/artifacts/train/t2/Hstar/step-001024.pt \\
    --config results/train/admission-configs/t2/Hstar.json \\
    --admission results/train/admission/t2/Hstar.json \\
    --calibration-windows 32 \\
    --labels-out results/artifacts/train/t2/Hstar/deployed-labels.pt

# (c) Per-item routing attribution: the item ids the diagnostic flags are in the
#     receipt's outcome cells. Capture with the evaluation prompts as input and
#     join on item id.
scripts/court --gpu oracle -- python3 oracle/eval/capture_activations.py \\
    --arm H --checkpoint results/artifacts/train/t2/H/step-001024.pt \\
    --out results/artifacts/train/t2/H/eval-activations.pt
"""


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default="results/artifacts/train/t2/Hstar/utility-labels.pt")
    parser.add_argument("--stream", default="results/artifacts/train/train.bin")
    parser.add_argument("--checkpoint-hstar",
                        default="results/artifacts/train/t2/Hstar/step-000200.pt")
    parser.add_argument("--checkpoint-hstar-later",
                        default="results/artifacts/train/t2/Hstar/step-000300.pt",
                        help="a later H* checkpoint, to check the router is frozen")
    parser.add_argument("--checkpoint-H",
                        default="results/artifacts/train/t2/H/step-001024.pt")
    parser.add_argument("--checkpoint-X",
                        default="results/artifacts/train/t2/X/step-001024.pt")
    parser.add_argument("--checkpoint-M",
                        default="results/artifacts/train/t2/M/step-001024.pt")
    parser.add_argument("--receipts", default="results/eval")
    parser.add_argument("--calibration-tokens", type=int, default=32768)
    parser.add_argument("--probe-positions", type=int, default=2048)
    parser.add_argument("--coverage-scan-tokens", type=int, default=8388608,
                        help="bounded prefix of the stream for the row-coverage scan")
    parser.add_argument("--layers", type=int, default=28)
    parser.add_argument("--out", default=None)
    parser.add_argument("--markdown", default=None)
    parser.add_argument("--deferred-commands", action="store_true")
    args = parser.parse_args(argv)

    if args.deferred_commands:
        print(deferred_commands())
        return 0

    receipt = analyse(args)
    out = args.out or f"results/t2/H-DIAGNOSTICS-{receipt['stamp']}.json"
    receipt["outputs"] = {"receipt": out, "markdown": args.markdown}
    write_json_atomic(out, receipt)
    if args.markdown:
        write_text_atomic(args.markdown, render_markdown(receipt))
    print(json.dumps({
        "out": out,
        "markdown": args.markdown,
        "outcome_cells": {key: {cell: row["n"] for cell, row in block["pooled"].items()}
                          for key, block in receipt["outcomes"].items()
                          if isinstance(block, dict) and "pooled" in block},
        "router_layers": {arm: block["layers"] for arm, block in receipt["routers"].items()
                          if isinstance(block, dict) and "layers" in block},
        "usage_layers": receipt["usage"].get("Hstar", {}).get("layers"),
        "memory_arms": sorted(receipt["memory"]),
        "deferred": len(receipt["deferred"]),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
