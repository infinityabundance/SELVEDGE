#!/usr/bin/env python3
"""S46: does a token-conditioned gate have headroom? Oracle upper bounds (CPU).

The measured damage is a *persistent* one: on the correct solution path the
memory-2m residual adds CE on about 80% of positions and the per-read gate is a
near-constant |alpha| ~ 0.47, so the token gate as implemented cannot separate
the positions it helps from the positions it hurts.  This script measures how
much a gate *could* recover, which is the decision the next mechanism depends
on:

* if a per-token oracle gate recovers most of the loss, a learned gate has
  headroom and is the right next mechanism;
* if even the oracle cannot, the residual content is wrong and the corrective
  objective has to change instead.

Three gating upper bounds are computed per signal (CE and KL to the 2B teacher):

(a) **per-token oracle** -- apply the memory exactly where it lowers the
    signal at that token.  This is an upper bound over every possible gate,
    including non-causal ones, and it is deliberately optimistic: the two
    branches are teacher-forced independently, so it ignores the fact that a
    gate changes the later hidden states.
(b) **class oracle** -- apply the memory per token class (digit, operator,
    space, letter, other, latex), fitted on one half of the prompts and
    evaluated on the other half as well as in-sample.
(c) **cheap-feature oracle** -- a one-threshold gate on a feature that is
    available without a second forward at generation time: the base model's
    entropy at that position, the memory residual norm, and the position index.
    The threshold and direction are fitted on one half and evaluated on the
    other, and the out-of-fold pooled number is reported.

``--forward`` measures the per-position quantities on a fixed, selected sample
of GSM8K prompts and writes a receipt.  ``--bounds`` turns that receipt into
the upper bounds above (pure Python, no torch).  ``--generate`` runs bounded
greedy decoding on the same prompts under five gate policies and reports the
final-answer accuracy, once per policy; a per-token *oracle* accuracy is not
reported because it would need the answer to choose the branch.

Every number in a receipt comes from the run that wrote it.  The sample is a
*selected* damage sample (base correct, memory-2m wrong) plus a small control
group; it is not the anchored benchmark and its accuracy is not a benchmark
score.

Usage
-----
    scripts/court oracle -- python3 scripts/s46_oracle_gate.py --forward
    scripts/court python -- python3 scripts/s46_oracle_gate.py --bounds \
        --input results/stage2/s46-oracle-forwards-<stamp>.json
    scripts/court oracle -- python3 scripts/s46_oracle_gate.py --generate \
        --bounds-input results/stage2/s46-oracle-bounds-<stamp>.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import s45_collapse_diagnosis as s45  # noqa: E402  (same sample, same loaders)

REPO = s45.REPO
OUT_DIR = s45.OUT_DIR
SCHEMA = "selvedge/s46-oracle-gate/v1"

#: the token classes the class oracle may use. `end` is reported by the forward
#: receipt but never gated on: it is EOS, which no generated position reaches.
CLASSES = ("digit", "operator", "space", "letter", "other", "latex")
FEATURES = ("base_entropy", "memory_residual_norm", "position")


def utcnow() -> str:
    return s45.utcnow()


def stamp() -> str:
    return s45.stamp()


def revision() -> str:
    return s45.revision()


def write_json(path: str, payload: dict) -> None:
    s45.write_json(os.path.join(REPO, path), payload)


def read_json(path: str) -> dict:
    with open(os.path.join(REPO, path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def _mean(values) -> float | None:
    kept = [v for v in values if v is not None]
    return statistics.fmean(kept) if kept else None


# ---------------------------------------------------------------------------
# forward pass


def _sample(tok, args) -> tuple[list[dict], dict]:
    """The fixed sample: shortest damage prompts, then the shortest controls."""
    items = s45._select_items(tok, args)
    base_items = s45.load_items(
        os.path.join(REPO, s45.RAW["base"]["gsm8k"]), "strict-match")
    for item in items:
        item["target"] = base_items[item["doc_id"]]["target"]
    damage = [item for item in items if item["base_ok"] and not item["memory_ok"]]
    controls = [item for item in items if item["base_ok"] and item["memory_ok"]]
    damage.sort(key=lambda r: r["doc_id"])
    controls.sort(key=lambda r: r["doc_id"])
    for index, item in enumerate(damage):
        item["split"] = "A" if index % 2 == 0 else "B"
    for index, item in enumerate(controls):
        item["split"] = "A" if index % 2 == 0 else "B"
        item["control"] = True
    sample = {
        "damage_prompts": [item["doc_id"] for item in damage],
        "control_prompts": [item["doc_id"] for item in controls],
        "folds": {
            "A": [item["doc_id"] for item in damage if item["split"] == "A"],
            "B": [item["doc_id"] for item in damage if item["split"] == "B"],
        },
        "selection_rule": "shortest GSM8K prompts (>= 64 tokens) where base is "
                          "correct and memory-2m is wrong, then the shortest "
                          "prompts where both are correct (controls); the "
                          "scored text is the memory-2m arm's own stored "
                          "generation",
        "not_the_benchmark": "selected damage sample; its accuracy is not an "
                             "anchored benchmark score",
    }
    return damage + controls, sample


def run_forward(args) -> int:
    import torch

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(s45.STUDENT_DIR, trust_remote_code=False)
    teacher_tok = AutoTokenizer.from_pretrained(s45.TEACHER_DIR,
                                                trust_remote_code=False)
    if tok.get_vocab() != teacher_tok.get_vocab():
        raise SystemExit("student and teacher tokenizers differ")

    items, sample = _sample(tok, args)
    if not items:
        raise SystemExit("no prompts selected")
    sequences = []
    for item in items:
        for source in args.source:
            built = s45._build_sequence(tok, item, args.prompt_tokens,
                                        args.completion_tokens, source)
            if built is not None and built["completion_len"] >= 4:
                sequences.append((item, source, built))

    torch_mod, arms, _memory, student, teacher, mem_module, _adapter = \
        s45._load_models()
    student.eval()
    teacher.eval()
    receipt = {
        "schema": SCHEMA,
        "kind": "oracle_gate_forward",
        "stamp": stamp(),
        "created_utc": utcnow(),
        "revision": revision(),
        "device": "cpu",
        "dtype": "bfloat16",
        "attention": "eager",
        "prompt_tokens_cap": args.prompt_tokens,
        "completion_tokens_cap": args.completion_tokens,
        "sample": sample,
        "sources": list(args.source),
        "sample_size": {"prompts": len(items),
                        "sequences": len(sequences),
                        "damage": sum(1 for item in items
                                      if not item.get("control")),
                        "controls": sum(1 for item in items
                                        if item.get("control"))},
        "checkpoints": {
            "memory": {"path": s45.MEMORY_CKPT,
                       "sha256": s45.sha256_file(
                           os.path.join(REPO, s45.MEMORY_CKPT))},
        },
        "student": s45.STUDENT_DIR,
        "teacher": s45.TEACHER_DIR,
        "cgroup_after_load": s45.cgroup_memory(),
        "rss_peak_after_load_bytes": s45.rss_peak_bytes(),
        "items": [],
    }

    started = time.time()
    for item, source, built in sequences:
        ids = built["input_ids"]
        prompt_len = built["prompt_len"]
        completion_len = built["completion_len"]
        input_tensor = torch_mod.tensor(ids, dtype=torch_mod.long)
        targets = input_tensor[prompt_len:]
        positions = list(range(prompt_len - 1, prompt_len - 1 + completion_len))

        base_logits = s45._logits(torch_mod, student, ids)
        teacher_logits = s45._logits(torch_mod, teacher, ids)

        stats: dict = {}
        handles = arms.install_memory(student, mem_module)
        try:
            def hook(module, _inputs, output):
                stats["mixed_norm"] = output.detach().float().norm(dim=-1)[0]
                if module.last_gate is not None:
                    gate = module.last_gate.float().abs()
                    stats["gate_mean"] = gate.mean(dim=-1)[0]
                    stats["gate_max"] = gate.amax(dim=-1)[0]

            handle = mem_module.register_forward_hook(hook)
            mem_module.reset_state()
            memory_logits = s45._logits(torch_mod, student, ids)
            handle.remove()
        finally:
            arms.uninstall(handles)
            mem_module.reset_state()

        b = base_logits[positions]
        m = memory_logits[positions]
        t = teacher_logits[positions]
        ce_base = s45._ce(b, targets).tolist()
        ce_mem = s45._ce(m, targets).tolist()
        kl_base = s45._kl(t, b).tolist()
        kl_mem = s45._kl(t, m).tolist()

        import torch.nn.functional as F
        logp_base = F.log_softmax(b, dim=-1)
        entropy = (-(logp_base.exp() * logp_base).sum(-1)).tolist()
        max_prob = logp_base.exp().amax(-1).tolist()

        token_text = [tok.decode([token_id]) for token_id in built["completion_ids"]]
        token_class = [s45.char_class(text[0] if text else None)
                       for text in token_text]

        def at(name):
            value = stats.get(name)
            return None if value is None else value[positions].tolist()

        receipt["items"].append({
            "doc_id": item["doc_id"],
            "source": source,
            "split": item["split"],
            "control": bool(item.get("control")),
            "base_ok": item["base_ok"],
            "memory_ok": item["memory_ok"],
            "prompt": item["prompt"],
            "target": item["target"],
            "stored_generation": item[f"{source}_response"],
            "prompt_tokens_used": prompt_len,
            "completion_tokens_used": completion_len,
            "positions": {
                "token": token_text,
                "class": token_class,
                "position": list(range(completion_len)),
                "ce_base": ce_base,
                "ce_memory": ce_mem,
                "kl_base_teacher": kl_base,
                "kl_memory_teacher": kl_mem,
                "base_entropy": entropy,
                "base_max_prob": max_prob,
                "memory_residual_norm": at("mixed_norm"),
                "memory_gate_mean_abs": at("gate_mean"),
                "memory_gate_max_abs": at("gate_max"),
            },
        })
        delta = [mm - bb for mm, bb in zip(ce_mem, ce_base)]
        print(f"doc {item['doc_id']} source={source} split={item['split']} "
              f"P={prompt_len} C={completion_len} "
              f"CE base/mem={statistics.fmean(ce_base):.4f}/"
              f"{statistics.fmean(ce_mem):.4f} "
              f"mean_delta={statistics.fmean(delta):+.4f} "
              f"help={sum(1 for d in delta if d < 0)}/{len(delta)}")

    receipt["seconds"] = round(time.time() - started, 3)
    receipt["tokens_scored"] = sum(item["completion_tokens_used"]
                                   for item in receipt["items"])
    receipt["cgroup"] = s45.cgroup_memory()
    receipt["rss_peak_bytes"] = s45.rss_peak_bytes()
    out = args.out or os.path.join(
        OUT_DIR, f"s46-oracle-forwards-{receipt['stamp']}.json")
    write_json(out, receipt)
    print(f"receipt: {out}")
    print(f"seconds={receipt['seconds']} tokens={receipt['tokens_scored']} "
          f"rss_peak_gib={(receipt['rss_peak_bytes'] or 0) / s45.GIB:.2f}")
    return 0


# ---------------------------------------------------------------------------
# upper bounds


def _flatten(items: list[dict], damage_only: bool = True) -> list[dict]:
    rows = []
    for item in items:
        if damage_only and item.get("control"):
            continue
        positions = item["positions"]
        norm = positions["memory_residual_norm"] or [None] * len(
            positions["ce_base"])
        for index in range(len(positions["ce_base"])):
            rows.append({
                "doc_id": item["doc_id"],
                "source": item.get("source"),
                "split": item["split"],
                "class": positions["class"][index],
                "position": positions["position"][index],
                "ce_base": positions["ce_base"][index],
                "ce_memory": positions["ce_memory"][index],
                "kl_base": positions["kl_base_teacher"][index],
                "kl_memory": positions["kl_memory_teacher"][index],
                "base_entropy": positions["base_entropy"][index],
                "memory_residual_norm": norm[index],
            })
    return rows


def _level(rows: list[dict], signal: str) -> dict:
    base = _mean(row[f"{signal}_base"] for row in rows)
    mem = _mean(row[f"{signal}_memory"] for row in rows)
    helps = sum(1 for row in rows
                if row[f"{signal}_memory"] < row[f"{signal}_base"])
    return {
        "all_off": base,
        "all_on": mem,
        "delta_all_on_minus_all_off": (None if base is None or mem is None
                                       else mem - base),
        "positions": len(rows),
        "positions_memory_helps": helps,
        "fraction_memory_helps": helps / len(rows) if rows else None,
    }


def _recovery(level: dict, value: float | None) -> dict:
    off, on = level["all_off"], level["all_on"]
    damage = None if off is None or on is None else on - off
    return {
        "value": value,
        "recovered_nats": (None if value is None or on is None else on - value),
        "recovery_fraction": (None if value is None or not damage or damage <= 0
                              else (on - value) / damage),
        "recovery_defined": bool(damage is not None and damage > 0),
    }


def _per_token_oracle(rows: list[dict], signal: str) -> dict:
    level = _level(rows, signal)
    value = _mean(min(row[f"{signal}_base"], row[f"{signal}_memory"])
                  for row in rows)
    return {"level": level, **_recovery(level, value),
            "gate_on_fraction": level["fraction_memory_helps"]}


def _class_table(rows: list[dict], signal: str) -> dict:
    table = {}
    for name in CLASSES:
        subset = [row for row in rows if row["class"] == name]
        if not subset:
            continue
        delta = _mean(row[f"{signal}_memory"] - row[f"{signal}_base"]
                      for row in subset)
        table[name] = {
            "positions": len(subset),
            "base": _mean(row[f"{signal}_base"] for row in subset),
            "memory": _mean(row[f"{signal}_memory"] for row in subset),
            "delta": delta,
            "apply_memory": bool(delta is not None and delta < 0),
        }
    return table


def _apply_class_gate(rows: list[dict], table: dict, signal: str):
    picked = []
    for row in rows:
        entry = table.get(row["class"])
        on = bool(entry and entry["apply_memory"])
        picked.append(row[f"{signal}_memory"] if on else row[f"{signal}_base"])
    return _mean(picked)


def _class_on_fraction(rows: list[dict], table: dict) -> float | None:
    if not rows:
        return None
    on = sum(1 for row in rows
             if (table.get(row["class"]) or {}).get("apply_memory"))
    return on / len(rows)


def _class_oracle(rows: list[dict], signal: str) -> dict:
    level = _level(rows, signal)
    in_sample_table = _class_table(rows, signal)
    in_sample = _apply_class_gate(rows, in_sample_table, signal)
    folds = {}
    fold_values = []
    for fit_split, eval_split in (("A", "B"), ("B", "A")):
        fit_rows = [row for row in rows if row["split"] == fit_split]
        eval_rows = [row for row in rows if row["split"] == eval_split]
        table = _class_table(fit_rows, signal)
        value = _apply_class_gate(eval_rows, table, signal)
        fold_values.append(value)
        folds[f"{fit_split}->{eval_split}"] = {
            "fitted_on": fit_split, "evaluated_on": eval_split, "value": value,
            "on_classes": sorted(name for name, entry in table.items()
                                 if entry["apply_memory"]),
            "gate_on_fraction": _class_on_fraction(eval_rows, table),
        }
    cross = _mean(fold_values)
    return {
        "level": level,
        "in_sample": {**_recovery(level, in_sample), "value": in_sample,
                      "table": in_sample_table,
                      "on_classes": sorted(name for name, entry
                                           in in_sample_table.items()
                                           if entry["apply_memory"])},
        "cross_fitted": {**_recovery(level, cross), "value": cross,
                         "folds": folds},
    }


def _threshold_candidates(values: list[float], count: int = 19) -> list[float]:
    ordered = sorted(values)
    if not ordered:
        return []
    picks = []
    for index in range(1, count + 1):
        position = min(len(ordered) - 1,
                       max(0, round(index / (count + 1) * (len(ordered) - 1))))
        picks.append(ordered[position])
    return sorted(set(picks))


def _apply_threshold(rows: list[dict], feature: str, direction: str,
                     threshold: float, signal: str):
    picked = []
    for row in rows:
        value = row[feature]
        if value is None:
            picked.append(row[f"{signal}_base"])
            continue
        on = value >= threshold if direction == "high" else value <= threshold
        picked.append(row[f"{signal}_memory"] if on else row[f"{signal}_base"])
    return _mean(picked)


def _fit_feature(rows: list[dict], feature: str, signal: str) -> dict | None:
    candidate_rows = [row for row in rows if row[feature] is not None]
    if not candidate_rows:
        return None
    best = None
    for direction in ("high", "low"):
        for threshold in _threshold_candidates(
                [row[feature] for row in candidate_rows]):
            value = _apply_threshold(candidate_rows, feature, direction,
                                     threshold, signal)
            if value is None:
                continue
            if best is None or value < best["fit_value"]:
                best = {"direction": direction, "threshold": threshold,
                        "fit_value": value}
    return best


def _feature_oracle(rows: list[dict], feature: str, signal: str) -> dict:
    level = _level(rows, signal)
    folds = {}
    fold_values = []
    for fit_split, eval_split in (("A", "B"), ("B", "A")):
        fit_rows = [row for row in rows if row["split"] == fit_split]
        eval_rows = [row for row in rows if row["split"] == eval_split]
        best = _fit_feature(fit_rows, feature, signal)
        if best is None or not eval_rows:
            continue
        value = _apply_threshold(eval_rows, feature, best["direction"],
                                 best["threshold"], signal)
        fold_values.append(value)
        counted = [row for row in eval_rows if row[feature] is not None]
        on_fraction = None
        if counted:
            on = sum(1 for row in counted
                     if (row[feature] >= best["threshold"]
                         if best["direction"] == "high"
                         else row[feature] <= best["threshold"]))
            on_fraction = on / len(counted)
        folds[f"{fit_split}->{eval_split}"] = {
            "fitted_on": fit_split, "evaluated_on": eval_split,
            "direction": best["direction"], "threshold": best["threshold"],
            "fit_value": best["fit_value"], "value": value,
            "gate_on_fraction": on_fraction,
        }
    cross = _mean(fold_values)
    return {**_recovery(level, cross), "value": cross, "feature": feature,
            "folds": folds}


def run_bounds(args) -> int:
    forward = read_json(args.input)
    items = forward["items"]
    damage = _flatten(items, damage_only=True)
    if not damage:
        raise SystemExit("no damage positions in the forward receipt")
    receipt = {
        "schema": SCHEMA,
        "kind": "oracle_gate_bounds",
        "stamp": stamp(),
        "created_utc": utcnow(),
        "revision": revision(),
        "input": {"path": args.input,
                  "sha256": s45.sha256_file(os.path.join(REPO, args.input)),
                  "stamp": forward.get("stamp"),
                  "revision": forward.get("revision")},
        "sample": forward.get("sample"),
        "sample_size": forward.get("sample_size"),
        "positions": len(damage),
        "notes": {
            "ce": "CE of the arm's own stored generation token, in nats",
            "kl": "KL(teacher_T || arm) at T=1, in nats",
            "per_token_is_an_upper_bound": "the two branches are teacher-forced "
                                           "independently, so the per-token "
                                           "oracle ignores the effect of the "
                                           "gate on later hidden states; every "
                                           "real gate is below it",
            "cross_fitted": "the class/feature gate is fitted on one half of "
                            "the damage prompts and evaluated on the other; "
                            "folds are assigned by doc_id order, not by score",
            "controls_excluded": "control prompts (base and memory both "
                                 "correct) are in the forward receipt but not "
                                 "in these bounds",
        },
        "results": {},
    }
    for signal in ("ce", "kl"):
        receipt["results"][signal] = {
            "per_token_oracle": _per_token_oracle(damage, signal),
            "class_oracle": _class_oracle(damage, signal),
            "feature_oracle": {feature: _feature_oracle(damage, feature, signal)
                               for feature in FEATURES},
            "by_source": {source: _per_token_oracle(
                [row for row in damage if row["source"] == source], signal)
                for source in sorted({row["source"] for row in damage})
                if any(row["source"] == source for row in damage)},
        }

    # The parameters a generating gate needs, fitted per fold on the other
    # fold's teacher-forced positions.
    receipt["folds"] = {}
    for fit_split in ("A", "B"):
        fit_rows = [row for row in damage if row["split"] == fit_split]
        table = _class_table(fit_rows, "ce")
        entry = {
            "fitted_on": fit_split,
            "on_classes": sorted(name for name, value in table.items()
                                 if value["apply_memory"]),
            "class_table": table,
        }
        for feature in FEATURES:
            entry[feature] = _fit_feature(fit_rows, feature, "ce")
        receipt["folds"][fit_split] = entry

    ce = receipt["results"]["ce"]
    receipt["headroom"] = {
        "ce_all_on_minus_all_off_nats_per_token":
            ce["per_token_oracle"]["level"]["delta_all_on_minus_all_off"],
        "ce_recovered_by_per_token_oracle_nats_per_token":
            ce["per_token_oracle"]["recovered_nats"],
        "ce_recovered_by_class_oracle_cross_fitted_nats_per_token":
            ce["class_oracle"]["cross_fitted"]["recovered_nats"],
        "ce_recovered_by_entropy_oracle_cross_fitted_nats_per_token":
            ce["feature_oracle"]["base_entropy"]["recovered_nats"],
        "reading": "recovered_nats is how much CE the gate removes relative to "
                   "memory-always-on; the fraction is against the full damage "
                   "of memory-always-on over memory-always-off",
    }
    out = args.out or os.path.join(
        OUT_DIR, f"s46-oracle-bounds-{receipt['stamp']}.json")
    write_json(out, receipt)

    for signal in ("ce", "kl"):
        result = receipt["results"][signal]

        def fmt(value):
            return "n/a" if value is None else f"{value:.4f}"

        per_token = result["per_token_oracle"]
        class_cross = result["class_oracle"]["cross_fitted"]
        print(f"{signal}: all_off={fmt(per_token['level']['all_off'])} "
              f"all_on={fmt(per_token['level']['all_on'])} "
              f"per_token_oracle={fmt(per_token['value'])} "
              f"recovery_fraction={fmt(per_token['recovery_fraction'])} "
              f"gate_on={fmt(per_token['gate_on_fraction'])}")
        print(f"{signal}: class_gate_cross_fitted={fmt(class_cross['value'])} "
              f"recovery_fraction={fmt(class_cross['recovery_fraction'])}")
        for source, entry in result["by_source"].items():
            print(f"{signal}: source={source} positions={entry['level']['positions']} "
                  f"all_off={fmt(entry['level']['all_off'])} "
                  f"all_on={fmt(entry['level']['all_on'])} "
                  f"delta={fmt(entry['level']['delta_all_on_minus_all_off'])} "
                  f"gate_on={fmt(entry['gate_on_fraction'])}")
        for feature, entry in result["feature_oracle"].items():
            print(f"{signal}: feature={feature} value={fmt(entry['value'])} "
                  f"recovery_fraction={fmt(entry['recovery_fraction'])}")
    print(f"receipt: {out}")
    return 0


# ---------------------------------------------------------------------------
# bounded generation


def _class_of_id(tok, token_id: int) -> str:
    text = tok.decode([token_id])
    return s45.char_class(text[0] if text else None)


def _gate_tensor(torch, policy: str, params: dict, tok, all_ids: list[int],
                 entropies):
    """Per-position gate for the call that covers the tail of `all_ids`."""
    length = len(entropies)
    values = []
    for offset in range(length):
        absolute = len(all_ids) - length + offset
        if policy == "memory":
            values.append(1.0)
        elif policy == "class":
            if absolute == 0:
                values.append(0.0)
            else:
                name = _class_of_id(tok, all_ids[absolute - 1])
                values.append(1.0 if name in params["on_classes"] else 0.0)
        elif policy == "entropy":
            entropy = entropies[offset]
            on = (entropy >= params["threshold"]
                  if params["direction"] == "high"
                  else entropy <= params["threshold"])
            values.append(1.0 if on else 0.0)
        else:
            values.append(1.0)
    return torch.tensor(values, dtype=torch.float32).reshape(1, length, 1)


def _greedy(torch, tok, base_model, mem_model, mem_module, ids: list[int],
            max_new: int, policy: str, params: dict) -> dict:
    import torch.nn.functional as F

    prompt_len = len(ids)
    generated: list[int] = []
    base_past = None
    mem_past = None
    gates: list[float] = []
    norms: list[float] = []
    entropies: list[float] = []
    mem_module.reset_state()
    current = torch.tensor([ids], dtype=torch.long)

    with torch.inference_mode():
        for _step in range(max_new):
            if policy == "base":
                out = base_model(input_ids=current, past_key_values=base_past,
                                 use_cache=True)
                base_past = out.past_key_values
                logits = out.logits[0].float()
                logp = F.log_softmax(logits, dim=-1)
                ent = -(logp.exp() * logp).sum(-1)
                chosen = logits[-1]
                gate = 0.0
                norm = 0.0
                ent_list = [float(value) for value in ent]
            else:
                # The base model is only run for the entropy feature; every
                # other policy reads its gate from data it already has.
                if policy == "entropy":
                    base_out = base_model(input_ids=current,
                                          past_key_values=base_past,
                                          use_cache=True)
                    base_past = base_out.past_key_values
                    base_logits = base_out.logits[0].float()
                    logp = F.log_softmax(base_logits, dim=-1)
                    ent_list = [float(value) for value in
                                -(logp.exp() * logp).sum(-1)]
                else:
                    ent_list = [0.0] * current.shape[1]
                all_ids = ids + generated
                gate_t = _gate_tensor(torch, policy, params, tok, all_ids,
                                      ent_list)
                if policy == "residual_norm":
                    state = {"threshold": params["threshold"],
                             "direction": params["direction"], "last": 0.0}

                    def hook(module, _inputs, output, state=state):
                        if module.last_mixed is None:
                            return output
                        magnitude = module.last_mixed.norm(dim=-1)  # [B,T]
                        if state["direction"] == "high":
                            on = (magnitude >= state["threshold"]).float()
                        else:
                            on = (magnitude <= state["threshold"]).float()
                        state["last"] = float(on.reshape(-1)[-1])
                        return output * on[..., None].to(output.dtype)

                else:
                    def hook(module, _inputs, output, gate_t=gate_t):
                        return output * gate_t.to(output.dtype)

                handle = mem_module.register_forward_hook(hook)
                try:
                    mem_out = mem_model(input_ids=current,
                                        past_key_values=mem_past,
                                        use_cache=True)
                finally:
                    handle.remove()
                mem_past = mem_out.past_key_values
                chosen = mem_out.logits[0, -1].float()
                if mem_module.last_mixed is not None:
                    norm = float(
                        mem_module.last_mixed.norm(dim=-1).reshape(-1)[-1])
                    gate = (float(gate_t.reshape(-1)[-1])
                            if policy != "residual_norm" else state["last"])
                else:
                    norm, gate = 0.0, 0.0

            gates.append(gate)
            norms.append(norm)
            entropies.append(ent_list[-1])
            token = int(torch.argmax(chosen))
            generated.append(token)
            if token == tok.eos_token_id:
                break
            current = torch.tensor([[token]], dtype=torch.long)

    text = tok.decode(generated, skip_special_tokens=True)
    return {
        "prompt_tokens": prompt_len,
        "new_tokens": len(generated),
        "text": text,
        "gate_on_fraction": (sum(1 for value in gates if value > 0) / len(gates)
                             if gates else None),
        "residual_norm_mean": _mean(norms),
        "base_entropy_mean": _mean(entropies),
    }


def run_generate(args) -> int:
    import torch

    from transformers import AutoTokenizer

    sys.path.insert(0, os.path.join(REPO, "oracle"))
    import frontier.rft as rft

    bounds = read_json(args.bounds_input)
    forward = read_json(bounds["input"]["path"])
    items = [item for item in forward["items"] if not item.get("control")]
    if not items:
        raise SystemExit("no damage items in the forward receipt")

    tok = AutoTokenizer.from_pretrained(s45.STUDENT_DIR, trust_remote_code=False)
    _torch, arms, _memory, base_model, teacher, mem_module, _adapter = \
        s45._load_models()
    del teacher
    mem_model = arms.load_model(s45.STUDENT_DIR, dtype=torch.bfloat16,
                               device="cpu", attn_implementation="eager")
    mem_handles = arms.install_memory(mem_model, mem_module)
    base_model.eval()
    mem_model.eval()

    policies = ["base", "memory", "class", "entropy", "residual_norm"]
    receipt = {
        "schema": SCHEMA,
        "kind": "oracle_gate_generation",
        "stamp": stamp(),
        "created_utc": utcnow(),
        "revision": revision(),
        "bounds_input": {"path": args.bounds_input,
                         "sha256": s45.sha256_file(
                             os.path.join(REPO, args.bounds_input)),
                         "stamp": bounds.get("stamp")},
        "forward_input": bounds["input"],
        "sample": forward.get("sample"),
        "sample_size": forward.get("sample_size"),
        "decode": {"strategy": "greedy", "max_new_tokens": args.max_new_tokens,
                   "stop": "eos",
                   "prompt": "the selected sample's own stored few-shot prompt",
                   "not_the_benchmark": "selected damage sample; not an "
                                        "anchored benchmark score"},
        "verifier": "oracle/frontier/rft.verify(text, target, 'gsm8k')",
        "gate_parameters": bounds.get("folds"),
        "policies": {name: {"items": []} for name in policies},
        "started_utc": utcnow(),
    }

    started = time.time()
    steps = 0
    try:
        for item in items:
            prompt_ids = tok(item["prompt"], add_special_tokens=False).input_ids
            prompt_ids = prompt_ids[-args.prompt_tokens:]
            # The gate is fitted on the *other* fold's teacher-forced
            # positions, so every generated prompt is out-of-fold.
            params = bounds["folds"]["B" if item["split"] == "A" else "A"]
            for policy in policies:
                gate_params = {
                    "on_classes": params["on_classes"],
                    "threshold": (params["base_entropy"] or {}).get("threshold"),
                    "direction": (params["base_entropy"] or {}).get("direction"),
                }
                if policy == "residual_norm":
                    fit = params["memory_residual_norm"] or {}
                    gate_params = {"threshold": fit.get("threshold"),
                                   "direction": fit.get("direction")}
                if policy in ("class", "entropy", "residual_norm") and \
                        gate_params.get("threshold") is None and \
                        policy != "class":
                    receipt["policies"][policy]["items"].append({
                        "doc_id": item["doc_id"],
                        "skipped": "the gate feature has no fitted threshold "
                                   "on the complementary fold",
                    })
                    continue
                result = _greedy(torch, tok, base_model, mem_model, mem_module,
                                 prompt_ids, args.max_new_tokens, policy,
                                 gate_params)
                steps += result["new_tokens"]
                # The harness target is the reference solution; the verifier
                # compares answers, so extract the reference answer first.
                target_answer = rft.extract(item["target"], "gsm8k").value
                verdict = ({"verdict": "unparseable", "extracted": None,
                            "reason": "the reference target has no extractable answer"}
                           if target_answer is None else
                           rft.verify(result["text"], target_answer, "gsm8k"))
                receipt["policies"][policy]["items"].append({
                    "doc_id": item["doc_id"],
                    "split": item["split"],
                    "verdict": verdict["verdict"],
                    "extracted": verdict.get("extracted"),
                    "target": item["target"],
                    "target_answer": target_answer,
                    **result,
                })
                print(f"doc {item['doc_id']} policy={policy:14s} "
                      f"new={result['new_tokens']:3d} "
                      f"gate_on={result['gate_on_fraction']} "
                      f"verdict={verdict['verdict']}")
    finally:
        arms.uninstall(mem_handles)
        mem_module.reset_state()

    for policy, entry in receipt["policies"].items():
        rows = [row for row in entry["items"] if "verdict" in row]
        entry["n"] = len(rows)
        entry["correct"] = sum(1 for row in rows if row["verdict"] == "correct")
        entry["accuracy"] = entry["correct"] / len(rows) if rows else None
        entry["gate_on_fraction_mean"] = _mean(
            row["gate_on_fraction"] for row in rows)

    receipt["seconds"] = round(time.time() - started, 3)
    receipt["decode_tokens"] = steps
    receipt["ms_per_decode_token"] = (None if not steps
                                      else 1000.0 * receipt["seconds"] / steps)
    receipt["cgroup"] = s45.cgroup_memory()
    receipt["rss_peak_bytes"] = s45.rss_peak_bytes()
    out = args.out or os.path.join(
        OUT_DIR, f"s46-oracle-generation-{receipt['stamp']}.json")
    write_json(out, receipt)

    print()
    for policy, entry in receipt["policies"].items():
        print(f"{policy:14s} correct={entry['correct']}/{entry['n']} "
              f"accuracy={entry['accuracy']} "
              f"gate_on={entry['gate_on_fraction_mean']}")
    print(f"receipt: {out}")
    print(f"seconds={receipt['seconds']} decode_tokens={steps} "
          f"ms_per_token={receipt['ms_per_decode_token']}")
    return 0


# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward", action="store_true")
    parser.add_argument("--bounds", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--input", default=None,
                        help="forward receipt for --bounds")
    parser.add_argument("--bounds-input", default=None,
                        help="bounds receipt for --generate")
    parser.add_argument("--out", default=None)
    parser.add_argument("--items", type=int, default=16)
    parser.add_argument("--controls", type=int, default=4)
    parser.add_argument("--prompt-tokens", type=int, default=320)
    parser.add_argument("--completion-tokens", type=int, default=80)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--source", action="append", default=None,
                        help="response source(s): base and/or memory (default "
                             "both, so the gate sees both trajectories)")
    args = parser.parse_args(argv)
    if args.source is None:
        args.source = ["base", "memory"]
    modes = sum(bool(flag) for flag in (args.forward, args.bounds, args.generate))
    if modes != 1:
        raise SystemExit("choose exactly one of --forward, --bounds, --generate")
    if args.bounds and not args.input:
        raise SystemExit("--bounds needs --input <forward receipt>")
    if args.generate and not args.bounds_input:
        raise SystemExit("--generate needs --bounds-input <bounds receipt>")
    if args.forward:
        return run_forward(args)
    if args.bounds:
        return run_bounds(args)
    return run_generate(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
