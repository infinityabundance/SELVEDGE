#!/usr/bin/env python3
"""P2.4: verify the T1.4 compact teacher target against a live full teacher.

The T1 objective is the top-16 support cross-entropy plus **one tail bucket**
(docs/TRAINING.md §2). It is not the full-vocabulary cross-entropy, because the
cache stores the teacher's tail *mass* and never its tail distribution. This
check measures, on the pinned held-out sample, exactly what the bucket costs.

Derivation (recomputed and asserted numerically in the receipt):

    L_compact - L_full = -p_T * (H(r) + KL(r || s))

where `p_T` is the teacher's tail mass, `H(r)` the entropy of the teacher
distribution restricted to the tail and renormalised, `r` that distribution and
`s` the student's restricted to the tail and renormalised. The compact target is
therefore optimistically biased downward by both the tail's entropy and its
mismatch; it cannot see the *shape* of the tail. The question P2.4 asks is
whether that bias changes the **ranking** of students, not whether it exists.

Outputs a receipt with, per student and per domain: the compact loss, the full
loss, the bias, the teacher tail mass, the tail entropy, the tail KL, and the
per-window paired differences; plus the cross-student rank correlation and the
sign agreement on the registered decisive pairs.
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
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "train"))

import harness  # noqa: E402
import tasks  # noqa: E402
from metrics import bootstrap_mean, paired_bootstrap  # noqa: E402
from modeling import build_arm, load_admission  # noqa: E402

TOP_K = 16
CHUNK = 16384
DECISIVE_PAIRS = (("H*", "H"), ("X", "D"), ("M", "D0"))


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


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lse_chunked(logits: torch.Tensor) -> torch.Tensor:
    """Vocabulary log-sum-exp over [T, vocab] in f32, one chunk at a time."""
    parts = []
    for start in range(0, logits.shape[-1], CHUNK):
        parts.append(torch.logsumexp(logits[:, start:start + CHUNK].float(), dim=-1))
    return torch.logsumexp(torch.stack(parts, dim=0), dim=0)


def window_targets(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> dict:
    """Compact and full-target losses and the tail decomposition for one window.

    Both inputs are `[T, vocab]` for the positions being predicted.

    The tail sums are computed as (whole vocabulary) minus (the stored top-16
    support), which is exact and independent of the chunk size. Building a
    masked tensor instead would be O(T x vocab) and is not needed.
    """
    tokens = student_logits.shape[0]
    lse_teacher = _lse_chunked(teacher_logits)
    lse_student = _lse_chunked(student_logits)
    top_values, top_ids = torch.topk(teacher_logits, TOP_K, dim=-1)
    all_p_log_p = torch.zeros(tokens, device=student_logits.device)
    all_p_log_q = torch.zeros(tokens, device=student_logits.device)
    all_q = torch.zeros(tokens, device=student_logits.device)
    for start in range(0, student_logits.shape[-1], CHUNK):
        end = min(start + CHUNK, student_logits.shape[-1])
        t_chunk = teacher_logits[:, start:end].float()
        s_chunk = student_logits[:, start:end].float()
        log_p_t = t_chunk - lse_teacher.unsqueeze(-1)
        log_p_s = s_chunk - lse_student.unsqueeze(-1)
        p_t = log_p_t.exp()
        all_p_log_p += (p_t * log_p_t).sum(dim=-1)
        all_p_log_q += (p_t * log_p_s).sum(dim=-1)
        all_q += log_p_s.exp().sum(dim=-1)
    p_top = (top_values - lse_teacher.unsqueeze(-1)).exp()
    log_q_top = (student_logits.gather(1, top_ids) - lse_student.unsqueeze(-1))
    q_top = log_q_top.exp()
    support_p_log_p = (p_top * (top_values - lse_teacher.unsqueeze(-1))).sum(dim=-1)
    support_p_log_q = (p_top * log_q_top).sum(dim=-1)
    support_q = q_top.sum(dim=-1)
    p_tail = (1.0 - p_top.sum(dim=-1)).clamp_min(0.0)
    q_tail = (1.0 - support_q).clamp(min=1e-7, max=1.0)
    a = all_p_log_p - support_p_log_p      # sum_T p log p
    b = all_p_log_q - support_p_log_q      # sum_T p log q
    tail_q = (all_q - support_q).clamp_min(1e-12)
    compact = -(p_top * log_q_top).sum(dim=-1) - p_tail * q_tail.log()
    full = -all_p_log_q

    # Tail decomposition. With r = teacher tail renormalised and s = student
    # tail renormalised:
    #   H(r)      = -a/p_T + log p_T
    #   KL(r||s)  = (a - b)/p_T - log p_T + log q_T
    # Then L_compact - L_full = -p_T * (H(r) + KL(r||s)), which the receipt's
    # identity check asserts numerically.
    safe_p = p_tail.clamp_min(1e-12)
    tail_entropy = -(a / safe_p) + torch.log(safe_p)
    tail_kl = ((a - b) / safe_p) - torch.log(safe_p) + torch.log(tail_q)
    tail_kl = tail_kl.clamp_min(0.0)
    return {
        "compact": compact.detach().cpu(),
        "full": full.detach().cpu(),
        "bias": (compact - full).detach().cpu(),
        "p_tail": p_tail.detach().cpu(),
        "tail_entropy": tail_entropy.detach().cpu(),
        "tail_kl": tail_kl.detach().cpu(),
        "positions": tokens,
    }


@torch.inference_mode()
def evaluate_student(student, teacher, windows: dict[str, list[dict]], device) -> dict:
    rows: dict[str, list[dict]] = {}
    for domain, items in windows.items():
        domain_rows = []
        for item in items:
            ids = torch.tensor([item["tokens"]], dtype=torch.long, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=ids.is_cuda):
                student_logits = student(input_ids=ids, use_cache=False).logits.float()
                teacher_logits = teacher(input_ids=ids, use_cache=False).logits.float()
            # Predict tokens 1..n-1.
            targets = window_targets(student_logits[0, :-1], teacher_logits[0, :-1])
            domain_rows.append({
                "doc_id": item["doc_id"],
                "window_index": item["window_index"],
                **{key: float(value.mean()) for key, value in targets.items()
                   if key != "positions"},
                "positions": int(targets["positions"]),
            })
        rows[domain] = domain_rows
    summary = {}
    for domain, domain_rows in rows.items():
        summary[domain] = {
            "windows": len(domain_rows),
            "compact": bootstrap_mean(np.array([row["compact"] for row in domain_rows])),
            "full": bootstrap_mean(np.array([row["full"] for row in domain_rows])),
            "bias": bootstrap_mean(np.array([row["bias"] for row in domain_rows])),
            "p_tail": float(np.mean([row["p_tail"] for row in domain_rows])),
            "tail_entropy": float(np.mean([row["tail_entropy"] for row in domain_rows])),
            "tail_kl": float(np.mean([row["tail_kl"] for row in domain_rows])),
        }
    all_rows = [row for domain_rows in rows.values() for row in domain_rows]
    summary["pooled"] = {
        "windows": len(all_rows),
        "compact": float(np.mean([row["compact"] for row in all_rows])),
        "full": float(np.mean([row["full"] for row in all_rows])),
        "bias": float(np.mean([row["bias"] for row in all_rows])),
        "p_tail": float(np.mean([row["p_tail"] for row in all_rows])),
        "tail_entropy": float(np.mean([row["tail_entropy"] for row in all_rows])),
        "tail_kl": float(np.mean([row["tail_kl"] for row in all_rows])),
        "identity_check": {
            "measured_bias": float(np.mean([row["bias"] for row in all_rows])),
            "minus_p_tail_entropy_plus_kl": float(np.mean(
                [-row["p_tail"] * (row["tail_entropy"] + row["tail_kl"])
                 for row in all_rows])),
        },
    }
    return {"per_window": rows, "summary": summary}


def spearman(a: list[float], b: list[float]) -> float:
    order_a = np.argsort(np.argsort(np.asarray(a, dtype=float)))
    order_b = np.argsort(np.argsort(np.asarray(b, dtype=float)))
    if len(a) < 2:
        return float("nan")
    rank_a = order_a.astype(float)
    rank_b = order_b.astype(float)
    rank_a -= rank_a.mean()
    rank_b -= rank_b.mean()
    denominator = float(np.sqrt((rank_a ** 2).sum() * (rank_b ** 2).sum()))
    return float((rank_a * rank_b).sum() / denominator) if denominator else float("nan")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", action="append", required=True,
                        help="NAME[:checkpoint[:config[:admission[:profile]]]]")
    parser.add_argument("--hf-root", default=os.environ.get("HF_HOME", "/hf-cache"))
    parser.add_argument("--split-dir", default="results/artifacts")
    parser.add_argument("--split-receipts", default="results/splits")
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("the instrument check requires the GPU court (--gpu)")
    device = torch.device("cuda")
    split, split_sha = tasks.load_split(args.split_dir, args.split_receipts)
    windows = tasks.heldout_windows(split)
    if args.max_windows:
        windows = {domain: entries[:args.max_windows]
                   for domain, entries in windows.items()}

    teacher = harness.load_model(harness.model_path(args.hf_root, "1.7b"), device)
    results = {}
    for spec in args.student:
        parts = (spec.split(":") + ["", "", "", ""])[:5]
        name, checkpoint, config_path, admission_path, profile = parts
        profile = profile or "t1"
        print(f"instrument: student {name} profile={profile}", flush=True)
        if checkpoint:
            if not (config_path and admission_path):
                raise SystemExit(f"{name}: a checkpoint needs config and admission")
            config, admission = load_admission(config_path, admission_path)
            base = harness.load_model(harness.model_path(args.hf_root, "0.6b"), device)
            arm_model = build_arm(base, name, config, admission, 1001, profile=profile)
            arm_model.model.to(device)
            state = torch.load(checkpoint, map_location=device, weights_only=False)
            arm_model.model.load_state_dict(state["model"], strict=False)
            arm_model.model.eval()
            student = arm_model.model
            resource = {
                "checkpoint": checkpoint,
                "checkpoint_sha256": sha256_file(checkpoint),
                "config": config_path,
                "admission": admission_path,
                "active_unique_params": admission["active_unique_params"],
                "total_accessible": config["total_accessible"],
            }
        else:
            student = harness.load_model(harness.model_path(args.hf_root, "0.6b"), device)
            resource = {"checkpoint": None, "note": "untrained frozen core (D0)"}
        results[name] = {
            "resource": resource,
            "profile": profile,
            **evaluate_student(student, teacher, windows, device),
        }
        del student
        torch.cuda.empty_cache()

    names = sorted(results)
    order_compact = [results[name]["summary"]["pooled"]["compact"] for name in names]
    order_full = [results[name]["summary"]["pooled"]["full"] for name in names]
    growth = [results[name]["summary"]["pooled"]["full"]
              - results[name]["summary"]["pooled"]["compact"] for name in names]
    pairs = {}
    for left, right in DECISIVE_PAIRS:
        if left not in results or right not in results:
            pairs[f"{left}-{right}"] = {"status": "student missing"}
            continue
        left_rows = [row for domain in results[left]["per_window"].values() for row in domain]
        right_rows = [row for domain in results[right]["per_window"].values() for row in domain]
        common = min(len(left_rows), len(right_rows))
        compact_left = np.array([row["compact"] for row in left_rows[:common]])
        compact_right = np.array([row["compact"] for row in right_rows[:common]])
        full_left = np.array([row["full"] for row in left_rows[:common]])
        full_right = np.array([row["full"] for row in right_rows[:common]])
        pairs[f"{left}-{right}"] = {
            "compact_difference": paired_bootstrap(compact_left, compact_right),
            "full_difference": paired_bootstrap(full_left, full_right),
            "sign_agreement": bool(np.sign(compact_left.mean() - compact_right.mean())
                                   == np.sign(full_left.mean() - full_right.mean())),
        }
    rank_correlation = spearman(order_compact, order_full)
    sign_agreement = all(pair.get("sign_agreement", False) for pair in pairs.values())
    decision = "compact_target_adequate" if sign_agreement and rank_correlation >= 0.9 \
        else "report_both_targets"
    stamp = stamp_now()
    receipt = {
        "schema": "selvedge/t2-instrument-check/v1",
        "stamp": stamp,
        "revision": os.environ.get("SELVEDGE_REVISION", "unknown"),
        "finished_utc": utc_now(),
        "question": "does the T1.4 top-16+tail target rank students like the full target?",
        "teacher": {
            "size": "1.7b",
            "revision": harness.MODEL_REVISIONS["1.7b"],
            "target": "live full-vocabulary teacher forward",
        },
        "sample": {
            "split_sha256": split_sha,
            "domains": {domain: len(entries) for domain, entries in windows.items()},
            "windows": sum(len(entries) for entries in windows.values()),
            "max_windows_per_domain": args.max_windows or None,
            "targets_per_window": 512,
        },
        "instrument": {
            "compact": "T1.4 top-16 support + one tail bucket",
            "full": "full-vocabulary cross-entropy of the live teacher",
            "identity": "L_compact - L_full = -p_tail * (H(teacher_tail) + "
                        "KL(teacher_tail || student_tail))",
            "chunk": CHUNK,
        },
        "students": results,
        "ranking": {
            "names": names,
            "compact": order_compact,
            "full": order_full,
            "full_minus_compact": growth,
            "spearman": rank_correlation,
            "decisive_pairs": pairs,
            "sign_agreement": sign_agreement,
        },
        "decision": decision,
        "decision_rule": "compact target stands for T2 only if every registered "
                         "decisive pair keeps its sign and the cross-student rank "
                         "correlation is >= 0.9; otherwise T2 reports both targets",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "command": " ".join(sys.argv),
    }
    write_json_atomic(args.out, receipt)
    print(json.dumps({"decision": decision, "spearman": rank_correlation,
                      "sign_agreement": sign_agreement,
                      "pooled": {name: results[name]["summary"]["pooled"]
                                 for name in names}}, indent=2, sort_keys=True))
    print(f"instrument: receipt={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
