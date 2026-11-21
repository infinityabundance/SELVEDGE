#!/usr/bin/env python3
"""S64: build the preference pairs and cache the frozen-reference log-probs.

Two passes, each of which is its own receipted job:

``teacher``
    Turn the corrected K=8 rollout ledger and its failed-continuation sidecar
    into preference pairs. The chosen side is a **verified-correct teacher
    continuation from the student's own prefix**; the rejected side is the
    student's own failed continuation from that same prefix. The registration
    (`docs/ws19-stage2-correction-s63-preregistration-v1.json`) requires the
    prefix to be shared exactly, so the builder fails closed on a token
    mismatch instead of pairing two different prompts.

    A teacher solution is read from an existing verified teacher corpus
    (`--teacher-corpus`, e.g. the S58 stage-2a accepted rows) when that corpus
    already carries the same problem id *and the same prompt token ids*. When
    it does not, `--generate-missing --teacher-dir DIR` samples the teacher
    from the student's prefix with the shared tokenizer. Every chosen
    continuation is re-verified with `frontier.rft.verify` against the train
    label whether it came from the corpus or from the teacher; a corpus row
    that no longer verifies is counted, never used.

    On-policy pairs carry the registered weight ``w(p) = p(1-p)``. A problem
    with ``p=0`` has no accepted student row, so its failed continuation and
    its verified teacher solution form a **teacher-guided recovery** pair with
    the mean positive on-policy weight. Both arms are written, separately.

``reference``
    Cache the frozen post-trained 0.8B's per-token log-probabilities for the
    chosen and rejected actions of every pair. This is the only place a
    reference model runs. No teacher weights are co-resident with the
    reference, and no teacher distribution is cached or claimed. The cache is
    the per-token vector; the registered token-length normalisation (mean) is
    applied in the loss, and both mean and sum are reported as diagnostics.

Neither pass writes decoded failed text. The rejected side is token ids only.

Usage (GPU, one job at a time, never co-resident with the rollout jobs)
----------------------------------------------------------------------
    SELVEDGE_MEM=14g scripts/court --gpu oracle -- python3 \
      scripts/s64_preference_data.py teacher \
        --ledger results/artifacts/stage2/s64-rollouts/gsm8k-shard0-ledger.jsonl \
        --rejected results/artifacts/stage2/s64-rollouts/gsm8k-shard0-rejected.jsonl \
        --teacher-corpus results/artifacts/stage2/s58-combined/stage2a-accepted-combined.jsonl \
        --tokenizer-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364... \
        --teacher-dir /hf-cache/models/Qwen/Qwen3.5-2B/15852e8c... \
        --generate-missing --vram-limit-gib 13.0 \
        --exclude-ids results/artifacts/stage2/s58-combined/stage2a-combined-holdout.jsonl \
        --out-dir results/artifacts/stage2/s64-preference --label pairs

    SELVEDGE_MEM=12g scripts/court --gpu oracle -- python3 \
      scripts/s64_preference_data.py reference \
        --pairs results/artifacts/stage2/s64-preference/pairs-pairs.jsonl \
        --reference-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364... \
        --tokenizer-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364... \
        --out-dir results/artifacts/stage2/s64-preference \
        --label pairs-with-reference
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "oracle"))
sys.path.insert(0, HERE)

from frontier import preference, rft  # noqa: E402

SCHEMA_PAIRS = "selvedge/s64-preference-pairs/v1"
SCHEMA_REFERENCE = "selvedge/s64-preference-reference-cache/v1"
GIB = float(1 << 30)


# ---------------------------------------------------------------------------
# small shared helpers (the repo keeps these copies local to each script)


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def revision() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def cgroup_memory() -> dict:
    values = {}
    for name in ("memory.max", "memory.peak", "memory.swap.peak", "memory.events"):
        path = f"/sys/fs/cgroup/{name}"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                values[name.replace(".", "_")] = handle.read().strip()
    return values


def _vram_guard(limit_gib: float) -> None:
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("s64-data: CUDA required; run through scripts/court --gpu")
    free_bytes, _total = torch.cuda.mem_get_info()
    if limit_gib * GIB > free_bytes:
        raise SystemExit(
            f"s64-data: vram limit {limit_gib} GiB exceeds free "
            f"{free_bytes / GIB:.2f} GiB")


# ---------------------------------------------------------------------------
# inputs


def index_ledgers(paths: list[str]) -> tuple[dict[str, dict], dict]:
    """One ledger entry per problem id, with a duplicate-id refusal."""
    entries: dict[str, dict] = {}
    duplicates = []
    for path in paths:
        for row in load_jsonl(_resolve(path)):
            problem_id = str(row.get("id"))
            if problem_id in entries:
                duplicates.append(problem_id)
                continue
            entries[problem_id] = row
    if duplicates:
        raise SystemExit(
            "s64-data: duplicate ledger ids across shards: "
            f"{sorted(set(duplicates))[:5]}")
    return entries, {"ledger_files": paths, "problems": len(entries)}


def index_rejected(paths: list[str]) -> tuple[dict[str, dict], dict]:
    """One failed continuation per problem id.

    The rollout writer already prefers a non-truncated failure within a shard.
    Across shards the same preference is applied again so a truncated side can
    never displace a clean one.
    """
    entries: dict[str, dict] = {}
    for path in paths:
        for row in load_jsonl(_resolve(path)):
            problem_id = str(row.get("id"))
            previous = entries.get(problem_id)
            if previous is None or (previous.get("truncated")
                                    and not row.get("truncated")):
                entries[problem_id] = row
    return entries, {"rejected_files": paths, "problems": len(entries)}


def index_teacher_corpus(paths: list[str]) -> tuple[dict[str, list[dict]], dict]:
    """All candidate teacher continuations per problem id, shortest first.

    Rows without ``solution_ids`` are ignored. A row may be a teacher row from
    the S58 accepted corpus (``kind`` absent) or an examples-stream row that is
    explicitly ``kind == "teacher"``; replay/control rows are never used.
    """
    corpus: dict[str, list[dict]] = {}
    skipped_kind = 0
    for path in paths:
        for row in load_jsonl(_resolve(path)):
            kind = row.get("kind")
            if kind is not None and kind != "teacher":
                skipped_kind += 1
                continue
            if not row.get("solution_ids"):
                skipped_kind += 1
                continue
            corpus.setdefault(str(row.get("id")), []).append(row)
    for rows in corpus.values():
        rows.sort(key=lambda row: len(row["solution_ids"]))
    return corpus, {"teacher_corpus_files": paths,
                    "problems_with_teacher": len(corpus),
                    "rows_skipped": skipped_kind}


def _verify_continuation(ids, target: str, dataset: str, decode_fn,
                         verify_fn) -> dict:
    text = decode_fn(list(ids))
    return verify_fn(text, target, dataset)


# ---------------------------------------------------------------------------
# the pair builder (importable and CPU-testable)


def build_pairs(ledger: dict[str, dict], rejected: dict[str, dict],
                teacher_corpus: dict[str, list[dict]], *,
                decode_fn, verify_fn=rft.verify,
                generate_fn=None, require_terminated: bool = True
                ) -> tuple[list[preference.PreferencePair],
                           list[preference.PreferencePair], dict]:
    """Build the on-policy and recovery pairs and their diagnostics.

    ``generate_fn(prompt_ids) -> (ids, terminated, truncated)`` is optional and
    called only when the corpus has no usable teacher row for a problem.
    """
    positive_weights = [float(entry.get("weight_p1mp", 0.0))
                        for entry in ledger.values()]
    on_policy: list[preference.PreferencePair] = []
    recovery_candidates: list[tuple[str, list, list, list]] = []
    diagnostics = {
        "ledger_problems": len(ledger),
        "pairs": 0,
        "recovery_pairs": 0,
        "missing_rejected": [],
        "p1_without_failure": [],
        "missing_teacher": [],
        "prefix_mismatch_rejected": [],
        "prefix_mismatch_teacher": [],
        "teacher_unverified": [],
        "teacher_generated": [],
        "teacher_from_corpus": [],
        "teacher_generation_failed": [],
        "identity_pair": [],
        "verdicts": {},
    }

    for problem_id in sorted(ledger):
        entry = ledger[problem_id]
        dataset = entry.get("dataset")
        target = entry.get("target")
        if target is None:
            raise SystemExit(
                f"s64-data: ledger row {problem_id!r} has no target")
        p = float(entry.get("p", 0.0))
        weight = float(entry.get("weight_p1mp", 0.0))
        prompt_ids = entry.get("prompt_ids")
        if not prompt_ids:
            raise SystemExit(
                f"s64-data: ledger row {problem_id!r} has no prompt_ids; the "
                "shared-prefix pair cannot be built")

        failed = rejected.get(problem_id)
        if failed is None:
            if p >= 1.0:
                diagnostics["p1_without_failure"].append(problem_id)
            else:
                diagnostics["missing_rejected"].append(problem_id)
            continue
        failed_ids = failed.get("solution_ids") or []
        if not failed_ids:
            diagnostics["missing_rejected"].append(problem_id)
            continue
        try:
            preference.assert_shared_prefix(prompt_ids,
                                            failed.get("prompt_ids") or [])
        except ValueError:
            diagnostics["prefix_mismatch_rejected"].append(problem_id)
            continue

        chosen_ids = None
        chosen_source = None
        for row in teacher_corpus.get(problem_id, []):
            try:
                preference.assert_shared_prefix(prompt_ids,
                                                row.get("prompt_ids") or [])
            except ValueError:
                diagnostics["prefix_mismatch_teacher"].append(problem_id)
                continue
            result = _verify_continuation(row["solution_ids"], target, dataset,
                                          decode_fn, verify_fn)
            diagnostics["verdicts"][result["verdict"]] = (
                diagnostics["verdicts"].get(result["verdict"], 0) + 1)
            if result["verdict"] == "correct":
                chosen_ids = list(row["solution_ids"])
                chosen_source = "corpus"
                break

        if chosen_ids is None and generate_fn is not None:
            generated, terminated, truncated = generate_fn(prompt_ids)
            if generated and (terminated or not require_terminated) and not truncated:
                result = _verify_continuation(generated, target, dataset,
                                              decode_fn, verify_fn)
                diagnostics["verdicts"][result["verdict"]] = (
                    diagnostics["verdicts"].get(result["verdict"], 0) + 1)
                if result["verdict"] == "correct":
                    chosen_ids = list(generated)
                    chosen_source = "generated"
                else:
                    diagnostics["teacher_generation_failed"].append(problem_id)
            else:
                diagnostics["teacher_generation_failed"].append(problem_id)

        if chosen_ids is None:
            diagnostics["missing_teacher"].append(problem_id)
            continue
        if list(chosen_ids) == list(failed_ids):
            diagnostics["identity_pair"].append(problem_id)
            continue
        if chosen_source == "corpus":
            diagnostics["teacher_from_corpus"].append(problem_id)
        elif chosen_source == "generated":
            diagnostics["teacher_generated"].append(problem_id)

        pair = preference.make_pair(problem_id, prompt_ids, chosen_ids,
                                    failed_ids, weight=weight)
        pair = preference.PreferencePair(
            problem_id=pair.problem_id, prompt_ids=pair.prompt_ids,
            chosen_ids=pair.chosen_ids, rejected_ids=pair.rejected_ids,
            weight=pair.weight, provenance="on-policy")
        on_policy.append(pair)

        if p == 0.0:
            recovery_candidates.append(
                (problem_id, list(prompt_ids), list(chosen_ids),
                 list(failed_ids)))

    # The recovery weight is the mean positive p(1-p) of the on-policy pairs
    # that were actually built, not of ledger rows whose teacher or failed
    # side was missing. A problem that never became a pair cannot set the
    # weight of a pair that does.
    positive_weights = [pair.weight for pair in on_policy if pair.weight > 0]
    recovery = [
        preference.recovery_pair(problem_id, prompt, chosen, failed, 0.0,
                                 positive_weights)
        for problem_id, prompt, chosen, failed in recovery_candidates]

    diagnostics["pairs"] = len(on_policy)
    diagnostics["recovery_pairs"] = len(recovery)
    diagnostics["matched_recovery_weight"] = preference.recovery_weight(
        positive_weights)
    diagnostics["basis"] = (
        "chosen = verified-correct teacher continuation from the student "
        "prefix; rejected = the student's own failed continuation from the "
        "same prefix; on-policy weight = p(1-p); p=0 recovery weight = mean "
        "positive p(1-p)")
    return on_policy, recovery, diagnostics


# ---------------------------------------------------------------------------
# pair serialisation (the reference pass consumes this)


def pair_row(pair: preference.PreferencePair) -> dict:
    return {
        "problem_id": pair.problem_id,
        "prompt_ids": list(pair.prompt_ids),
        "chosen_ids": list(pair.chosen_ids),
        "rejected_ids": list(pair.rejected_ids),
        "weight": float(pair.weight),
        "provenance": pair.provenance,
        "chosen_tokens": len(pair.chosen_ids),
        "rejected_tokens": len(pair.rejected_ids),
    }


def pair_from_row(row: dict) -> preference.PreferencePair:
    return preference.PreferencePair(
        problem_id=str(row["problem_id"]),
        prompt_ids=tuple(int(x) for x in row["prompt_ids"]),
        chosen_ids=tuple(int(x) for x in row["chosen_ids"]),
        rejected_ids=tuple(int(x) for x in row["rejected_ids"]),
        weight=float(row.get("weight", 1.0)),
        chosen_reference_logprobs=(tuple(row["chosen_reference_logprobs"])
                                   if row.get("chosen_reference_logprobs")
                                   else None),
        rejected_reference_logprobs=(tuple(row["rejected_reference_logprobs"])
                                     if row.get("rejected_reference_logprobs")
                                     else None),
        provenance=str(row.get("provenance", "on-policy")))


def write_pairs(path: str, pairs: list[preference.PreferencePair],
                *, reference: bool = False) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for pair in pairs:
            row = pair_row(pair)
            if reference:
                row["chosen_reference_logprobs"] = list(
                    pair.chosen_reference_logprobs or [])
                row["rejected_reference_logprobs"] = list(
                    pair.rejected_reference_logprobs or [])
                row["chosen_reference_mean"] = float(
                    preference.reference_logprob(
                        pair.chosen_reference_logprobs, "mean"))
                row["rejected_reference_mean"] = float(
                    preference.reference_logprob(
                        pair.rejected_reference_logprobs, "mean"))
                row["chosen_reference_sum"] = float(
                    preference.reference_logprob(
                        pair.chosen_reference_logprobs, "sum"))
                row["rejected_reference_sum"] = float(
                    preference.reference_logprob(
                        pair.rejected_reference_logprobs, "sum"))
            handle.write(json.dumps(row, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# teacher subcommand


class LazyTeacher:
    """Load the 2B teacher only if the corpus leaves a problem uncovered."""

    def __init__(self, model_dir: str, tokenizer_dir: str, device: str,
                 max_new_tokens: int, temperature: float, top_p: float,
                 seed: int):
        self.model_dir = model_dir
        self.tokenizer_dir = tokenizer_dir
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from frontier import arms
        from s55_stage2_data import load_tokenizer
        self._tokenizer = load_tokenizer(self.tokenizer_dir)
        self._model = arms.load_model(self.model_dir, dtype=torch.bfloat16,
                                      device=self.device)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def __call__(self, prompt_ids):
        import torch
        self._load()
        eos = int(self._tokenizer.eos_token_id)
        pad = int(self._tokenizer.pad_token_id)
        input_ids = torch.tensor([list(prompt_ids)], dtype=torch.long,
                                 device=self.device)
        with torch.inference_mode():
            output = self._model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=self.max_new_tokens, do_sample=True,
                temperature=self.temperature, top_p=self.top_p,
                pad_token_id=pad, eos_token_id=eos)
        ids = output[0, input_ids.shape[1]:].tolist()
        terminated = False
        while ids and ids[-1] == pad:
            ids.pop()
        if ids and ids[-1] == eos:
            terminated = True
            ids.pop()
        truncated = (not terminated and len(ids) >= self.max_new_tokens)
        return ids, terminated, truncated


def run_teacher(args) -> int:
    ledger, ledger_report = index_ledgers(args.ledger)
    rejected, rejected_report = index_rejected(args.rejected)
    corpus, corpus_report = index_teacher_corpus(args.teacher_corpus or [])

    excluded = set()
    for path in args.exclude_ids or []:
        for row in load_jsonl(_resolve(path)):
            if row.get("id"):
                excluded.add(str(row["id"]))
    if excluded:
        removed = [problem_id for problem_id in ledger if problem_id in excluded]
        for problem_id in removed:
            del ledger[problem_id]
        ledger_report["excluded_holdout"] = len(removed)

    from s55_stage2_data import load_tokenizer
    tokenizer = load_tokenizer(args.tokenizer_dir)

    generate_fn = None
    if args.generate_missing:
        if not args.teacher_dir:
            raise SystemExit("s64-data: --generate-missing needs --teacher-dir")
        generate_fn = LazyTeacher(
            args.teacher_dir, args.tokenizer_dir, args.device,
            args.max_new_tokens, args.temperature, args.top_p, args.seed)

    on_policy, recovery, diagnostics = build_pairs(
        ledger, rejected, corpus, decode_fn=tokenizer.decode,
        verify_fn=rft.verify, generate_fn=generate_fn)

    if not on_policy:
        raise SystemExit("s64-data: no on-policy pairs were built; refusing to "
                         "write an empty preference corpus")

    out_dir = _resolve(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    pairs_path = os.path.join(out_dir, f"{args.label}-pairs.jsonl")
    recovery_path = os.path.join(out_dir, f"{args.label}-recovery.jsonl")
    receipt_path = os.path.join(out_dir, f"{args.label}-receipt.json")
    write_pairs(pairs_path, on_policy)
    write_pairs(recovery_path, recovery)

    receipt = {
        "schema": SCHEMA_PAIRS,
        "kind": "preference-pairs",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "registration": (
            "docs/ws19-stage2-correction-s63-preregistration-v1.json"),
        "command": sys.argv,
        "inputs": {
            "ledger": ledger_report,
            "rejected": rejected_report,
            "teacher_corpus": corpus_report,
            "excluded_holdout": len(excluded),
        },
        "student_prefix": (
            "the ledger's tokenised prompt_ids; the teacher corpus row is used "
            "only when its prompt_ids are token-identical, and a generated "
            "teacher continuation is conditioned on those exact ids"),
        "teacher": {
            "dir": args.teacher_dir,
            "generate_missing": bool(args.generate_missing),
            "sampling": {
                "temperature": args.temperature, "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens, "seed": args.seed,
                "stop": "EOS; a truncated or loop-free-but-capped continuation "
                        "is not a chosen target"},
            "verification": "frontier.rft.verify against the ledger target, "
                            "re-applied to corpus rows and generated rows alike",
        },
        "pairs": {
            "path": os.path.relpath(pairs_path, ROOT),
            "sha256": sha256_file(pairs_path),
            "rows": len(on_policy),
            "weight_basis": "p(1-p)",
            "normalisation": preference.REGISTERED_NORMALIZATION,
        },
        "recovery": {
            "path": os.path.relpath(recovery_path, ROOT),
            "sha256": sha256_file(recovery_path),
            "rows": len(recovery),
            "weight_basis": "mean positive p(1-p) of the on-policy problems",
        },
        "diagnostics": diagnostics,
        "preference_processed_tokens_per_pass": (
            preference.preference_processed_tokens(on_policy)),
        "cgroup_memory": cgroup_memory(),
        "claims": {
            "chosen_is_teacher_continuation": True,
            "rejected_is_student_failed_continuation": True,
            "failed_text_stored": False,
            "no_teacher_distribution_cached": True,
        },
    }
    write_json(receipt_path, receipt)
    print(f"s64-data: {len(on_policy)} on-policy pairs, {len(recovery)} "
          f"recovery pairs -> {pairs_path}")
    print(f"s64-data: receipt {receipt_path}")
    return 0


# ---------------------------------------------------------------------------
# reference subcommand


def run_reference(args) -> int:
    pairs: list[preference.PreferencePair] = []
    sources = []
    for path in args.pairs:
        rows = load_jsonl(_resolve(path))
        sources.append({"path": path,
                        "sha256": sha256_file(_resolve(path)),
                        "rows": len(rows)})
        pairs.extend(pair_from_row(row) for row in rows)
    if not pairs:
        raise SystemExit("s64-data: no pairs to cache")
    if args.limit is not None:
        pairs = pairs[:args.limit]

    _vram_guard(args.vram_limit_gib)
    import torch
    from frontier import arms
    from s55_stage2_data import load_tokenizer

    device = "cuda"
    tokenizer = load_tokenizer(args.tokenizer_dir)
    del tokenizer  # only the ids are needed; decode is never used here
    model = arms.load_model(args.reference_dir, dtype=torch.bfloat16,
                            device=device)
    reference_checkpoint = None
    if args.reference_checkpoint:
        from frontier import full

        checkpoint_path = _resolve(args.reference_checkpoint)
        payload = torch.load(checkpoint_path, map_location="cpu",
                             weights_only=False)
        if payload.get("arm") not in (None, "full"):
            raise SystemExit(
                "s64-data: --reference-checkpoint is not a full-arm checkpoint "
                f"(arm={payload.get('arm')!r})")
        full.load_full_state_into(model, payload["arm_state"])
        model.eval()
        reference_checkpoint = {
            "path": args.reference_checkpoint,
            "sha256": sha256_file(checkpoint_path),
            "step": payload.get("step"),
            "tokens_seen": payload.get("tokens_seen"),
            "arm": "full",
        }
        print(f"s64-data: reference checkpoint {args.reference_checkpoint} "
              f"step={payload.get('step')}", flush=True)
    torch.cuda.reset_peak_memory_stats()

    cached = []
    started = time.time()
    for index, pair in enumerate(pairs):
        cached.append(preference.cache_reference_logprobs(
            model, pair, device, args.normalization))
        if index % 200 == 0:
            print(f"s64-data: reference {index}/{len(pairs)}", flush=True)

    wall = time.time() - started
    out_dir = _resolve(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.label}.jsonl")
    write_pairs(out_path, cached, reference=True)
    receipt_path = os.path.join(out_dir, f"{args.label}-receipt.json")

    chosen_tokens = sum(len(pair.chosen_ids) for pair in cached)
    rejected_tokens = sum(len(pair.rejected_ids) for pair in cached)
    peak_alloc = torch.cuda.max_memory_allocated() / GIB
    peak_reserved = torch.cuda.max_memory_reserved() / GIB
    receipt = {
        "schema": SCHEMA_REFERENCE,
        "kind": "frozen-reference-logprob-cache",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "registration": (
            "docs/ws19-stage2-correction-s63-preregistration-v1.json"),
        "command": sys.argv,
        "reference": {
            "dir": args.reference_dir,
            "checkpoint": reference_checkpoint,
            "identity": (
                "frozen soup champion loaded from --reference-checkpoint; "
                "pi_ref is the policy the student is initialised from"
                if reference_checkpoint else
                "frozen post-trained Qwen/Qwen3.5-0.8B; pi_0"),
            "normalisation": args.normalization,
            "registered_normalisation": preference.REGISTERED_NORMALIZATION,
            "teacher_co_resident": False,
            "what_is_cached": "per-action log pi_0(a|h) for the fixed chosen "
                              "and rejected actions, one vector per side",
        },
        "inputs": sources,
        "pairs": len(cached),
        "tokens": {"chosen": chosen_tokens, "rejected": rejected_tokens},
        "output": {
            "path": os.path.relpath(out_path, ROOT),
            "sha256": sha256_file(out_path),
        },
        "wall_seconds": wall,
        "tokens_per_second": ((chosen_tokens + rejected_tokens) / wall
                              if wall > 0 else None),
        "peak_allocated_gib": peak_alloc,
        "peak_reserved_gib": peak_reserved,
        "cgroup_memory": cgroup_memory(),
        "claims": {
            "reverse_kl_not_claimed": True,
            "teacher_distribution_not_cached": True,
        },
    }
    write_json(receipt_path, receipt)
    print(f"s64-data: cached {len(cached)} pairs -> {out_path}")
    print(f"s64-data: receipt {receipt_path}")
    return 0


# ---------------------------------------------------------------------------
# CLI


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    teacher = sub.add_parser("teacher", help="build the preference pairs")
    teacher.add_argument("--ledger", action="append", required=True)
    teacher.add_argument("--rejected", action="append", required=True)
    teacher.add_argument("--teacher-corpus", action="append", default=[])
    teacher.add_argument("--tokenizer-dir", required=True)
    teacher.add_argument("--teacher-dir", default=None)
    teacher.add_argument("--generate-missing", action="store_true")
    teacher.add_argument("--exclude-ids", action="append", default=[])
    teacher.add_argument("--temperature", type=float, default=0.8)
    teacher.add_argument("--top-p", type=float, default=0.95)
    teacher.add_argument("--max-new-tokens", type=int, default=2048)
    teacher.add_argument("--seed", type=int, default=6400)
    teacher.add_argument("--device", default="cuda")
    teacher.add_argument("--out-dir", required=True)
    teacher.add_argument("--label", required=True)
    teacher.set_defaults(func=run_teacher)

    reference = sub.add_parser("reference",
                               help="cache the frozen-reference log-probs")
    reference.add_argument("--pairs", action="append", required=True)
    reference.add_argument("--reference-dir", required=True)
    reference.add_argument("--reference-checkpoint", default=None,
                           help="a stage-2 full-arm checkpoint loaded over "
                                "--reference-dir; the receipt records its "
                                "path, sha256 and step")
    reference.add_argument("--tokenizer-dir", required=True)
    reference.add_argument("--normalization", default="mean",
                           choices=preference.NORMALIZATIONS)
    reference.add_argument("--limit", type=int, default=None)
    reference.add_argument("--vram-limit-gib", type=float, default=12.0)
    reference.add_argument("--out-dir", required=True)
    reference.add_argument("--label", required=True)
    reference.set_defaults(func=run_reference)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
