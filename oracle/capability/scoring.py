#!/usr/bin/env python3
"""Likelihood-based candidate selection: the S30 structural fix (instrument v2).

The v1 selection formats asked a completion model for a letter. The S30 forensics
showed what that measures: the arm repeats the instruction's example letter (or a
fixed position prior) instead of choosing, so the answer is constant across items
and the format cannot discriminate. Free-form generation has the same failure.

This module removes free-form generation from the selection formats by
construction. For every candidate it computes the model's own log-probability of
that candidate's *content* as a continuation of the prompt, and the choice is the
arg-max. There is no sampled or greedy token to collapse:

* the sum of token log-probabilities and the length-normalised mean are both
  recorded, and both rankings are reported (the registered primary is the mean,
  because a sum rewards long candidates structurally);
* a letter-score ranking — the log-probability of `` A``/`` B``/`` C`` given the
  prompt — is recorded as an **unscored diagnostic**, because it is the direct
  test of position bias;
* the scored content is the candidate text alone. The label, the position, the
  reference flag and any checker output are never passed to the model, and the
  correct candidate's position varies across items (recorded per item and
  summarised), so a position prior cannot be a systematic advantage.

Nothing here decides a capability claim. The primary metric for a selection item
remains *the chosen candidate passes the executable checker*; the ranking margin
is a diagnostic and is never substituted for it.
"""

from __future__ import annotations

import math

from . import formats

#: The registered ranking modes. `content-mean` is primary; `letter-mean` is a
#: diagnostic that is scored only if a run is registered for it.
CONTENT_MEAN = "content-mean"
CONTENT_SUM = "content-sum"
LETTER_MEAN = "letter-mean"
LETTER_SUM = "letter-sum"
RANKING_MODES = (CONTENT_MEAN, CONTENT_SUM, LETTER_MEAN, LETTER_SUM)
PRIMARY_MODE = CONTENT_MEAN
DIAGNOSTIC_MODES = (LETTER_MEAN, LETTER_SUM)

#: The prompt variant the selection instrument renders. The registered v1
#: generation prompt shows an example letter; the selection prompt does not, and
#: it says what is scored rather than what to type. The variant name is carried in
#: the receipt so the check stage recomputes the same prompt hash.
PROMPT_VARIANT = "selection-v1"

#: A scored sequence may not be silently truncated (AGENTS.md directive 9). The
#: calibration prompts are under 1000 tokens and a candidate under 200, so this
#: bound refuses a pathological item instead of scoring a prefix of it.
MAX_SCORED_TOKENS = 4096

SCHEMA_REGISTRATION = "selvedge/capability-calibration-preregistration/v2"


class ScoringError(RuntimeError):
    """The scorer cannot produce an exact score; refuse rather than approximate."""


def selection_contents(task) -> list[dict]:
    """(label, scored content) for every candidate, in prompt order.

    The content is exactly what the prompt shows for that candidate, without the
    positional label. For a patch item whose target is a whole file the content is
    the file body, so the same rule covers both target kinds.
    """
    from . import forensics

    return forensics.candidate_contents(task)


def letter_continuations(labels: list[str]) -> list[str]:
    """The continuation each letter is scored as, identical for every item."""
    return [f" {label}" for label in labels]


def choose(scores: list[dict], labels: list[str], field: str) -> dict:
    """Arg-max over one score field, with the margin and a deterministic tie rule.

    `scores[i]` is a dict of {"sum": float, "mean": float}. A tie is broken by
    prompt order (the earlier label wins) and flagged, so a tie can never be
    reported as a decision.
    """
    if len(scores) != len(labels):
        raise ScoringError("a score list and a label list must be the same length")
    if not scores:
        raise ScoringError("no candidates to choose between")
    values = [(float(score[field]), index) for index, score in enumerate(scores)]
    ranked = sorted(values, key=lambda pair: (-pair[0], pair[1]))
    best_value, best_index = ranked[0]
    second_value, second_index = ranked[1] if len(ranked) > 1 else (float("-inf"), None)
    tie = len(ranked) > 1 and math.isclose(best_value, second_value, rel_tol=0.0,
                                           abs_tol=1e-9)
    return {
        "field": field,
        "choice": labels[best_index],
        "choice_index": best_index,
        "score": best_value,
        "runner_up": labels[second_index] if second_index is not None else None,
        "runner_up_score": (None if second_index is None else second_value),
        "margin": (None if second_index is None else best_value - second_value),
        "tie": tie,
        "ranking": [{"label": labels[index], "score": value}
                    for value, index in ranked],
        "tie_rule": "prompt order breaks a tie; a flagged tie is never a decision",
    }


def content_continuation(content: str) -> str:
    """The exact text appended to the selection prompt for a candidate.

    Two newlines, so the continuation starts a block the way the prompt's own
    sections do. The separator is fixed for every candidate, so it cannot favour
    one of them.
    """
    return "\n\n" + content.strip()


def score_with_model(model, tokenizer, prompt: str, texts: list[str], device,
                     chunk: int = 64) -> list[dict]:
    """Exact continuation log-probabilities, one forward pass per text.

    The vocabulary head is applied only to the continuation's positions (the same
    streamed-head path the registered evaluation instrument uses), so the peak
    memory is the model plus one hidden-state pass, not ``positions x vocab``.

    Returns, per text, the token count, the exact sum of token log-probs and the
    length-normalised mean, plus the per-token log-probs (a list of small floats;
    the receipt keeps them so a reader can recompute either aggregate).
    """
    import torch

    from .adapters import model_path_dirs  # noqa: F401 - keeps the import graph explicit

    import sys
    for directory in model_path_dirs():
        if directory not in sys.path:
            sys.path.insert(0, directory)
    from harness import final_hidden_states, logits_chunk

    prefix = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
    prefix_length = int(prefix.shape[1])
    results = []
    for text in texts:
        continuation = tokenizer(text, return_tensors="pt",
                                 add_special_tokens=False).input_ids
        count = int(continuation.shape[1])
        if count == 0:
            raise ScoringError("a scored continuation tokenised to nothing")
        total = prefix_length + count
        if total > MAX_SCORED_TOKENS:
            raise ScoringError(
                f"the scored sequence is {total} tokens, above the exactness bound "
                f"of {MAX_SCORED_TOKENS}; refusing to score a prefix")
        ids = torch.cat([prefix, continuation], dim=1).to(device)
        with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.bfloat16,
                enabled=str(device).startswith("cuda")):
            hidden = final_hidden_states(model, ids)
        per_token: list[float] = []
        for offset in range(0, count, chunk):
            stop = min(offset + chunk, count)
            logits = logits_chunk(model, hidden, prefix_length - 1 + offset,
                                  prefix_length - 1 + stop)
            log_probs = torch.log_softmax(logits, dim=-1)
            gathered = log_probs.gather(
                -1, continuation[:, offset:stop].to(device).unsqueeze(-1)).squeeze(-1)
            per_token.extend(float(value) for value in
                             gathered.reshape(-1).float().cpu().tolist())
        total_logprob = float(sum(per_token))
        results.append({
            "tokens": count,
            "sum": total_logprob,
            "mean": total_logprob / count,
            "per_token": [round(value, 6) for value in per_token],
        })
        del hidden
    return results


def select_item(task, prompt: str, score_fn, modes=RANKING_MODES) -> dict:
    """Score one item's candidates and return its selection record.

    `score_fn(prompt, texts)` returns one {"tokens", "sum", "mean", ...} record per
    text. Content scores use the candidate content; letter scores use `` A`` and so
    on. The reference label is recorded for the audit only and is never an input to
    the choice.
    """
    from . import forensics

    rows = selection_contents(task)
    labels = [row["label"] for row in rows]
    contents = [row["content"] for row in rows]
    content_scores = score_fn(prompt, [content_continuation(content)
                                       for content in contents])
    letter_scores = score_fn(prompt, letter_continuations(labels))
    reference = forensics.reference_label(task)
    record = {
        "id": task.id,
        "format": formats.task_format(task),
        "metric": formats.metric_for(task),
        "prompt_variant": PROMPT_VARIANT,
        "prompt_sha256": forensics.sha256_text(prompt),
        "labels": labels,
        "reference": reference,
        "reference_position": labels.index(reference) if reference in labels else None,
        "candidates": [
            {
                "label": row["label"],
                "content_sha256": forensics.sha256_text(row["content"]),
                "content_tokens": content_score["tokens"],
                "content_sum": content_score["sum"],
                "content_mean": content_score["mean"],
                "letter_tokens": letter_score["tokens"],
                "letter_sum": letter_score["sum"],
                "letter_mean": letter_score["mean"],
                "content_per_token": content_score.get("per_token"),
            }
            for row, content_score, letter_score
            in zip(rows, content_scores, letter_scores)
        ],
        "rankings": {},
    }
    flat = {
        CONTENT_MEAN: [{"sum": row["content_sum"], "mean": row["content_mean"]}
                       for row in record["candidates"]],
        CONTENT_SUM: [{"sum": row["content_sum"], "mean": row["content_mean"]}
                      for row in record["candidates"]],
        LETTER_MEAN: [{"sum": row["letter_sum"], "mean": row["letter_mean"]}
                      for row in record["candidates"]],
        LETTER_SUM: [{"sum": row["letter_sum"], "mean": row["letter_mean"]}
                     for row in record["candidates"]],
    }
    for mode in modes:
        field = "mean" if mode.endswith("mean") else "sum"
        record["rankings"][mode] = choose(flat[mode], labels, field)
        record["rankings"][mode]["scored"] = mode == PRIMARY_MODE
    return record


def selection_prompt(task) -> str:
    """The selection instrument's prompt: no example letter, content is scored."""
    from . import suite as suite_module

    base = formats.render_prompt(task, example_letter=None)
    if formats.task_format(task) == formats.FORMAT_PATCH:
        replacement = (
            "## Answer format\n\n"
            "The harness scores each candidate above as a continuation of this\n"
            "prompt and takes the highest-scoring one; nothing is generated. The\n"
            "answer is the candidate's text.")
    elif formats.task_format(task) == formats.FORMAT_DIAGNOSTIC:
        replacement = (
            "## Answer format\n\n"
            "The harness scores each option above as a continuation of this prompt\n"
            "and takes the highest-scoring one; nothing is generated. The answer is\n"
            "the option's text.")
    else:
        raise ScoringError(f"{task.id}: {formats.task_format(task)} is not a "
                           "selection format")
    head = base[:base.rindex("## Answer format")].rstrip("\n")
    return head + "\n\n" + replacement + "\n"


def position_audit(records: list[dict]) -> dict:
    """Where the reference sits, per format — the leakage guard's evidence.

    The *choice* is made on content scores, which never see the label or the
    position; this histogram is recorded so a reader can see that the reference is
    not always shown at the same place. A skewed histogram is reported, not
    corrected after the fact.
    """
    by_format: dict = {}
    for record in records:
        entry = by_format.setdefault(record["format"],
                                     {"items": 0, "positions": {}})
        entry["items"] += 1
        position = record["reference_position"]
        key = str(position)
        entry["positions"][key] = entry["positions"].get(key, 0) + 1
    for entry in by_format.values():
        counts = list(entry["positions"].values())
        entry["distinct_positions"] = len(entry["positions"])
        entry["constant_position"] = len(entry["positions"]) == 1
        entry["max_share"] = round(max(counts) / max(1, entry["items"]), 6)
    return {
        "by_format": by_format,
        "position_free_by_construction": (
            "the scored continuation is the candidate's content; the label, the "
            "position and the reference flag are never inputs to the ranking"),
    }


def verify_registration(registration: dict, suite_sha256: str,
                        formats_seen: list[str]) -> list[str]:
    """Problems between the v2 registration and what this stage actually does."""
    problems = []
    if registration.get("schema") != SCHEMA_REGISTRATION:
        problems.append(f"the registration schema is not {SCHEMA_REGISTRATION!r}")
    if registration.get("suite_sha256") != suite_sha256:
        problems.append("the registration does not bind this suite hash")
    instrument = registration.get("instrument") or {}
    if instrument.get("prompt_variant") != PROMPT_VARIANT:
        problems.append(f"the registered prompt variant is "
                        f"{instrument.get('prompt_variant')!r}, the stage applies "
                        f"{PROMPT_VARIANT!r}")
    if instrument.get("primary_ranking_mode") != PRIMARY_MODE:
        problems.append(f"the registered primary mode is "
                        f"{instrument.get('primary_ranking_mode')!r}, the stage "
                        f"applies {PRIMARY_MODE!r}")
    if instrument.get("example_letter_in_prompt") is not False:
        problems.append("the registration does not require the prompt to omit the "
                        "example letter")
    registered_formats = instrument.get("formats") or []
    missing = [fmt for fmt in formats_seen if fmt not in registered_formats]
    if missing:
        problems.append(f"the registration does not cover formats {missing}")
    thresholds = registration.get("entropy_screen") or {}
    from . import forensics as forensics_module

    if thresholds.get("min_answer_entropy_bits") != (
            forensics_module.MIN_ANSWER_ENTROPY_BITS):
        problems.append("the registered entropy minimum differs from the harness")
    if thresholds.get("max_modal_answer_share") != (
            forensics_module.MAX_MODAL_ANSWER_SHARE):
        problems.append("the registered modal-share maximum differs from the harness")
    primary = registration.get("primary") or {}
    if primary.get("metric") != formats.METRIC_EXECUTABLE:
        problems.append("the registration does not name the executable checker as "
                        "the primary metric for selection formats")
    if primary.get("margin_is_diagnostic") is not True:
        problems.append("the registration does not state that the ranking margin is "
                        "a diagnostic")
    return problems
