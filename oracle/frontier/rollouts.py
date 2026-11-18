#!/usr/bin/env python3
"""Termination-aware stage-2b rollouts.

Registered by `docs/ws19-post-trained-route-stage2a-amendment-s56-v1.json`:

* the rollout cap is 2,048 new tokens;
* a rollout that does not emit EOS (cap or early cut) counts as **wrong** in the
  verified reward — it is counted, never dropped;
* a rollout is cut early when its decoded tail is loop-classified, which saves
  the compute the loop would have consumed;
* every cut records the emitted and saved tokens, and the false-cut rate is
  measured on a sample rather than assumed.

Decoding is untouched. The cut only ends a generation; it changes no
temperature, top_p, seed or penalty. A repetition penalty would change the
distribution and is therefore not added.

The false-cut measurement replays stored completions: it scans prefixes of the
stored text at a fixed character stride, applies the same detector, and verifies
the prefix that would have been emitted. For a deterministic rollout the tokens
before the cut are identical to the uncut run, so the simulation is exact up to
the detector's text/token boundary; the receipt says so.
"""

from __future__ import annotations

import torch

from frontier import rft, termination


class LoopCutCriteria:
    """A transformers `StoppingCriteria` that stops a degenerate tail.

    It is deliberately *not* a subclass at import time so this module can be
    reasoned about without transformers; `as_criteria()` returns the object HF
    expects. The rule is the calibrated detector, checked every
    `check_every_tokens` new tokens after `min_tokens`.
    """

    def __init__(self, tokenizer, prompt_length: int,
                 thresholds: dict | None = None, check_every_tokens: int = 64,
                 min_tokens: int = 128):
        if check_every_tokens <= 0:
            raise ValueError("check_every_tokens must be positive")
        self.tokenizer = tokenizer
        self.prompt_length = int(prompt_length)
        self.thresholds = {**termination.DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.check_every_tokens = int(check_every_tokens)
        self.min_tokens = int(min_tokens)
        self.cut: list[bool] = []
        self.cut_tokens: list[int | None] = []
        self.cut_reason: list[str | None] = []

    def _ensure(self, rows: int) -> None:
        if len(self.cut) < rows:
            self.cut.extend([False] * (rows - len(self.cut)))
            self.cut_tokens.extend([None] * (rows - len(self.cut_tokens)))
            self.cut_reason.extend([None] * (rows - len(self.cut_reason)))

    def __call__(self, input_ids, scores=None, **kwargs):
        rows = input_ids.shape[0]
        self._ensure(rows)
        generated = int(input_ids.shape[1]) - self.prompt_length
        stop = torch.zeros(rows, dtype=torch.bool, device=input_ids.device)
        if generated < self.min_tokens or generated % self.check_every_tokens:
            return stop
        for row in range(rows):
            if self.cut[row]:
                stop[row] = True
                continue
            text = self.tokenizer.decode(input_ids[row, self.prompt_length:],
                                         skip_special_tokens=True)
            signal = termination.classify(text, self.thresholds)
            if signal["loop"]:
                self.cut[row] = True
                self.cut_tokens[row] = generated
                self.cut_reason[row] = signal["reason"]
                stop[row] = True
        return stop


def generate_termination_aware(model, tokenizer, prompts: list[str], *,
                               max_new_tokens: int = 2048,
                               num_return_sequences: int = 1,
                               do_sample: bool = True, temperature: float = 1.0,
                               top_p: float = 1.0, seed: int | None = None,
                               device: str = "cuda",
                               thresholds: dict | None = None,
                               check_every_tokens: int = 64,
                               min_tokens: int = 128) -> tuple[list[str], dict]:
    """Generate rollouts, cutting a loop tail early, and account for the cut."""
    from transformers import StoppingCriteriaList

    encoded = tokenizer(prompts, return_tensors="pt", padding=True,
                        add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    prompt_length = int(input_ids.shape[1])
    if seed is not None:
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
    criteria = LoopCutCriteria(tokenizer, prompt_length, thresholds,
                               check_every_tokens, min_tokens)
    kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=int(max_new_tokens),
        num_return_sequences=int(num_return_sequences),
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        stopping_criteria=StoppingCriteriaList([criteria]),
    )
    if do_sample:
        kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        kwargs.update(do_sample=False)
    with torch.inference_mode():
        output = model.generate(**kwargs)
    generated = output[:, prompt_length:]
    eos = tokenizer.eos_token_id
    completions, rows = [], []
    for index, row in enumerate(generated):
        ids = row.tolist()
        while ids and ids[-1] == tokenizer.pad_token_id:
            ids.pop()
        terminated = bool(ids) and ids[-1] == eos
        cut = bool(criteria.cut[index]) if index < len(criteria.cut) else False
        completions.append(tokenizer.decode(row, skip_special_tokens=True))
        rows.append({
            "new_tokens": len(ids),
            "terminated": terminated,
            # No EOS means the rollout is not a finished answer, whether the
            # cap or the loop cut ended it.
            "truncated": not terminated,
            "cut": cut,
            "cut_tokens": criteria.cut_tokens[index] if cut else None,
            "cut_reason": criteria.cut_reason[index] if cut else None,
            "saved_tokens": (int(max_new_tokens) - len(ids)) if cut else 0,
        })
    stats = {
        "max_new_tokens": int(max_new_tokens),
        "rollouts": rows,
        "new_tokens": sum(row["new_tokens"] for row in rows),
        "truncated": sum(1 for row in rows if row["truncated"]),
        "cut": sum(1 for row in rows if row["cut"]),
        "saved_tokens": sum(row["saved_tokens"] for row in rows),
        "thresholds": criteria.thresholds,
    }
    return completions, stats


def verified_reward(completion: str, target: str, dataset: str,
                    truncated: bool) -> dict:
    """Reward 1 only for a terminated, verified-correct rollout.

    A truncated rollout is wrong by definition, even when the prefix happens to
    contain a parseable correct answer: the model did not finish the answer.
    """
    if truncated:
        return {"reward": 0.0, "verdict": "truncated", "verified": False,
                "reason": "no EOS before the cap or the loop cut"}
    result = rft.verify(completion, target, dataset)
    return {"reward": 1.0 if result["verdict"] == "correct" else 0.0,
            "verdict": result["verdict"],
            "verified": result["verdict"] == "correct",
            "extracted": result.get("extracted")}


def simulate_cut(text: str, thresholds: dict | None = None,
                 stride_chars: int = 64, min_chars: int = 128
                 ) -> dict | None:
    """The earliest prefix at which the detector would cut, or None."""
    if stride_chars <= 0:
        raise ValueError("stride_chars must be positive")
    for end in range(min_chars, len(text) + 1, stride_chars):
        signal = termination.classify(text[:end], thresholds)
        if signal["loop"]:
            return {"cut_chars": end, "reason": signal["reason"]}
    return None


def false_cut_analysis(rows: list[dict], thresholds: dict | None = None,
                       stride_chars: int = 64) -> dict:
    """Measure cuts, saved tokens and false cuts on stored generations.

    Each row needs `completion`, `new_tokens`, `flexible_correct`, `gold` and
    `dataset` (or `task`). A cut is **false** when the uncut answer verifies
    correct and the truncated prefix does not. That is the definition the
    registration uses; it does not count a cut as false merely because the
    prefix is unparseable when the full answer was also unparseable.
    """
    total = 0
    cut_rows = 0
    saved_tokens = 0.0
    false_cuts = 0
    correct_uncut = 0
    per_group: dict[tuple, dict] = {}
    examples = []
    for row in rows:
        total += 1
        text = row["completion"]
        dataset = row.get("dataset") or row.get("task") or "gsm8k"
        target = row.get("gold") or row.get("target")
        key = (row.get("task") or row.get("dataset"), row.get("mode"),
               row.get("decoding"))
        group = per_group.setdefault(key, {"samples": 0, "cuts": 0,
                                           "false_cuts": 0, "saved_tokens": 0.0,
                                           "correct_uncut": 0})
        group["samples"] += 1
        uncut_correct = bool(row.get("flexible_correct"))
        if uncut_correct:
            correct_uncut += 1
            group["correct_uncut"] += 1
        cut = simulate_cut(text, thresholds, stride_chars)
        if cut is None:
            continue
        cut_rows += 1
        group["cuts"] += 1
        chars = max(1, len(text))
        estimated = int(round(row["new_tokens"] * (chars - cut["cut_chars"])
                              / chars))
        saved_tokens += estimated
        group["saved_tokens"] += estimated
        prefix = text[:cut["cut_chars"]]
        if target is None:
            prefix_correct = False
        else:
            prefix_correct = (rft.verify(prefix, target, dataset)["verdict"]
                              == "correct")
        if uncut_correct and not prefix_correct:
            false_cuts += 1
            group["false_cuts"] += 1
            if len(examples) < 5:
                examples.append({
                    "task": row.get("task"), "mode": row.get("mode"),
                    "decoding": row.get("decoding"),
                    "cut_chars": cut["cut_chars"], "reason": cut["reason"],
                    "uncut_correct": True,
                })
    return {
        "samples": total,
        "cut_samples": cut_rows,
        "cut_fraction": (cut_rows / total) if total else None,
        "saved_tokens_estimate": saved_tokens,
        "correct_uncut": correct_uncut,
        "false_cuts": false_cuts,
        "false_cut_rate": (false_cuts / correct_uncut) if correct_uncut else None,
        "false_cut_rate_of_cuts": (false_cuts / cut_rows) if cut_rows else None,
        "stride_chars": stride_chars,
        "thresholds": {**termination.DEFAULT_THRESHOLDS, **(thresholds or {})},
        "by_task_mode_decoding": [
            {"task": key[0], "mode": key[1], "decoding": key[2],
             "samples": value["samples"], "cuts": value["cuts"],
             "cut_fraction": value["cuts"] / value["samples"],
             "correct_uncut": value["correct_uncut"],
             "false_cuts": value["false_cuts"],
             "false_cut_rate": (value["false_cuts"] / value["correct_uncut"]
                                if value["correct_uncut"] else None),
             "saved_tokens_estimate": value["saved_tokens"]}
            for key, value in sorted(per_group.items())
        ],
        "false_cut_examples": examples,
        "method": ("scan stored completions at a fixed character stride with "
                   "the calibrated detector, verify the prefix that would have "
                   "been emitted, and compare with the uncut verdict. For a "
                   "deterministic rollout the prefix tokens are identical to "
                   "the uncut run; the stride is the text/token approximation."),
    }


__all__ = ["LoopCutCriteria", "generate_termination_aware", "verified_reward",
           "simulate_cut", "false_cut_analysis"]
