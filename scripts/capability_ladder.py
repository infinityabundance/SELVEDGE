#!/usr/bin/env python3
"""Run the prompt sanity ladder on one T2 arm (S30).

    SELVEDGE_MEM=<measured> SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=4g \
    scripts/court --gpu oracle -- python3 scripts/capability_ladder.py \
      --arm D --out results/capability/ladder/D.json

Eighteen short prompts (a trivial MCQ at three positions, two example-letter
controls, and a real patch item under the registered prompt, without the example,
with an explicit one-character instruction, under three candidate permutations,
and with two few-shot examples). Decoding is capped at 32 tokens: the ladder is
asking what letter the arm emits, not how well it writes.

It writes a receipt and prints one line per condition. It never scores an item
and never touches a registered metric or the T2 verdict.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "oracle"))

from capability import adapters, court, forensics, ladder, suite as suite_module  # noqa: E402

SCHEMA = "selvedge/capability-ladder/v1"


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


def conditions_for(tasks: dict) -> list[dict]:
    conditions: list[dict] = []
    for position, letter in enumerate("ABC"):
        prompt, correct = ladder.mcq_prompt(position)
        conditions.append({
            "name": f"mcq_answer_at_{letter}",
            "kind": "mcq",
            "prompt": prompt,
            "correct": correct,
            "example_letter": None,
            "note": f"a trivial MCQ whose correct option is at {letter}",
        })
    prompt, correct = ladder.mcq_prompt(2, example_letter="C")
    conditions.append({
        "name": "mcq_example_correct",
        "kind": "mcq",
        "prompt": prompt,
        "correct": correct,
        "example_letter": "C",
        "note": "the correct option is also the instruction's example letter",
    })
    prompt, correct = ladder.mcq_prompt(2, example_letter="A")
    conditions.append({
        "name": "mcq_example_wrong",
        "kind": "mcq",
        "prompt": prompt,
        "correct": correct,
        "example_letter": "A",
        "note": ("the correct option is C and the example is A: emitting A is "
                 "copying the instruction"),
    })
    for task in tasks:
        for condition in ladder.patch_conditions(task):
            condition["task"] = task
            condition["name"] = f"{task.id}:{condition['name']}"
            conditions.append(condition)
    return conditions


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default=os.path.join(REPO, "fixtures",
                                                        "capability-calibrated"))
    parser.add_argument("--arm", required=True)
    parser.add_argument("--t2-checkpoint", required=True)
    parser.add_argument("--t2-config", required=True)
    parser.add_argument("--t2-admission", required=True)
    parser.add_argument("--t2-training-receipt", default=None)
    parser.add_argument("--t2-mask-order", default="legacy")
    parser.add_argument("--provenance-amendment", default=None)
    parser.add_argument("--hf-root", default=os.environ.get("HF_HOME", "/hf-cache"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--items", default=None,
                        help="comma-separated patch item ids (default: two items, "
                             "one whose reference is the example letter and one "
                             "whose reference is not)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    collections = suite_module.load_suite(args.suite)
    patch_tasks = {task.id: task for task in collections["suite"]
                   if task.metadata.get("format") == "patch-selection"}
    if args.items:
        wanted = args.items.split(",")
    else:
        ## One item whose reference is `C` (the example letter) and one whose
        ## reference is not, so a constant `C` cannot look like an answer.
        wanted = []
        for task_id in sorted(patch_tasks):
            prompt = suite_module.render_prompt(patch_tasks[task_id])
            if "### Candidate C" not in prompt:
                continue
            reference = forensics.reference_label(patch_tasks[task_id])
            wanted.append((reference, task_id))
        chosen: list[str] = []
        for reference in ("C", "B"):
            for label, task_id in wanted:
                if label == reference and task_id not in chosen:
                    chosen.append(task_id)
                    break
        wanted = chosen[:2]
    missing = [task_id for task_id in wanted if task_id not in patch_tasks]
    if missing:
        raise SystemExit(f"unknown patch items: {missing}")

    conditions = conditions_for([patch_tasks[task_id] for task_id in wanted])
    requests = []
    for condition in conditions:
        prompt = condition["prompt"]
        if condition.get("few_shot"):
            examples = ["\n\n".join([example, f"Answer: {answer}"])
                        for example, answer in condition["few_shot"]]
            prompt = "\n\n".join(examples + [prompt])
        condition["rendered_prompt"] = prompt
        condition["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
        requests.append(adapters.Request(
            item_id=condition["name"], prompt=prompt,
            max_new_tokens=ladder.LADDER_MAX_NEW_TOKENS, stops=(), seed=20261115))

    adapter = adapters.T2ArmAdapter(
        args.arm, args.t2_checkpoint, args.t2_config, args.t2_admission,
        args.hf_root, device=args.device, batch=args.batch,
        training_receipt=args.t2_training_receipt,
        mask_order=args.t2_mask_order,
        provenance_amendment=args.provenance_amendment)
    metadata = adapter.metadata()
    generations = adapter.generate(requests)
    if len(generations) != len(conditions):
        raise SystemExit("the adapter returned a different number of generations")

    rows = []
    for condition, generation in zip(conditions, generations):
        record = ladder.judge_condition(condition, generation.as_dict())
        record.pop("task", None)
        rows.append(record)

    payload = {
        "schema": SCHEMA,
        "stage": "ladder",
        "stamp": stamp_now(),
        "created_utc": utc_now(),
        "command": " ".join(sys.argv),
        "revision": court.revision(REPO),
        "suite": {"root": os.path.abspath(args.suite),
                  "suite_sha256": suite_module.build_manifest(args.suite)["suite_sha256"]},
        "adapter": metadata,
        "decoding": {"max_new_tokens": ladder.LADDER_MAX_NEW_TOKENS,
                     "greedy": True, "seed": 20261115, "stops": []},
        "items": wanted,
        "conditions": [
            {key: value for key, value in condition.items() if key != "task"}
            for condition in conditions
        ],
        "rows": rows,
        "verdict": ladder.ladder_verdict(rows),
        "resources": court.resource_usage(),
        "statement": ("a diagnostic ladder: it reports which letter the arm emitted "
                      "under each prompt condition. It scores no item, changes no "
                      "registered metric, and is not a capability measurement."),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"capability_ladder: wrote {args.out}")
    for row in rows:
        print(f"  {row['name'][:70]:70s} letter={str(row.get('letter')):6s} "
              f"example={str(row.get('example_letter')):5s} "
              f"ref={str(row.get('reference_letter')):5s} "
              f"{'CORRECT' if row.get('correct') else ''}"
              f"{'REF' if row.get('chosen_is_reference') else ''} "
              f"finish={row.get('finish_reason')}")
    verdict = payload["verdict"]
    print(f"  verdict: {json.dumps(verdict, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
