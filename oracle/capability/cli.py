#!/usr/bin/env python3
"""Command line for the capability court.

    python3 scripts/capability_court.py manifest --check
    python3 scripts/capability_court.py preregister --out docs/capability-preregistration.json
    python3 scripts/capability_court.py generate --adapter-spec arm.json --out gen.json
    python3 scripts/capability_court.py check --generations gen.json --out result.json
    python3 scripts/capability_court.py compare --arm result.json --baseline base.json
    python3 scripts/capability_court.py complementarity --arm-a d.json --arm-b m.json
    python3 scripts/capability_court.py controls --rows rows.json --table
    python3 scripts/capability_court.py order-invariance --a a.json --b b.json

`generate` runs where the model lives, `check` where `cargo` lives. The two-stage
split is the structural guarantee that the arm never saw the checker.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import adapters, court, suite

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_SUITE = os.path.join(REPO, "fixtures", "capability")
DEFAULT_PREREG = os.path.join(REPO, "docs", "capability-preregistration.json")


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--check", action="store_true")
    manifest.add_argument("--write", action="store_true")

    register = subparsers.add_parser("preregister")
    register.add_argument("--out", default=DEFAULT_PREREG)
    register.add_argument("--force", action="store_true")

    fake = subparsers.add_parser(
        "fake-reference",
        help="write an adapter spec whose answers are the reference solutions; the "
             "result receipt is an instrument check, never an arm result")
    fake.add_argument("--out", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--adapter-spec", default=None)
    generate.add_argument("--t2-arm", default=None,
                          help="T2 arm letter (D, M, X, H, H*)")
    generate.add_argument("--t2-checkpoint", default=None)
    generate.add_argument("--t2-config", default=None)
    generate.add_argument("--t2-admission", default=None)
    generate.add_argument("--t2-accounting", default=None)
    generate.add_argument("--t2-training-receipt", default=None,
                          help="results/t2/<arm>/run.json; required for a verifiable "
                               "checkpoint (arm, config, admission, step, mask order)")
    generate.add_argument("--t2-mask-order", default="legacy",
                          help="the dispatch mask placement the arm was trained with "
                               "(legacy or pair)")
    generate.add_argument("--provenance-amendment", default=None,
                          help="a certified amendment (results/t2/PROVENANCE-AMENDMENT-*.json) "
                               "that supplies a missing dispatch mask order for a routed "
                               "arm whose training receipt predates the field; its digest "
                               "is recorded in the downstream receipt")
    generate.add_argument("--hf-root", default=os.environ.get("HF_HOME", "/hf-cache"))
    generate.add_argument("--device", default=None)
    generate.add_argument("--batch", type=int, default=4)
    generate.add_argument("--out", required=True)
    generate.add_argument("--max-new-tokens", type=int,
                          default=court.DEFAULT_MAX_NEW_TOKENS)
    generate.add_argument("--order", default="sorted")
    generate.add_argument("--only", default=None,
                          help="comma-separated task ids (a subset run)")
    generate.add_argument("--preregistration", default=None)
    generate.add_argument("--few-shot", default=None,
                          help="JSON array of few-shot example strings")

    check = subparsers.add_parser("check")
    check.add_argument("--generations", required=True)
    check.add_argument("--out", required=True)
    check.add_argument("--preregistration", default=None)
    check.add_argument("--workroot", default="/work/target/capability-work")
    check.add_argument("--keep", action="store_true")

    compare = subparsers.add_parser("compare")
    compare.add_argument("--arm", required=True)
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--out", required=True)
    compare.add_argument("--threshold", type=float, default=None,
                         help="only for unregistered instrument runs; a registered "
                              "comparison uses the pre-registered threshold and "
                              "refuses a different one")
    compare.add_argument("--preregistration", default=None)

    complement = subparsers.add_parser("complementarity")
    complement.add_argument("--arm-a", required=True)
    complement.add_argument("--arm-b", required=True)
    complement.add_argument("--out", required=True)

    capacity = subparsers.add_parser("capacity")
    capacity.add_argument("--rows", required=True)
    capacity.add_argument("--out", required=True)

    controls = subparsers.add_parser("controls")
    controls.add_argument("--rows", required=True)
    controls.add_argument("--table", action="store_true")
    controls.add_argument("--out", default=None)

    order = subparsers.add_parser("order-invariance")
    order.add_argument("--a", required=True)
    order.add_argument("--b", required=True)
    order.add_argument("--out", required=True)

    level3 = subparsers.add_parser("level3")
    level3.add_argument("--result", required=True)
    level3.add_argument("--p95-ms", type=float, default=None,
                        help="p95 latency from a Court F receipt")
    level3.add_argument("--p95-source", default=None,
                        help="the Court F receipt the p95 value comes from")
    level3.add_argument("--out", required=True)

    args = parser.parse_args(argv)

    if args.command == "manifest":
        if args.check:
            ok, message = suite.check_manifest(args.suite)
            print(f"capability_court: {message}")
            if not ok:
                return 3
        if args.write:
            payload = suite.build_manifest(args.suite)
            suite.write_manifest(args.suite, payload)
            print(f"capability_court: wrote {suite.manifest_path(args.suite)} "
                  f"({payload['counts']['total']} tasks)")
        return 0

    if args.command == "preregister":
        payload = court.register(args.out, args.suite,
                                 refuse_overwrite=not args.force)
        print(f"capability_court: registered {args.out} for suite "
              f"{payload['suite_sha256'][:12]}")
        return 0

    if args.command == "fake-reference":
        collections = suite.load_suite(args.suite)
        outputs = {}
        for kind in ("suite", "trio"):
            for task in collections[kind]:
                blocks = []
                for path in task.reference_files:
                    blocks.append("```rust path=" + path + "\n"
                                  + task.read_reference(path).rstrip() + "\n```")
                outputs[task.id] = "\n".join(blocks)
        spec = {"kind": "fake", "name": "fake-reference", "outputs": outputs,
                "accounting": {"task_success": None},
                "note": "instrument smoke only: answers are the reference solutions"}
        court.write_json_atomic(args.out, spec)
        print(f"capability_court: wrote {len(outputs)} scripted answers -> {args.out}")
        return 0

    if args.command == "generate":
        if args.t2_arm:
            missing = [name for name, value in (
                ("--t2-checkpoint", args.t2_checkpoint),
                ("--t2-config", args.t2_config),
                ("--t2-admission", args.t2_admission)) if not value]
            if missing:
                raise SystemExit(f"--t2-arm needs {', '.join(missing)}")
            adapter = adapters.T2ArmAdapter(
                args.t2_arm, args.t2_checkpoint, args.t2_config, args.t2_admission,
                args.hf_root, device=args.device, batch=args.batch,
                accounting=args.t2_accounting,
                training_receipt=args.t2_training_receipt,
                mask_order=args.t2_mask_order,
                provenance_amendment=args.provenance_amendment)
        elif args.adapter_spec:
            spec = load_json(args.adapter_spec)
            adapter = adapters.adapter_from_spec(spec)
        else:
            raise SystemExit("give --adapter-spec or --t2-arm")
        few_shot = None
        if args.few_shot:
            few_shot = load_json(args.few_shot)
            if not isinstance(few_shot, list):
                raise SystemExit("--few-shot must be a JSON array of strings")
        only = args.only.split(",") if args.only else None
        receipt = court.generate(
            adapter, args.suite, args.out,
            max_new_tokens=args.max_new_tokens, order=args.order, only=only,
            preregistration=args.preregistration, few_shot_texts=few_shot,
            repo=REPO)
        print(f"capability_court: generated {len(receipt['items'])} items -> {args.out}")
        return 0

    if args.command == "check":
        generations = load_json(args.generations)
        result = court.check(generations, args.suite, args.out,
                             preregistration=args.preregistration,
                             keep=args.keep, workroot=args.workroot)
        scores = result["scores"]
        print(f"capability_court: composite {scores['primary_composite']:.4f} "
              f"[{scores['primary_ci']['ci_low']:.4f}, {scores['primary_ci']['ci_high']:.4f}]")
        for category, entry in sorted(scores["by_category"].items()):
            print(f"  {category:18s} {entry['mean']:.4f} (n={entry['n']})")
        print(f"  counts {scores['counts']}")
        return 0

    if args.command == "compare":
        prereg = load_json(args.preregistration) if args.preregistration else None
        comparison = court.compare(load_json(args.arm), load_json(args.baseline),
                                   threshold=args.threshold, preregistration=prereg)
        court.write_json_atomic(args.out, comparison)
        primary = comparison["primary"]
        print(f"capability_court: {comparison['verdict']}")
        print(f"  composite {primary['difference']['arm_composite']:.4f} vs "
              f"{primary['difference']['baseline_composite']:.4f}; "
              f"relative {primary['difference']['relative_difference']}; "
              f"CI [{primary['difference']['ci_low']:.4f}, "
              f"{primary['difference']['ci_high']:.4f}]")
        return 0

    if args.command == "complementarity":
        payload = court.complementarity(load_json(args.arm_a), load_json(args.arm_b))
        court.write_json_atomic(args.out, payload)
        print(f"capability_court: A {payload['composite_a']:.4f}, "
              f"B {payload['composite_b']:.4f}, oracle union "
              f"{payload['oracle_union']:.4f} (not deployable)")
        return 0

    if args.command == "capacity":
        rows = load_json(args.rows)
        if not isinstance(rows, list):
            raise SystemExit("--rows must be a JSON array")
        payload = court.capacity_scaling(rows)
        court.write_json_atomic(args.out, payload)
        print(f"capability_court: slope {payload['slope']:.3f} "
              f"[{payload['ci_low']:.3f}, {payload['ci_high']:.3f}], "
              f"sublinear={payload['sublinear']}")
        return 0

    if args.command == "controls":
        rows = load_json(args.rows)
        if not isinstance(rows, list):
            raise SystemExit("--rows must be a JSON array")
        for row in rows:
            court.validate_control_row(row)
        if args.table:
            print(court.render_control_table(rows))
        if args.out:
            court.write_json_atomic(args.out, {"schema": "selvedge/capability-controls/v1",
                                               "rows": rows})
        return 0

    if args.command == "level3":
        result = load_json(args.result)
        responsiveness = None
        if args.p95_ms is not None or args.p95_source:
            responsiveness = {"p95_ms": args.p95_ms, "source": args.p95_source}
        payload = court.level3(result, responsiveness)
        court.write_json_atomic(args.out, payload)
        print(f"capability_court: level 3 complete={payload['complete']}")
        return 0

    if args.command == "order-invariance":
        payload = court.prompt_order_invariance(load_json(args.a), load_json(args.b))
        court.write_json_atomic(args.out, payload)
        print(f"capability_court: invariant={payload['invariant']} "
              f"differences={len(payload['differences'])}")
        return 0 if payload["invariant"] else 4

    parser.error("unreachable")
    return 64


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
