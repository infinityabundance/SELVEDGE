#!/usr/bin/env python3
"""S52: P19.9 counterfactual pre-read router screen — executable runner.

Stages (serialised; one GPU job at a time):

    frame     validate the registered prompt set and write its fingerprint
    rollouts  GPU: paired base / base+memory rollouts + causal hidden states
    train     CPU: pre-read router for the outcome and CE supervision sources
    eval      GPU: base, dense memory, routed arms, gate and dense controls
    screen    CPU: apply the registered win bar; emit the screen receipt
    check     CPU: verify registration, prompt set and runner agree

The `rollouts` and `eval` stages refuse to start until the corrected 20M bar has
a receipt and a real-gradient first-moment validation receipt exists, and they
refuse a first-moment storage that contradicts that receipt's decision. This is
the S50 queue ordering, enforced here rather than remembered.

No number in a receipt is typed by hand: accuracies come from the S45 exact
extractor over generated completions, and read cost comes from the measured
gather ledger plus `physical_read_accounting`.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "oracle"))

from frontier import routing_runner as runner  # noqa: E402
from frontier import rft  # noqa: E402

PROMPT_SET = "docs/s52-p19.9-calibration-prompts-v2.json"
REGISTRATION = "docs/s52-p19.9-router-screen-registration-v2.json"
BAR_RECEIPT = "results/stage2/runs/s50-bar-20m-lr1e-5/receipt.json"
STUDENT_DIR = ("/hf-cache/models/Qwen/Qwen3.5-0.8B-Base/"
               "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
MEMORY_INIT = "results/stage2/graft/20261116T231451Z/memory-init.pt"
MEMORY_CHECKPOINT = "results/stage2/runs/memory-2m/ckpt-000976.pt"
GATED_DENSE_CHECKPOINT = "results/stage2/runs/gated-dense-2m-math/ckpt-000976.pt"
GIB = float(1 << 30)


# ---------------------------------------------------------------------------
# receipts and environment


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("SELVEDGE_REVISION", "unknown")


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def cgroup_memory() -> dict:
    report = {}
    for name in ("memory.peak", "memory.max", "memory.swap.peak"):
        try:
            with open(os.path.join("/sys/fs/cgroup", name), encoding="utf-8") as handle:
                report[name.replace(".", "_")] = handle.read().strip()
        except OSError:
            report[name.replace(".", "_")] = None
    events = {}
    try:
        with open("/sys/fs/cgroup/memory.events", encoding="utf-8") as handle:
            for line in handle:
                key, _, value = line.partition(" ")
                events[key] = int(value)
    except (OSError, ValueError):
        events = {}
    report["memory_events"] = events
    return report


def spec_from_args(args) -> runner.RolloutSpec:
    return runner.RolloutSpec(
        k=args.k, temperature=args.temperature, top_p=args.top_p,
        mode=args.mode, max_new_tokens=args.max_new_tokens, seed=args.seed)


def load_registration(path: str) -> dict:
    if not os.path.isfile(path):
        raise SystemExit(f"p19.9: missing registration {path}")
    return json.load(open(path, encoding="utf-8"))


def require_queue_order(args) -> dict:
    """The S50 queue gate: bar receipt, then the first-moment decision."""
    if not os.path.isfile(args.bar_receipt):
        raise SystemExit(
            "p19.9: the corrected 20M bar has no receipt yet; the queue forbids "
            f"this GPU stage before {args.bar_receipt} exists")
    paths = sorted(glob.glob(os.path.join(
        ROOT, "results/stage2/s50-real-gradient-validation-*.json")))
    if not paths:
        raise SystemExit(
            "p19.9: no real-gradient first-moment validation receipt; queue "
            "item (d) must settle the storage before the router screen")
    latest = paths[-1]
    payload = json.load(open(latest, encoding="utf-8"))
    decision = payload.get("decision") or {}
    status = decision.get("status")
    if not status:
        raise SystemExit(f"p19.9: {latest} has no decision.status")
    switched = str(status).startswith("switch")
    if switched and args.first_moment != "bfloat16":
        raise SystemExit(
            f"p19.9: first-moment validation decided {status!r}; the screen must "
            "be launched with --first-moment bfloat16")
    return {
        "bar_receipt": os.path.relpath(args.bar_receipt, ROOT),
        "bar_receipt_sha256": runner.sha256_file(args.bar_receipt),
        "first_moment_receipt": os.path.relpath(latest, ROOT),
        "first_moment_receipt_sha256": runner.sha256_file(latest),
        "first_moment_decision": decision,
        "first_moment_storage": args.first_moment,
    }


def require_cuda(limit_gib: float) -> dict:
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("p19.9: CUDA required for this stage")
    free, total = torch.cuda.mem_get_info()
    if limit_gib * GIB > free:
        raise SystemExit(
            f"p19.9: vram limit {limit_gib:.1f} GiB exceeds free "
            f"{free / GIB:.2f} GiB")
    return {"free_gib": free / GIB, "total_gib": total / GIB,
            "limit_gib": float(limit_gib)}


def check_host_memory(required_gib: float) -> None:
    available = None
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) / (1 << 20)
                break
    if available is not None and available < required_gib:
        raise SystemExit(
            f"p19.9: host MemAvailable {available:.1f} GiB < required "
            f"{required_gib:.1f} GiB (limit + 4 GiB)")


# ---------------------------------------------------------------------------
# stages


def stage_frame(args) -> int:
    spec = spec_from_args(args)
    items, meta = runner.load_prompt_set(args.prompt_set, spec)
    registration = load_registration(args.registration)
    arms = registration.get("arms") or {}
    missing = [arm for arm in runner.ROUTER_ARMS if arm not in arms]
    if missing:
        raise SystemExit(f"p19.9: registration is missing arms {missing}")
    payload = {
        "schema": "selvedge/p19.9-frame/v1",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "prompt_set": meta,
        "sampling": spec.mapping(),
        "splits": {
            split: [item.prompt_id for item in items if item.split == split]
            for split in runner.SPLITS},
        "registration": args.registration,
        "registration_sha256": runner.sha256_file(args.registration),
        "arms": runner.ROUTER_ARMS,
        "win_bar": registration.get("win_bar"),
        "out": args.out,
    }
    write_json(args.out, payload)
    print(f"p19.9-frame: {args.out} "
          f"({payload['prompt_set']['counts']}, sha256 {meta['sha256'][:16]})")
    return 0


def _load_model(student_dir: str, device: str):
    from frontier import arms
    return arms.load_model(student_dir, dtype=torch_bfloat16(), device=device,
                           attn_implementation="sdpa")


def torch_bfloat16():
    import torch
    return torch.bfloat16


def _build_routed_arm(memory_init: str, memory_checkpoint: str | None,
                      device: str, router_dir: str | None = None,
                      route: bool = True, threshold: float = 0.5,
                      zero_read: bool = False, dtype=None):
    import torch
    from frontier import memory as memory_mod
    init = torch.load(memory_init, map_location="cpu", weights_only=False)
    config = init["config"]
    core = memory_mod.ConditionalMemory(
        hidden=config["hidden"], value_dim=config["value_dim"],
        rows=config["rows"], sites=tuple(config["sites"]),
        conv_kernel=config["conv_kernel"], table_dtype=torch.float32,
        seed=config["seed"])
    state = init["state_dict"]
    if memory_checkpoint:
        payload = torch.load(memory_checkpoint, map_location="cpu",
                             weights_only=False)
        state = payload.get("arm_state", payload)
    core.load_state_dict(state, strict=True)
    core = core.to(device)
    router = None
    if router_dir:
        router, _ = runner.load_router(router_dir)
        if router.hidden != config["hidden"]:
            raise SystemExit(
                f"p19.9: router hidden {router.hidden} != memory hidden "
                f"{config['hidden']}")
        router = router.to(device)
    arm = runner.RoutedMemoryArm(core, router=router, threshold=threshold,
                                 route=route, zero_read=zero_read)
    return arm


def _sampling_lookup(model_dir: str, device: str, items, spec, condition,
                     memory_init=None, memory_checkpoint=None,
                     router_path=None, route=True, zero_read=False,
                     threshold=0.5):
    """Generate every (prompt, sample) for one condition, one model state."""
    import torch
    from transformers import AutoTokenizer
    from frontier import arms

    tokenizer = AutoTokenizer.from_pretrained(model_dir,
                                              trust_remote_code=False)
    model = _load_model(model_dir, device)
    handles = []
    ledger = None
    arm = None
    if condition == "memory" or condition == "routed" or condition == "zero":
        arm = _build_routed_arm(
            memory_init, memory_checkpoint, device,
            router_dir=router_path, route=(condition == "memory" or route),
            zero_read=zero_read or condition == "zero", threshold=threshold)
        ledger = arm.ledger
        handles = arms.install_memory(model, arm)
    results = {}
    prompt_hidden = {}
    for item in items:
        plan = runner.sampling_plan(spec, item.prompt_id)
        for sample in plan:
            if arm is not None:
                arm.ledger.mark_generation()
            results[(item.prompt_id, sample["sample_index"])] = _generate_one(
                model, tokenizer, item.prompt, sample, spec.max_new_tokens,
                device)
            if arm is not None:
                arm.reset_state()
        if arm is None:
            prompt_hidden[item.prompt_id] = _capture_site_hidden(
                model, tokenizer, item.prompt, args_site(), device)
    arms.uninstall(handles)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return results, prompt_hidden, (ledger.report() if ledger else None)


def args_site() -> int:
    return 2


def _generate_one(model, tokenizer, prompt: str, sample: dict, max_new_tokens: int,
                  device: str) -> dict:
    import torch
    ids = tokenizer(prompt, return_tensors="pt",
                    add_special_tokens=False)["input_ids"].to(device)
    torch.manual_seed(sample["seed"])
    if device == "cuda":
        torch.cuda.manual_seed_all(sample["seed"])
    kwargs = dict(
        input_ids=ids, max_new_tokens=max_new_tokens, num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if sample["greedy"]:
        kwargs["do_sample"] = False
    else:
        kwargs.update(do_sample=True, temperature=sample["temperature"],
                      top_p=sample["top_p"])
    with torch.inference_mode():
        output = model.generate(**kwargs)
        prompt_length = int(ids.shape[1])
        completion = tokenizer.decode(output[0, prompt_length:],
                                      skip_special_tokens=True)
        logits = model(output).logits.float()
        log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
        targets = output[:, 1:]
        losses = -log_probs.gather(-1, targets[..., None]).squeeze(-1)
        token_losses = losses[0, prompt_length - 1:].tolist()
    ce = sum(token_losses) / len(token_losses) if token_losses else None
    return {"completion": completion, "ce": ce, "token_ce": token_losses}


def _capture_site_hidden(model, tokenizer, prompt: str, site: int,
                         device: str):
    """Causal hidden state at the site's output, before any residual."""
    import torch
    ids = tokenizer(prompt, return_tensors="pt",
                    add_special_tokens=False)["input_ids"].to(device)
    captured = {}

    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden.detach().to("cpu", torch.float16)

    from frontier import arms
    layer = arms._layer(model, site)
    handle = layer.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(ids)
    finally:
        handle.remove()
    return captured["hidden"][0]


def stage_rollouts(args) -> int:
    queue = require_queue_order(args)
    device_report = require_cuda(args.vram_limit_gib)
    spec = spec_from_args(args)
    items, meta = runner.load_prompt_set(args.prompt_set, spec)
    os.makedirs(args.out_dir, exist_ok=True)
    started = time.time()
    base_results, hidden, _ = _sampling_lookup(
        args.student_dir, "cuda", items, spec, "base")
    print(f"p19.9-rollouts: base {len(base_results)} completions", flush=True)
    memory_results, _, ledger = _sampling_lookup(
        args.student_dir, "cuda", items, spec, "memory",
        memory_init=args.memory_init,
        memory_checkpoint=args.memory_checkpoint)
    print(f"p19.9-rollouts: memory {len(memory_results)} completions", flush=True)

    def generate(condition, item, sample):
        source = base_results if condition == "base" else memory_results
        return source[(item.prompt_id, sample["sample_index"])]

    results = runner.run_paired_rollouts(items, generate,
                                        _exact_verifier(spec), spec)
    hidden_path = os.path.join(args.out_dir, "hidden.pt")
    import torch
    torch.save({key: value for key, value in hidden.items()}, hidden_path)
    receipt = runner.rollout_receipt(results, spec, meta, extra={
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "run_dir": args.out_dir,
        "student_dir": args.student_dir,
        "memory_init": args.memory_init,
        "memory_init_sha256": runner.sha256_file(args.memory_init),
        "memory_checkpoint": args.memory_checkpoint,
        "memory_checkpoint_sha256": (runner.sha256_file(args.memory_checkpoint)
                                     if args.memory_checkpoint else None),
        "memory_read_mechanism": (
            "conv-free 8-row table gather with the memory key gate and value "
            "projection; no cross-token convolution, so a sparse gather is exact"),
        "hidden_artifact": os.path.relpath(hidden_path, ROOT),
        "hidden_artifact_sha256": runner.sha256_file(hidden_path),
        "dense_read_ledger": ledger,
        "queue": queue,
        "cuda": device_report,
        "container_memory": cgroup_memory(),
        "wall_seconds": round(time.time() - started, 1),
        "command": sys.argv,
    })
    out = os.path.join(args.out_dir, "rollouts.json")
    write_json(out, receipt)
    print(f"p19.9-rollouts: {out}", flush=True)
    return 0


def _exact_verifier(spec):
    from frontier import correctness
    def verify(item, completion):
        verdict = rft.verify(completion, item.answer,
                             correctness.extractor_dataset(item.dataset))
        return {"verdict": verdict["verdict"]}
    return verify


def stage_train(args) -> int:
    import torch
    payload = runner.require_rollouts(args.rollouts)
    hidden_path = os.path.join(ROOT, payload.get("hidden_artifact") or "")
    if not os.path.isfile(hidden_path):
        raise SystemExit(
            f"p19.9: missing hidden-state artifact {hidden_path}; rerun the "
            "rollout stage")
    hidden = torch.load(hidden_path, map_location="cpu", weights_only=False)
    results = payload["prompts"]
    sources = (["counterfactual-outcome", "counterfactual-token-loss"]
               if args.source == "both" else [args.source])
    os.makedirs(args.out_dir, exist_ok=True)
    reports = {}
    for source in sources:
        labels = runner.labels_by_split(results, source)
        hidden_dim = int(next(iter(hidden.values())).shape[-1])
        router = runner.build_router(hidden_dim, tau=args.tau, seed=args.seed)
        metadata = {
            prompt_id: runner.metadata_for_prompt(prompt_id,
                                                  int(value.shape[0]))
            for prompt_id, value in hidden.items()}
        examples = runner.assemble_examples(router, hidden, metadata, labels,
                                            source)
        fit = [row for row in examples if row.split == "fit"]
        calib = [row for row in examples if row.split == "calib"]
        training = runner.train_router(
            router, fit, calib, lr=args.lr, epochs=args.epochs,
            batch=args.batch, seed=args.seed, threshold=args.threshold)
        name = "outcome" if source == "counterfactual-outcome" else "ce"
        router_path = os.path.join(args.out_dir, f"router-{name}.pt")
        runner.save_router(router, router_path, training)
        holdout = [row for row in examples if row.split == "holdout"]
        holdout_loss = None
        if holdout:
            with torch.no_grad():
                x = torch.cat([row.features for row in holdout])
                y = torch.cat([row.targets for row in holdout])
                w = torch.cat([row.weights for row in holdout])
                prediction = router.predictor(x).squeeze(-1)
                holdout_loss = float((((prediction - y) ** 2) * w).sum() / w.sum())
        reports[name] = {
            "source": source,
            "router": os.path.relpath(router_path, ROOT),
            "router_sha256": runner.sha256_file(router_path),
            "hidden_dim": hidden_dim,
            "holdout_loss": holdout_loss,
            **training,
        }
        print(f"p19.9-train: {source} chosen epoch "
              f"{training['chosen_epoch']} calib {training['calib_loss_at_chosen_epoch']:.6f} "
              f"holdout {holdout_loss}", flush=True)
    out = os.path.join(args.out_dir, "train.json")
    write_json(out, {
        "schema": runner.ROUTER_RECEIPT_SCHEMA,
        "created_utc": utcnow(), "stamp": stamp(), "revision": revision(),
        "rollouts": os.path.relpath(args.rollouts, ROOT),
        "rollouts_sha256": runner.sha256_file(args.rollouts),
        "sources": {key: value["source"] for key, value in reports.items()},
        "routers": reports,
        "split_unit": "prompt_id; fit trains, calib chooses the epoch, holdout "
                      "is scored once",
        "command": sys.argv,
    })
    print(f"p19.9-train: {out}", flush=True)
    return 0


def stage_eval(args) -> int:
    queue = require_queue_order(args)
    device_report = require_cuda(args.vram_limit_gib)
    payload = runner.require_rollouts(args.rollouts)
    spec = runner.RolloutSpec(**payload["sampling"])
    items = [runner.PromptItem(
        prompt_id=row["prompt_id"], dataset=row["dataset"], prompt=row["prompt"],
        answer=row["answer"], split=row["split"])
        for row in payload["prompts"] if row["split"] == "holdout"]
    if not items:
        raise SystemExit("p19.9: no holdout prompts in the rollout receipt")
    arms_to_score = ("base", "memory", "utility-outcome", "utility-ce",
                     "no-memory-gate", "gated-dense")
    if args.arms:
        arms_to_score = tuple(name.strip() for name in args.arms.split(",")
                              if name.strip())
    summary = {}
    ledgers = {}
    selection = {}
    started = time.time()

    def score(name, results_lookup):
        records = []
        per_prompt = []
        for item in items:
            prompt_records = []
            for sample in runner.sampling_plan(spec, item.prompt_id):
                generated = results_lookup[(item.prompt_id,
                                           sample["sample_index"])]
                verdict = _exact_verifier(spec)(item, generated["completion"])
                prompt_records.append(verdict["verdict"])
                records.append(verdict["verdict"])
            correct = sum(1 for value in prompt_records if value == "correct")
            per_prompt.append({
                "prompt_id": item.prompt_id,
                "score": {name: correct / len(prompt_records)
                          if prompt_records else 0.0},
                "correct": correct,
                "samples": len(prompt_records),
            })
        correct = sum(1 for value in records if value == "correct")
        return {"correct": correct, "items": len(records),
                "answered": sum(1 for value in records
                                if value in ("correct", "wrong")),
                "distinct_problems": len({row["prompt_id"]
                                          for row in per_prompt}),
                "per_prompt": per_prompt,
                "verdicts": {value: records.count(value)
                             for value in set(records)}}

    if "base" in arms_to_score:
        lookup, hidden, _ = _sampling_lookup(
            args.student_dir, "cuda", items, spec, "base")
        summary["base"] = score("base", lookup)
    if "memory" in arms_to_score:
        lookup, _, ledger = _sampling_lookup(
            args.student_dir, "cuda", items, spec, "memory",
            memory_init=args.memory_init,
            memory_checkpoint=args.memory_checkpoint)
        summary["memory"] = score("memory", lookup)
        ledgers["memory"] = ledger
    for name, router_path in (("utility-outcome", args.router_outcome),
                              ("utility-ce", args.router_ce)):
        if name not in arms_to_score:
            continue
        lookup, _, ledger = _sampling_lookup(
            args.student_dir, "cuda", items, spec, "routed",
            memory_init=args.memory_init,
            memory_checkpoint=args.memory_checkpoint,
            router_path=router_path)
        summary[name] = score(name, lookup)
        ledgers[name] = ledger
    if "no-memory-gate" in arms_to_score:
        lookup, _, ledger = _sampling_lookup(
            args.student_dir, "cuda", items, spec, "zero",
            memory_init=args.memory_init,
            memory_checkpoint=args.memory_checkpoint,
            router_path=args.router_outcome)
        summary["no-memory-gate"] = score("no-memory-gate", lookup)
        ledgers["no-memory-gate"] = ledger
    if "gated-dense" in arms_to_score:
        lookup, _, _ = _sampling_lookup_gated_dense(
            args.student_dir, "cuda", items, spec, args.gated_dense_checkpoint)
        summary["gated-dense"] = score("gated-dense", lookup)

    for name in ("utility-outcome", "utility-ce"):
        if name in summary:
            ledger = ledgers.get(name) or {}
            summary[name]["read_requests"] = int(ledger.get("requested_rows", 0))
            summary[name]["read_bytes"] = int(ledger.get("bytes", 0))
            summary[name]["read_unique_rows"] = int(ledger.get("unique_rows", 0))
            summary[name]["read_repeated_rows"] = int(ledger.get("repeated_rows", 0))
            summary[name]["read_byte_basis"] = (
                "physical requests from the measured gather ledger; cache "
                f"model {ledger.get('cache_model')}")
            summary[name]["ledger"] = ledger
    if "memory" in summary:
        ledger = ledgers.get("memory") or {}
        summary["memory"]["read_requests"] = int(ledger.get("requested_rows", 0))
        summary["memory"]["read_bytes"] = int(ledger.get("bytes", 0))
        summary["memory"]["read_unique_rows"] = int(ledger.get("unique_rows", 0))
        summary["memory"]["read_repeated_rows"] = int(ledger.get("repeated_rows", 0))
        summary["memory"]["read_byte_basis"] = (
            "physical requests from the measured gather ledger; this is the "
            "dense always-read condition, one read at every token")
        summary["memory"]["ledger"] = ledger
    if "no-memory-gate" in summary:
        summary["no-memory-gate"]["read_requests"] = 0
        summary["no-memory-gate"]["read_bytes"] = 0
        summary["no-memory-gate"]["read_byte_basis"] = (
            "zero read at every selected token by construction")
    # The gated-dense control has no external store. It must not inherit the
    # memory arm's read counts: its read numbers are null, and it is used only
    # as the accuracy control.
    if "gated-dense" in summary:
        summary["gated-dense"]["read_requests"] = None
        summary["gated-dense"]["read_bytes"] = None
        summary["gated-dense"]["read_byte_basis"] = (
            "no external store; its dense adapter weights are not memory "
            "reads, so no byte claim is made for this control")
        summary["gated-dense"]["role"] = "accuracy attribution control"
    if "memory" in summary:
        summary["memory"]["role"] = "dense memory (always-read) byte control"

    # prompt-token selection audit: gate statistics and physical read cost of
    # the pre-read decision, measured before any generation read happens
    selection = {}
    if any(name in summary for name in ("utility-outcome", "utility-ce")):
        hidden = _holdout_hidden(args.student_dir, items, "cuda")
        for name, path in (("utility-outcome", args.router_outcome),
                           ("utility-ce", args.router_ce)):
            if name not in summary:
                continue
            router, _ = runner.load_router(path)
            examples = _selection_examples(router, hidden)
            audit = runner.selection_report(
                router, examples, threshold=args.threshold,
                row_bytes_by_tier={"table": runner.ROW_BYTES_BY_TIER["table"]})
            selection[name] = audit
            summary[name]["gate_mean"] = audit["gate_mean"]
            summary[name]["selected_fraction_prompt_tokens"] = (
                audit["selected_fraction"])
    # prompt-level paired rows: one per distinct problem, with every arm's
    # fraction correct for that problem. The bootstrap resamples these rows.
    prompt_rows = {}
    for name, value in summary.items():
        for row in value.get("per_prompt") or []:
            entry = prompt_rows.setdefault(row["prompt_id"], {
                "prompt_id": row["prompt_id"], "score": {}})
            entry["score"].update(row["score"])
    prompt_rows = [prompt_rows[key] for key in sorted(prompt_rows)]
    scored = runner.evaluate_arms(
        {name: value for name, value in summary.items()
         if name in ("base", "gated-dense", "memory", "utility-outcome",
                     "utility-ce", "no-memory-gate")},
        len(items) * spec.k, threshold=args.threshold,
        per_prompt=prompt_rows)
    out = args.out
    write_json(out, {
        "schema": "selvedge/p19.9-eval/v1",
        "created_utc": utcnow(), "stamp": stamp(), "revision": revision(),
        "rollouts": os.path.relpath(args.rollouts, ROOT),
        "rollouts_sha256": runner.sha256_file(args.rollouts),
        "routers": {"utility-outcome": args.router_outcome,
                    "utility-ce": args.router_ce},
        "arms": summary,
        "scored": scored,
        "selection_audit": selection,
        "read_ledgers": ledgers,
        "prompt_level_rows": len(prompt_rows),
        "statistics": ("prompt-level paired bootstrap over distinct problems; "
                       "the resampling unit is the problem, not the "
                       "continuation"),
        "screen_label": ("mechanism-validation only: a 64-prompt calibration "
                         "set is not a capability claim"),
        "byte_accounting": {
            "accuracy_control": "gated-dense (no external store, read_bytes null)",
            "byte_control": "memory (dense always-read)",
            "no_global_unique_byte_claim": (
                "the memory arm's read_bytes is the request-based physical "
                "count from its own gather ledger; the globally deduplicated "
                "unique-row number is reported as unique_bytes_modelled and is "
                "never a physical-byte claim"),
        },
        "evaluation": ("holdout prompts only; exact-answer accuracy over K "
                       "samples per prompt; read cost is the measured gather "
                       "ledger"),
        "queue": queue,
        "cuda": device_report,
        "container_memory": cgroup_memory(),
        "wall_seconds": round(time.time() - started, 1),
        "command": sys.argv,
    })
    print(f"p19.9-eval: {out}", flush=True)
    return 0


def _holdout_hidden(student_dir, items, device):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(student_dir,
                                              trust_remote_code=False)
    model = _load_model(student_dir, device)
    hidden = {}
    for item in items:
        hidden[item.prompt_id] = _capture_site_hidden(
            model, tokenizer, item.prompt, args_site(), device)
    del model
    if device == "cuda":
        import torch
        torch.cuda.empty_cache()
    return hidden


def _selection_examples(router, hidden):
    import torch
    examples = []
    for prompt_id, value in hidden.items():
        tokens = int(value.shape[0])
        metadata = runner.metadata_for_prompt(prompt_id, tokens)
        examples.append(runner.RouterExample(
            prompt_id=prompt_id, split="holdout", hidden=value,
            metadata=metadata,
            features=router.features(value, metadata),
            targets=torch.zeros(tokens), weights=torch.full((tokens,),
                                                           1.0 / tokens)))
    return examples


def _sampling_lookup_gated_dense(student_dir, device, items, spec,
                                 checkpoint):
    import torch
    from transformers import AutoTokenizer
    from frontier import arms, matched_controls
    tokenizer = AutoTokenizer.from_pretrained(student_dir,
                                              trust_remote_code=False)
    model = _load_model(student_dir, device)
    module = matched_controls.IdentityGatedDenseResidual(hidden=1024).to(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    module.load_state_dict(payload.get("arm_state", payload), strict=True)
    handles = arms.install_adapter(model, module, (args_site(),))
    results = {}
    for item in items:
        for sample in runner.sampling_plan(spec, item.prompt_id):
            results[(item.prompt_id, sample["sample_index"])] = _generate_one(
                model, tokenizer, item.prompt, sample, spec.max_new_tokens,
                device)
    arms.uninstall(handles)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return results, {}, None


def _write_holdout_hidden(hidden, out_path):
    import torch
    path = os.path.splitext(out_path)[0] + "-hidden.pt"
    torch.save(hidden, path)
    return path


def stage_screen(args) -> int:
    train = json.load(open(args.train, encoding="utf-8"))
    evaluation = json.load(open(args.eval, encoding="utf-8"))
    scored = evaluation.get("scored") or {}
    verdict = scored.get("verdict") or {}
    payload = {
        "schema": runner.SCREEN_RECEIPT_SCHEMA,
        "created_utc": utcnow(), "stamp": stamp(), "revision": revision(),
        "screen_label": ("mechanism-validation only: 64 calibration prompts, "
                         "not a capability claim"),
        "statistics": ("prompt-level paired bootstrap over distinct problems; "
                       "the resampling unit is the problem, not the "
                       "continuation"),
        "byte_accounting": (evaluation.get("byte_accounting") or {}),
        "train_receipt": os.path.relpath(args.train, ROOT),
        "train_receipt_sha256": runner.sha256_file(args.train),
        "eval_receipt": os.path.relpath(args.eval, ROOT),
        "eval_receipt_sha256": runner.sha256_file(args.eval),
        "routers": train.get("routers"),
        "arms": evaluation.get("arms"),
        "win_bar": scored,
        "verdict": verdict,
        "status": ("win" if verdict.get("chosen") else "null at 2M"),
        "next_route": verdict.get("next_route"),
    }
    write_json(args.out, payload)
    print(f"p19.9-screen: {args.out} status={payload['status']} "
          f"chosen={verdict.get('chosen')}")
    return 0


def stage_check(args) -> int:
    spec = spec_from_args(args)
    items, meta = runner.load_prompt_set(args.prompt_set, spec)
    registration = load_registration(args.registration)
    arms = registration.get("arms") or {}
    missing = [arm for arm in runner.ROUTER_ARMS if arm not in arms]
    if missing:
        raise SystemExit(f"p19.9: registration missing arms {missing}")
    if not registration.get("win_bar"):
        raise SystemExit("p19.9: registration has no win_bar")
    if not registration.get("next_route"):
        raise SystemExit("p19.9: registration has no next_route")
    prompt_set = (registration.get("prompt_set") or {}).get("path", PROMPT_SET)
    if not os.path.isfile(os.path.join(ROOT, prompt_set)):
        raise SystemExit("p19.9: registration prompt_set path is missing")
    if os.path.normpath(prompt_set) != os.path.normpath(args.prompt_set):
        raise SystemExit(
            f"p19.9: registration names prompt set {prompt_set}, runner uses "
            f"{args.prompt_set}")
    if (registration.get("prompt_set") or {}).get("sha256") != meta["sha256"]:
        raise SystemExit(
            "p19.9: prompt-set sha256 does not match the registration")
    print(f"p19.9-check: ok — {meta['items']} prompts "
          f"{meta['counts']}, sha256 {meta['sha256'][:16]}; "
          f"arms {runner.ROUTER_ARMS}")
    return 0


# ---------------------------------------------------------------------------
# cli


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt-set", default=PROMPT_SET)
    parser.add_argument("--registration", default=REGISTRATION)
    parser.add_argument("--mode", choices=runner.MODES,
                        default=runner.DEFAULT_SPEC["mode"])
    parser.add_argument("--k", type=int, default=runner.DEFAULT_SPEC["k"])
    parser.add_argument("--temperature", type=float,
                        default=runner.DEFAULT_SPEC["temperature"])
    parser.add_argument("--top-p", type=float,
                        default=runner.DEFAULT_SPEC["top_p"])
    parser.add_argument("--max-new-tokens", type=int,
                        default=runner.DEFAULT_SPEC["max_new_tokens"])
    parser.add_argument("--seed", type=int, default=runner.DEFAULT_SPEC["seed"])
    parser.add_argument("--threshold", type=float,
                        default=runner.DEFAULT_TRAINING["threshold"])
    parser.add_argument("--lr", type=float,
                        default=runner.DEFAULT_TRAINING["lr"])
    parser.add_argument("--epochs", type=int,
                        default=runner.DEFAULT_TRAINING["epochs"])
    parser.add_argument("--batch", type=int,
                        default=runner.DEFAULT_TRAINING["batch"])
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--student-dir", default=STUDENT_DIR)
    parser.add_argument("--memory-init", default=MEMORY_INIT)
    parser.add_argument("--memory-checkpoint", default=MEMORY_CHECKPOINT)
    parser.add_argument("--gated-dense-checkpoint", default=GATED_DENSE_CHECKPOINT)
    parser.add_argument("--bar-receipt", default=BAR_RECEIPT)
    parser.add_argument("--first-moment", choices=("int8", "bfloat16"),
                        default="int8")
    parser.add_argument("--vram-limit-gib", type=float, default=12.5)
    sub = parser.add_subparsers(dest="stage", required=True)

    frame = sub.add_parser("frame")
    frame.add_argument("--out", default="results/stage2/s52-p19.9-frame-"
                                         + "<stamp>.json")

    rollouts = sub.add_parser("rollouts")
    rollouts.add_argument("--out-dir", required=True)

    train = sub.add_parser("train")
    train.add_argument("--rollouts", required=True)
    train.add_argument("--out-dir", required=True)
    train.add_argument("--source", choices=("counterfactual-outcome",
                                            "counterfactual-token-loss", "both"),
                       default="both")

    evaluate = sub.add_parser("eval")
    evaluate.add_argument("--rollouts", required=True)
    evaluate.add_argument("--router-outcome", required=True)
    evaluate.add_argument("--router-ce", required=True)
    evaluate.add_argument("--arms", default=None)
    evaluate.add_argument("--out", required=True)

    screen = sub.add_parser("screen")
    screen.add_argument("--train", required=True)
    screen.add_argument("--eval", required=True)
    screen.add_argument("--out", required=True)

    sub.add_parser("check")
    args = parser.parse_args(argv)

    if args.stage == "frame" and "<stamp>" in (args.out or ""):
        args.out = args.out.replace("<stamp>", stamp())

    stages = {
        "frame": stage_frame, "rollouts": stage_rollouts, "train": stage_train,
        "eval": stage_eval, "screen": stage_screen, "check": stage_check,
    }
    return stages[args.stage](args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
