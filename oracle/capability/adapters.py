#!/usr/bin/env python3
"""Uniform arm adapters for the capability court.

Every arm — a Hugging Face/PyTorch checkpoint, the Rust runtime, a future module
set, or a scripted fake used by the CPU tests — is plugged in through one
interface and reports the same accounting fields. The court owns the prompt
templates, greedy decoding, stop sets, extraction, checking and scoring; an
adapter only turns prompts into text and reports what it measurably knows.

Two stages, deliberately separate:

* `generate` runs where the model lives (the oracle court, or the rust court for
  the Rust runtime) and writes model outputs to a JSON receipt;
* `check` runs where `cargo` lives (the rust court), after generation, and never
  gives the model a path to the tests.

That split is what makes "no access to the checker at inference" structural
rather than a promise: the checker process does not exist while the model runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field

#: Accounting fields the control matrix expects every arm to report or to
#: declare unavailable with a reason (P12.10).
ACCOUNTING_FIELDS = (
    "training_tokens",
    "optimizer_flops",
    "offline_teacher_construction_cost",
    "total_accessible_capacity",
    "active_unique_params",
    "gpu_residency_bytes",
    "host_residency_bytes",
    "measured_io",
    "task_success",
)


class AdapterError(RuntimeError):
    """The adapter cannot produce a measurement; fail rather than guess."""


@dataclass
class Request:
    item_id: str
    prompt: str
    max_new_tokens: int
    stops: tuple = ()
    seed: int = 20261115


@dataclass
class Generation:
    text: str
    finish_reason: str = "stop"
    tokens_out: int = 0
    latency_ms: int = 0
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "finish_reason": self.finish_reason,
            "tokens_out": self.tokens_out,
            "latency_ms": self.latency_ms,
            "extra": self.extra,
        }


class Adapter:
    """The interface every arm implements."""

    name = "adapter"
    kind = "abstract"
    #: Whether the arm may use external tools. Baselines must match it.
    tools = False
    #: Declared accounting values; None means "not measured", which the control
    #: matrix records with a reason rather than inventing a number.
    accounting: dict = {}
    null_reasons: dict = {}
    #: Provenance: source model revision and checkpoint digest when known.
    provenance: dict = {}

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "tools": self.tools,
            "accounting": {field: self.accounting.get(field)
                           for field in ACCOUNTING_FIELDS},
            "null_reasons": dict(self.null_reasons),
            "provenance": dict(self.provenance),
        }

    def generate(self, requests: list) -> list:
        raise NotImplementedError


class T2ArmAdapter(Adapter):
    """One trained T2 arm: the frozen 0.6B base plus its checkpointed modules.

    The arm is rebuilt with the same `build_arm`, seed and dispatch mask order the
    training loop used, then the checkpoint's `model` state dict is loaded, so the
    arm being scored is the arm that was trained. Active unique parameters come
    from the Rust admission record (the ledger), never from an assertion in this
    file.

    **Loading policy (S16C).** The checkpoint is not loaded with an unchecked
    `strict=False`. The allowlist is exact and derived, not hand-written:

    * expected *present* keys — exactly the arm's `trainable_names`, the
      parameters `build_arm` created as trainable by construction (for `H*` this
      still includes the router, which is frozen only on the gradient path);
    * expected *missing* keys — exactly the frozen base parameters, which are
      loaded from the pinned HF revision and are not part of the checkpoint;
    * expected unexpected keys — none.

    Any deviation refuses the load. When a training receipt is supplied it is
    also checked against the built arm: arm letter, config digest, admission
    identity, checkpoint name and digest, checkpoint step, and (for routed arms)
    the recorded dispatch mask placement from the P12.1 audit.
    """

    kind = "t2-arm"
    ROUTED_ARMS = ("X", "H", "H*")

    def __init__(self, arm: str, checkpoint: str, config: str, admission: str,
                 hf_root: str, device: str | None = None, batch: int = 4,
                 accounting: str | None = None, seed: int = 1001,
                 training_receipt: str | None = None,
                 mask_order: str = "legacy"):
        self.arm = arm
        self.checkpoint = checkpoint
        self.config = config
        self.admission = admission
        self.hf_root = hf_root
        self.device_name = device
        self.batch = batch
        self.accounting_path = accounting
        self.seed = seed
        self.training_receipt = training_receipt
        self.mask_order = mask_order
        self.name = f"t2-{arm}"
        self.tools = False
        self._loaded = None
        self._admission = None
        self._verification = None

    def _load(self):
        if self._loaded is not None:
            return self._loaded
        train_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "train")
        if train_dir not in sys.path:
            sys.path.insert(0, train_dir)
        import torch
        from transformers import AutoModelForCausalLM
        import harness
        from modeling import build_arm, load_admission

        config, admission = load_admission(self.config, self.admission)
        self._admission = admission
        device = self.device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        base_path = harness.model_path(self.hf_root, "0.6b")
        base = AutoModelForCausalLM.from_pretrained(
            base_path, dtype=torch.bfloat16, local_files_only=True,
            trust_remote_code=False, attn_implementation="sdpa")
        base.to(device)
        base.eval()
        arm_model = build_arm(base, self.arm, config, admission, self.seed,
                              profile="t2", mask_order=self.mask_order)
        arm_model.model.to(device)
        # Load on the host and copy into the device model. The checkpoint holds
        # ~2.5 GiB of fp32 trainable tensors; `map_location=device` would hold a
        # second device copy for the whole evaluation, the same defect the
        # streamed evaluation instrument fixed for the T2 arms.
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        self._verify_checkpoint(state, arm_model)
        self._verification = self._verify_receipt(state)
        arm_model.model.load_state_dict(state["model"], strict=False)
        arm_model.model.eval()
        tokenizer = harness.load_tokenizer(self.hf_root)
        self._loaded = (torch, tokenizer, arm_model.model, device, state)
        return self._loaded

    def _verify_checkpoint(self, state: dict, arm_model) -> None:
        """Refuse a checkpoint that is not this arm's, with the reason named."""
        problems = []
        recorded = state.get("model")
        if not isinstance(recorded, dict):
            raise AdapterError(f"{self.checkpoint}: no 'model' state dict")
        if state.get("arm") != self.arm:
            problems.append(f"checkpoint arm is {state.get('arm')!r}, "
                            f"the adapter was built for {self.arm!r}")
        expected = set(arm_model.trainable_names)
        if not expected:
            raise AdapterError(
                f"arm {self.arm!r} declares no trainable parameters; a checkpoint "
                "cannot be verified against an empty allowlist")
        loaded = set(recorded)
        missing = sorted(expected - loaded)
        unexpected = sorted(loaded - expected)
        if missing:
            problems.append(f"{len(missing)} trainable tensors are missing from the "
                            f"checkpoint (first: {missing[:5]})")
        if unexpected:
            problems.append(f"{len(unexpected)} tensors are not parameters of this arm "
                            f"(first: {unexpected[:5]})")
        shapes = {name: tuple(parameter.shape) for name, parameter
                  in arm_model.model.named_parameters() if name in expected}
        for name in sorted(expected & loaded):
            shape = tuple(recorded[name].shape)
            if shape != shapes.get(name):
                problems.append(f"{name} has shape {shape}, the arm expects "
                                f"{shapes.get(name)}")
                break
        if problems:
            raise AdapterError(
                f"{self.checkpoint} is not the checkpoint of arm {self.arm!r}: "
                + "; ".join(problems)
                + ". The allowlist is exact: the only keys a checkpoint may omit are "
                  "the frozen base parameters, and it may carry no others")

    def _verify_receipt(self, state: dict) -> dict:
        """Cross-check the checkpoint against the run that produced it."""
        if not self.training_receipt:
            return {"checked": False,
                    "reason": "no training receipt was supplied to the adapter"}
        if not os.path.isfile(self.training_receipt):
            raise AdapterError(f"training receipt {self.training_receipt} does not exist")
        with open(self.training_receipt, "r", encoding="utf-8") as handle:
            receipt = json.load(handle)
        admission = self._admission or {}
        problems = []
        if receipt.get("arm") != self.arm:
            problems.append(f"the receipt is for arm {receipt.get('arm')!r}")
        if not admission.get("admitted"):
            problems.append("the admission record does not admit this arm")
        if admission.get("arm") != self.arm:
            problems.append(f"the admission record is for arm {admission.get('arm')!r}")
        config_sha = (receipt.get("config") or {}).get("sha256")
        if config_sha != sha256_file(self.config):
            problems.append("the receipt's config digest does not match the config file")
        admission_path = (receipt.get("admission") or {}).get("path")
        if not admission_path or (os.path.normpath(admission_path)
                                  != os.path.normpath(self.admission)):
            problems.append(
                f"the receipt names admission record {admission_path!r}, not "
                f"{self.admission!r}")
        receipt_active = (receipt.get("admission") or {}).get("active_unique_params")
        if receipt_active != admission.get("active_unique_params"):
            problems.append("the receipt's admitted active-parameter count differs from "
                            "the admission record's")
        checkpoint = receipt.get("checkpoint") or {}
        if checkpoint.get("sha256") != sha256_file(self.checkpoint):
            problems.append("the checkpoint digest does not match the training receipt")
        if os.path.basename(str(checkpoint.get("path") or "")) != os.path.basename(
                self.checkpoint):
            problems.append("the checkpoint filename does not match the training receipt")
        step = _step_from_checkpoint_name(checkpoint.get("name") or checkpoint.get("path"))
        if step is None:
            problems.append("the training receipt names no step-NNNNNN checkpoint")
        elif state.get("step") != step:
            problems.append(f"the checkpoint is at step {state.get('step')!r}, the "
                            f"receipt names step {step}")
        if self.arm in self.ROUTED_ARMS:
            recorded_mask = receipt.get("dispatch_mask_order")
            if recorded_mask is None:
                problems.append(
                    "the training receipt records no dispatch mask placement (P12.1), "
                    "so a routed checkpoint's dispatch semantics are unverified")
            elif recorded_mask != self.mask_order:
                problems.append(f"the arm was trained with mask_order={recorded_mask!r}, "
                                f"the adapter is building {self.mask_order!r}")
        if problems:
            raise AdapterError(
                f"the checkpoint does not match its training receipt "
                f"{self.training_receipt}: " + "; ".join(problems))
        return {"checked": True, "receipt": self.training_receipt,
                "receipt_sha256": sha256_file(self.training_receipt),
                "step": state.get("step"), "mask_order": self.mask_order}

    def metadata(self) -> dict:
        _torch, _tokenizer, _model, _device, state = self._load()
        admission = self._admission or {}
        verification = self._verification or {"checked": False}
        declared = {}
        if self.accounting_path:
            with open(self.accounting_path, "r", encoding="utf-8") as handle:
                declared = json.load(handle)
        reasons = {
            "training_tokens": "not recorded in the arm's admission record",
            "optimizer_flops": "not recorded in the arm's admission record",
            "offline_teacher_construction_cost": "no teacher at inference",
            "host_residency_bytes": "not instrumented for the training-model path",
            "measured_io": "the PyTorch path does not expose device read counters",
            "task_success": "filled by the check stage",
            "gpu_residency_bytes": "measured after the first generation",
        }
        accounting = {
            "training_tokens": declared.get("training_tokens"),
            "optimizer_flops": declared.get("optimizer_flops"),
            "offline_teacher_construction_cost": declared.get(
                "offline_teacher_construction_cost"),
            # From the Rust ledger's admission record.
            "active_unique_params": admission.get("active_unique_params"),
            "total_accessible_capacity": admission.get("total_accessible"),
            "gpu_residency_bytes": None,
            "host_residency_bytes": None,
            "measured_io": declared.get("measured_io"),
            "task_success": None,
        }
        return {
            "name": self.name,
            "kind": self.kind,
            "tools": self.tools,
            "accounting": accounting,
            "null_reasons": {field: reasons[field] for field in accounting
                             if accounting[field] is None},
            "provenance": {
                "arm": self.arm,
                "checkpoint": self.checkpoint,
                "checkpoint_step": state.get("step"),
                "checkpoint_sha256": checkpoint_digest_file(self.checkpoint),
                "config": self.config,
                "admission": self.admission,
                "admission_stamp": admission.get("stamp"),
                "admission_revision": admission.get("revision"),
                "dispatch_mask_order": self.mask_order,
                "checkpoint_verification": verification,
                "source_model_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
            },
        }

    def generate(self, requests: list) -> list:
        _torch, tokenizer, model, device, _state = self._load()
        return greedy_generate(model, tokenizer, requests, device, self.batch)


class FakeAdapter(Adapter):
    """Scripted outputs for the CPU tests.

    `outputs` maps an item id to text, or to a callable receiving the request.
    `order_sensitive` makes the reply depend on its position, which the
    prompt-order-invariance check must catch.
    """

    kind = "fake"

    def __init__(self, outputs, name: str = "fake", accounting: dict | None = None,
                 tools: bool = False, order_sensitive: bool = False):
        self.outputs = outputs
        self.name = name
        self.tools = tools
        self.order_sensitive = order_sensitive
        self.accounting = dict(accounting or {})
        self.null_reasons = {
            field: "scripted fake model; no such measurement exists"
            for field in ACCOUNTING_FIELDS if field not in self.accounting
        }
        self.provenance = {"source_model_revision": None,
                           "checkpoint_sha256": None}

    def generate(self, requests: list) -> list:
        results = []
        for position, request in enumerate(requests):
            entry = self.outputs.get(request.item_id, "")
            if callable(entry):
                entry = entry(request)
            text = entry if not self.order_sensitive else (
                entry if position % 2 == 0 else "")
            results.append(Generation(text=text, tokens_out=len(text.split()),
                                      latency_ms=1))
        return results


class HFAdapter(Adapter):
    """A local Hugging Face/PyTorch checkpoint, greedy decoding, fixed seed."""

    kind = "hf"

    def __init__(self, model_path: str, revision: str | None = None,
                 device: str | None = None, dtype: str = "bfloat16",
                 batch: int = 4, tools: bool = False):
        self.model_path = model_path
        self.revision = revision
        self.device_name = device
        self.dtype_name = dtype
        self.batch = batch
        self.tools = tools
        self.name = f"hf:{os.path.basename(model_path)}"
        self._loaded = None
        self._metadata_cache = None

    def _load(self):
        if self._loaded is not None:
            return self._loaded
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = self.device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = getattr(torch, self.dtype_name)
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path, dtype=dtype, local_files_only=True,
            trust_remote_code=False, attn_implementation="sdpa")
        model.to(device)
        model.eval()
        self._loaded = (torch, tokenizer, model, device)
        return self._loaded

    def metadata(self) -> dict:
        torch = None
        try:
            import torch as torch_module
            torch = torch_module
        except ImportError as exc:  # pragma: no cover - the oracle court has torch
            raise AdapterError(f"torch is required for the HF adapter: {exc}")
        _torch, _tokenizer, model, device = self._load()
        parameters = sum(parameter.numel() for parameter in model.parameters())
        residency = {"value": None, "reason": "residency is measured after generation"}
        if torch.cuda.is_available():
            residency = {"value": int(torch.cuda.memory_allocated()),
                         "reason": "torch.cuda.memory_allocated after load",
                         "device": torch.cuda.get_device_name(0)}
        payload = {
            "name": self.name,
            "kind": self.kind,
            "tools": self.tools,
            "accounting": {
                # A frozen dense student: its active set is every parameter, and
                # the number comes from the checkpoint, not from an assertion.
                "active_unique_params": parameters,
                "total_accessible_capacity": parameters,
                "training_tokens": None,
                "optimizer_flops": None,
                "offline_teacher_construction_cost": None,
                "gpu_residency_bytes": (residency["value"] if device == "cuda" else 0),
                "host_residency_bytes": None,
                "measured_io": None,
                "task_success": None,
            },
            "null_reasons": {
                "training_tokens": "the checkpoint's training budget is not recorded here",
                "optimizer_flops": "not derivable from a frozen checkpoint",
                "offline_teacher_construction_cost": "no teacher was used at inference",
                "host_residency_bytes": "not instrumented for this adapter",
                "measured_io": "the PyTorch path does not expose device read counters",
                "task_success": "filled by the check stage",
            },
            "provenance": {
                "source_model_revision": self.revision,
                "checkpoint_path": self.model_path,
                "checkpoint_sha256": checkpoint_digest(self.model_path),
                "gpu_residency": residency,
            },
        }
        self._metadata_cache = payload
        return payload

    def generate(self, requests: list) -> list:
        torch, tokenizer, model, device = self._load()
        results = []
        for start in range(0, len(requests), self.batch):
            chunk = requests[start:start + self.batch]
            prompts = [request.prompt for request in chunk]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True,
                                truncation=True, max_length=2048,
                                padding_side="left")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            max_new = max(request.max_new_tokens for request in chunk)
            stops = sorted({stop for request in chunk for stop in request.stops})
            torch.manual_seed(chunk[0].seed)
            began = time.monotonic_ns()
            with torch.inference_mode():
                generated = model.generate(
                    **encoded, max_new_tokens=max_new, do_sample=False, num_beams=1,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    stop_strings=stops or None, tokenizer=tokenizer,
                )
            elapsed = (time.monotonic_ns() - began) // 1_000_000
            new_tokens = generated[:, encoded["input_ids"].shape[1]:]
            for offset, text in enumerate(
                    tokenizer.batch_decode(new_tokens, skip_special_tokens=True)):
                results.append(Generation(
                    text=text, finish_reason="stop",
                    tokens_out=int(new_tokens.shape[1]),
                    latency_ms=elapsed // max(1, len(chunk))))
        return results


def greedy_generate(model, tokenizer, requests: list, device, batch: int) -> list:
    """Greedy decoding with fixed seed, fixed stop strings and no sampling."""
    import torch

    results = []
    for start in range(0, len(requests), batch):
        chunk = requests[start:start + batch]
        prompts = [request.prompt for request in chunk]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True,
                            truncation=True, max_length=4096, padding_side="left")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        max_new = max(request.max_new_tokens for request in chunk)
        stops = sorted({stop for request in chunk for stop in request.stops})
        torch.manual_seed(chunk[0].seed)
        began = time.monotonic_ns()
        with torch.inference_mode():
            generated = model.generate(
                **encoded, max_new_tokens=max_new, do_sample=False, num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stop_strings=stops or None, tokenizer=tokenizer,
            )
        elapsed = (time.monotonic_ns() - began) // 1_000_000
        new_tokens = generated[:, encoded["input_ids"].shape[1]:]
        for text in tokenizer.batch_decode(new_tokens, skip_special_tokens=True):
            results.append(Generation(
                text=text, finish_reason="stop",
                tokens_out=int(new_tokens.shape[1]),
                latency_ms=elapsed // max(1, len(chunk))))
    return results


class RustRuntimeAdapter(Adapter):
    """The Rust runtime in exact mode, one subprocess per prompt.

    The runtime holds the backbone and its bundle cache in host memory, so its
    address-space bound is the court's own memory limit (explicit, and verified
    by the court), not the 3 GiB checker envelope.
    """

    kind = "rust-runtime"

    def __init__(self, binary: str, model_dir: str, bundle: str, threads: int = 8,
                 cache_bytes: int | None = None, mode: str = "exact"):
        self.binary = binary
        self.model_dir = model_dir
        self.bundle = bundle
        self.threads = threads
        self.cache_bytes = cache_bytes
        self.mode = mode
        self.name = "rust-runtime-exact"
        self.tools = False
        self._summaries = []
        self.accounting = {}
        self.null_reasons = {}
        court_mem = os.environ.get("SELVEDGE_COURT_MEM_BYTES")
        if not court_mem:
            raise AdapterError(
                "the Rust runtime adapter needs the court memory pin to size its "
                "own address-space bound (AGENTS.md directive 2)")
        self.address_space_bytes = int(court_mem)

    def _run(self, request: Request) -> dict:
        command = [self.binary, "run", "--model", self.model_dir,
                   "--bundle", self.bundle, "--mode", self.mode,
                   "--prompt", request.prompt,
                   "--max-new-tokens", str(request.max_new_tokens),
                   "--threads", str(self.threads), "--json"]
        if self.cache_bytes:
            command.extend(["--cache-bytes", str(self.cache_bytes), "--policy", "lru"])

        def limit() -> None:
            import resource
            resource.setrlimit(resource.RLIMIT_AS,
                               (self.address_space_bytes, self.address_space_bytes))

        completed = subprocess.run(command, capture_output=True, text=True,
                                   timeout=3600, preexec_fn=limit)
        if completed.returncode != 0:
            raise AdapterError(f"runtime failed ({completed.returncode}): "
                               f"{completed.stderr[-800:]}")
        payload = json.loads(completed.stdout)
        self._summaries.append(payload)
        return payload

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "tools": self.tools,
            "accounting": {
                "active_unique_params": None,
                "total_accessible_capacity": None,
                "training_tokens": None,
                "optimizer_flops": None,
                "offline_teacher_construction_cost": None,
                "gpu_residency_bytes": 0,
                "host_residency_bytes": None,
                "measured_io": None,
                "task_success": None,
            },
            "null_reasons": {
                "active_unique_params": "measured per run; see the runtime summary",
                "total_accessible_capacity": "measured per run; see the runtime summary",
                "training_tokens": "the runtime executes a checkpoint it did not train",
                "optimizer_flops": "not derivable from a frozen checkpoint",
                "offline_teacher_construction_cost": "no teacher is used at inference",
                "host_residency_bytes": "not instrumented in this adapter",
                "measured_io": "summarised after generation from the runtime's own counters",
                "task_success": "filled by the check stage",
            },
            "provenance": {
                "source_model_revision": None,
                "checkpoint_path": self.model_dir,
                "bundle": self.bundle,
                "checkpoint_sha256": checkpoint_digest(self.model_dir),
            },
        }

    def generate(self, requests: list) -> list:
        results = []
        for request in requests:
            began = time.monotonic_ns()
            summary = self._run(request)
            elapsed = (time.monotonic_ns() - began) // 1_000_000
            generated = summary.get("generated_tokens") or []
            totals = summary.get("totals") or {}
            results.append(Generation(
                text=summary.get("generated_text", ""),
                finish_reason="stop",
                tokens_out=len(generated),
                latency_ms=elapsed,
                extra={
                    "sequence_digest": summary.get("sequence_digest"),
                    "active_unique_max": totals.get("active_unique_max"),
                    "total_accessible_elements": summary.get("total_accessible_elements"),
                    "bytes_device": totals.get("bytes_device"),
                    "decode_p95_ns": (summary.get("decode") or {}).get("p95_ns"),
                    "decode_tokens_per_second": summary.get("decode_tokens_per_second"),
                },
            ))
        return results


# ---------------------------------------------------------------------------


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_digest_file(path: str) -> str | None:
    if not os.path.isfile(path):
        return None
    return sha256_file(path)


def checkpoint_digest(path: str) -> str | None:
    """SHA-256 over a checkpoint's files, or None when the path is not a tree."""
    if not os.path.isdir(path):
        return None
    digest = hashlib.sha256()
    count = 0
    for base, _dirs, files in os.walk(path):
        for name in sorted(files):
            if not name.endswith((".safetensors", ".bin", ".pt", ".json")):
                continue
            full = os.path.join(base, name)
            relative = os.path.relpath(full, path).replace(os.sep, "/")
            digest.update(relative.encode("utf-8") + b"\0")
            with open(full, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            count += 1
    return digest.hexdigest() if count else None


def _step_from_checkpoint_name(name) -> int | None:
    """The step encoded in a `step-001024.pt` filename, or None."""
    if not name:
        return None
    match = re.search(r"step-(\d+)", os.path.basename(str(name)))
    return int(match.group(1)) if match else None


def adapter_from_spec(spec: dict) -> Adapter:
    """Build an adapter from a JSON spec (the CLI's `--adapter-spec`)."""
    kind = spec.get("kind")
    if kind == "fake":
        outputs = spec["outputs"]
        path = spec.get("outputs_path")
        if path:
            with open(path, "r", encoding="utf-8") as handle:
                outputs = json.load(handle)
        return FakeAdapter(outputs, name=spec.get("name", "fake"),
                           accounting=spec.get("accounting"),
                           tools=bool(spec.get("tools", False)),
                           order_sensitive=bool(spec.get("order_sensitive", False)))
    if kind == "hf":
        return HFAdapter(spec["model_path"], revision=spec.get("revision"),
                         device=spec.get("device"), dtype=spec.get("dtype", "bfloat16"),
                         batch=int(spec.get("batch", 4)),
                         tools=bool(spec.get("tools", False)))
    if kind == "rust-runtime":
        return RustRuntimeAdapter(spec["binary"], spec["model_dir"], spec["bundle"],
                                  threads=int(spec.get("threads", 8)),
                                  cache_bytes=spec.get("cache_bytes"),
                                  mode=spec.get("mode", "exact"))
    if kind == "t2-arm":
        return T2ArmAdapter(spec["arm"], spec["checkpoint"], spec["config"],
                            spec["admission"], spec.get("hf_root", "/hf-cache"),
                            device=spec.get("device"),
                            batch=int(spec.get("batch", 4)),
                            accounting=spec.get("accounting"),
                            training_receipt=spec.get("training_receipt"),
                            mask_order=spec.get("mask_order", "legacy"))
    raise AdapterError(f"unknown adapter kind {kind!r}")
