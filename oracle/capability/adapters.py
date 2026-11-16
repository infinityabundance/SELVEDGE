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
import subprocess
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
    raise AdapterError(f"unknown adapter kind {kind!r}")
