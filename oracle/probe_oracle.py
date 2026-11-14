#!/usr/bin/env python3
"""Oracle court probe: versions plus a pinned tiny tensor forward.

This is the oracle court's smoke test, not a benchmark. It proves that the
pinned Python stack imports and that the same pinned arithmetic runs on the
CPU and, when `--gpu` device passthrough worked, on the CUDA device inside the
container. Every input is written literally here; nothing is random.

Checks:

1. a 2x3 by 3x2 f32 matmul on the CPU equals the hand-computed expected
   matrix exactly (all values are exactly representable in f32);
2. SiLU on [-2, -1, 0, 1, 2] matches a documented f64 reference within a
   stated tolerance (exp is not exact);
3. with CUDA available, the same matmul on the GPU equals the CPU result
   bitwise, and SiLU agrees within the same tolerance;
4. when `--require-gpu` is passed, a missing CUDA device is a hard failure.

Usage:
  python3 oracle/probe_oracle.py [--json] [--require-gpu] [--out PATH]
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import sys

import numpy
import safetensors
import tokenizers
import torch
import transformers

A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
B = [[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]]
EXPECTED_MATMUL = [[11.0, 14.0], [24.5, 32.0]]

SILU_INPUT = [-2.0, -1.0, 0.0, 1.0, 2.0]
# silu(x) = x / (1 + exp(-x)) computed in f64 from the same inputs.
SILU_REFERENCE = [
    -0.23840584404423515,
    -0.2689414213699951,
    0.0,
    0.7310585786300049,
    1.7615941559557649,
]
SILU_TOLERANCE = 1e-6


def tensor_checks(device: torch.device) -> dict:
    a = torch.tensor(A, dtype=torch.float32, device=device)
    b = torch.tensor(B, dtype=torch.float32, device=device)
    product = a @ b
    expected = torch.tensor(EXPECTED_MATMUL, dtype=torch.float32, device=device)
    matmul_ok = bool(torch.equal(product, expected))

    x = torch.tensor(SILU_INPUT, dtype=torch.float32, device=device)
    silu = torch.nn.functional.silu(x)
    reference = torch.tensor(SILU_REFERENCE, dtype=torch.float32, device=device)
    silu_max_abs = float((silu - reference).abs().max().item())
    silu_ok = silu_max_abs <= SILU_TOLERANCE

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "device": str(device),
        "matmul_result": product.cpu().tolist(),
        "matmul_expected": EXPECTED_MATMUL,
        "matmul_bitwise_equal": matmul_ok,
        "silu_input": SILU_INPUT,
        "silu_result": silu.cpu().tolist(),
        "silu_max_abs_error": silu_max_abs,
        "silu_tolerance": SILU_TOLERANCE,
        "silu_ok": silu_ok,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="fail if no CUDA device is usable inside the container",
    )
    parser.add_argument("--out", help="also write the JSON report to this path")
    parser.add_argument("--stamp", help="receipt stamp recorded verbatim")
    parser.add_argument("--revision", help="git revision recorded verbatim")
    args = parser.parse_args(argv)

    report: dict = {
        "schema": "selvedge/oracle-probe/v1",
        "stamp": args.stamp,
        "revision": args.revision,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
            "safetensors": safetensors.__version__,
            "numpy": numpy.__version__,
        },
        "cuda_available": torch.cuda.is_available(),
        "cpu": tensor_checks(torch.device("cpu")),
    }

    gpu_ok = True
    if torch.cuda.is_available():
        try:
            capability = torch.cuda.get_device_capability(0)
            report["cuda"] = {
                "device_name": torch.cuda.get_device_name(0),
                "capability": f"{capability[0]}.{capability[1]}",
                "device_count": torch.cuda.device_count(),
                "checks": tensor_checks(torch.device("cuda:0")),
            }
            gpu_ok = report["cuda"]["checks"]["matmul_bitwise_equal"] and report["cuda"][
                "checks"
            ]["silu_ok"]
        except Exception as exc:  # pragma: no cover - hardware dependent
            report["cuda_error"] = f"{type(exc).__name__}: {exc}"
            gpu_ok = False
    else:
        report["cuda_error"] = "torch.cuda.is_available() is false"

    cpu_ok = report["cpu"]["matmul_bitwise_equal"] and report["cpu"]["silu_ok"]
    report["ok"] = bool(
        cpu_ok and gpu_ok and (torch.cuda.is_available() or not args.require_gpu)
    )
    if args.require_gpu and not torch.cuda.is_available():
        report["ok"] = False

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    if args.json:
        print(text)
    else:
        print(
            "oracle probe: torch={torch} cuda_runtime={cuda} transformers={tf} "
            "tokenizers={tok} safetensors={st} numpy={np}".format(
                torch=report["versions"]["torch"],
                cuda=report["versions"]["torch_cuda_runtime"],
                tf=report["versions"]["transformers"],
                tok=report["versions"]["tokenizers"],
                st=report["versions"]["safetensors"],
                np=report["versions"]["numpy"],
            )
        )
        print(
            f"cpu matmul bitwise={report['cpu']['matmul_bitwise_equal']} "
            f"{report['cpu']['matmul_result']}"
        )
        print(f"cpu silu max_abs_error={report['cpu']['silu_max_abs_error']:.3e}")
        if "cuda" in report:
            checks = report["cuda"]["checks"]
            print(
                f"gpu {report['cuda']['device_name']} capability={report['cuda']['capability']} "
                f"matmul bitwise={checks['matmul_bitwise_equal']} "
                f"silu max_abs_error={checks['silu_max_abs_error']:.3e}"
            )
        else:
            print(f"gpu unavailable: {report.get('cuda_error')}")
        print(f"ok={report['ok']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
