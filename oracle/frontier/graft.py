#!/usr/bin/env python3
"""Teacher-grafted row initialisation for the stage-2 memory (WS19 P19.2).

Method, in the order it is executed:

1. **Capture.** Run the frozen teacher (Qwen3.5-2B-Base by default) over the
   pinned graft split and record the hidden state at the memory site (the output
   of decoder layer `site`). Run the frozen student (0.8B-Base) over the same
   tokens and record its hidden state at the same relative site.
2. **Least squares.** Fit `W: R^teacher -> R^student` by ridge regression on the
   first 80% of the split; report relative Frobenius error on the held-out 20%.
   This is the documented teacher->student projection.
3. **PCA.** Centre the mapped teacher states `y = W h_teacher` and take the top
   `value_dim` principal directions. Report explained variance and the relative
   reconstruction error of `mu + B B^T (y - mu)`.
4. **Rows.** For every token and every one of the 8 heads, compute the address
   and accumulate the coefficient vector `c = B^T (y - mu)` into that row. A
   row is the mean of the coefficients that hashed to it; the count is stored.
   Empty rows stay zero (their output is the site mean, a defensible prior).
5. **Projections.** `key_proj = B^T`, `value_proj = B`, `out_bias = mu`, and the
   causal conv keeps its identity initialisation, so the grafted memory's
   initial output is `sum_i sigmoid(<B^T h, c_i>/sqrt(d)) (mu + B c_i)`.

The leakage guard lives in `stream.py`: the graft split is the first
`graft_tokens` of a stream whose every document was shingle-checked against the
cached test splits of the evaluation sets.

This script needs a GPU for the captures, but the linear algebra is exact and
could run on CPU; it is intentionally small enough that a 250k-token graft is
minutes, not hours.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.dirname(HERE)
sys.path.insert(0, ORACLE)

from frontier import arms, corrective, memory  # noqa: E402
from frontier.stream import sha256_file, load_stream  # noqa: E402

CACHE = os.environ.get("HF_HOME", "/hf-cache")
STUDENT_DIR = os.path.join(CACHE, "models", "Qwen", "Qwen3.5-0.8B-Base",
                           "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
TEACHER_DIR = os.path.join(CACHE, "models", "Qwen", "Qwen3.5-2B-Base",
                           "b1485b2fa6dfa1287294f269f5fb618e03d52d7c")
FOUR_B_DIR = os.path.join(CACHE, "models", "Qwen", "Qwen3.5-4B-Base",
                          "1001bb4d826a52d1f399e183466143f4da7b741b")
GIB = float(1 << 30)


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def revision() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "-C", os.path.dirname(os.path.dirname(HERE)), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return os.environ.get("SELVEDGE_REVISION", "unknown")


# ---------------------------------------------------------------------------
# capture


def _text_model(model):
    return model.model if hasattr(model.model, "layers") else model


@torch.no_grad()
def capture(model, tokens: np.ndarray, site: int, batch: int, device: str,
            vram_limit_gib: float) -> torch.Tensor:
    """Hidden state at the output of decoder layer `site`, `[N, hidden]` bf16 CPU."""
    layers = _text_model(model).layers
    if site < 0 or site >= len(layers):
        raise IndexError(f"site {site} outside 0..{len(layers) - 1}")
    holder: dict[str, torch.Tensor] = {}

    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        holder["hidden"] = hidden.detach()

    handle = layers[site].register_forward_hook(hook)
    first = None
    rows = []
    try:
        for start in range(0, tokens.size, batch):
            chunk = torch.from_numpy(
                np.asarray(tokens[start:start + batch], dtype=np.int64)).to(device)[None, :]
            model(input_ids=chunk, use_cache=False)
            hidden = holder["hidden"]
            if first is None:
                first = hidden.shape[-1]
            elif hidden.shape[-1] != first:
                raise RuntimeError("captured width changed between batches")
            rows.append(hidden[0].to(torch.bfloat16).cpu())
            if (start // batch) % 50 == 0:
                peak = torch.cuda.max_memory_allocated() / GIB
                if peak > vram_limit_gib:
                    raise RuntimeError(f"VRAM peak {peak:.2f} GiB over limit "
                                       f"{vram_limit_gib:.2f}")
                print(f"graft: captured {start + chunk.shape[1]}/{tokens.size} "
                      f"peak={peak:.2f}GiB", flush=True)
    finally:
        handle.remove()
    return torch.cat(rows, dim=0)


# ---------------------------------------------------------------------------
# fits


def fit_ridge(teacher: torch.Tensor, student: torch.Tensor, ridge: float = 1e-3,
              device: str = "cuda", chunk: int = 8192) -> tuple[torch.Tensor, dict]:
    """`W` minimising ||student - teacher @ W||^2 + ridge * ||W||^2."""
    width_t = teacher.shape[1]
    width_s = student.shape[1]
    split = int(teacher.shape[0] * 0.8)
    A = torch.zeros(width_t, width_t, dtype=torch.float32, device=device)
    b = torch.zeros(width_t, width_s, dtype=torch.float32, device=device)
    count = 0
    for start in range(0, split, chunk):
        t = teacher[start:start + chunk].to(device, torch.float32)
        s = student[start:start + chunk].to(device, torch.float32)
        A += t.T @ t
        b += t.T @ s
        count += t.shape[0]
    trace = float(torch.diagonal(A).mean())
    lam = ridge * trace
    A += torch.eye(width_t, device=device) * lam
    W = torch.linalg.solve(A, b)
    # held-out relative error
    err_sq = 0.0
    ref_sq = 0.0
    with torch.no_grad():
        for start in range(split, teacher.shape[0], chunk):
            t = teacher[start:start + chunk].to(device, torch.float32)
            s = student[start:start + chunk].to(device, torch.float32)
            pred = t @ W
            err_sq += float(((pred - s) ** 2).sum())
            ref_sq += float((s ** 2).sum())
    report = {
        "fit_tokens": int(split),
        "heldout_tokens": int(teacher.shape[0] - split),
        "teacher_width": int(width_t),
        "student_width": int(width_s),
        "ridge_lambda": lam,
        "ridge_fraction": ridge,
        "heldout_relative_frobenius_error": (err_sq / ref_sq) ** 0.5,
        "solver": "normal equations, torch.linalg.solve, float32",
    }
    return W, report


def fit_pca(targets: torch.Tensor, value_dim: int, device: str = "cuda",
            chunk: int = 16384) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Mean and top-`value_dim` principal directions of the mapped states."""
    width = targets.shape[1]
    total = torch.zeros(width, dtype=torch.float64, device=device)
    for start in range(0, targets.shape[0], chunk):
        total += targets[start:start + chunk].to(device, torch.float64).sum(0)
    mean = (total / targets.shape[0]).to(torch.float32)
    covariance = torch.zeros(width, width, dtype=torch.float64, device=device)
    for start in range(0, targets.shape[0], chunk):
        y = targets[start:start + chunk].to(device, torch.float32) - mean
        covariance += (y.T @ y).to(torch.float64)
    covariance /= max(1, targets.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    basis = eigenvectors[:, order[:value_dim]].to(torch.float32)
    keep = eigenvalues[:value_dim].clamp(min=0).sum()
    total_var = eigenvalues.clamp(min=0).sum()
    # sign convention: largest absolute component of each direction positive
    pivot = basis.abs().argmax(dim=0)
    signs = torch.sign(basis[pivot, torch.arange(value_dim, device=device)])
    signs[signs == 0] = 1
    basis = basis * signs[None, :]
    report = {
        "value_dim": int(value_dim),
        "explained_variance": float(keep / total_var),
        "total_variance": float(total_var),
        "mean_norm": float(mean.norm()),
        "basis_shape": list(basis.shape),
        "eigensolver": "torch.linalg.eigh on the float64 covariance",
    }
    return mean, basis, report


# ---------------------------------------------------------------------------
# row grafting


def graft_rows(tokens: torch.Tensor, coefficients: torch.Tensor, rows: int,
               site: int, device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """Mean coefficient per address for each of the 8 heads."""
    table = torch.zeros(rows, coefficients.shape[1], dtype=torch.float64,
                        device=device)
    counts = torch.zeros(rows, dtype=torch.int64, device=device)
    coefficients = coefficients.to(device, torch.float64)
    batch = tokens[None, :] if tokens.dim() == 1 else tokens
    batch = batch.to(device)
    if batch.shape[0] != 1:
        raise ValueError("graft_rows expects one sequence")
    for head, order in enumerate(memory.ORDERS):
        addresses = memory.address(batch, site, order, head, rows)[0]
        table.index_add_(0, addresses, coefficients)
        counts.index_add_(0, addresses, torch.ones_like(addresses))
    used = counts > 0
    table[used] /= counts[used].to(torch.float64)[:, None]
    return table.to(torch.float32).cpu(), counts.cpu()


# ---------------------------------------------------------------------------
# reconstruction


@torch.no_grad()
def reconstruction_report(module: memory.ConditionalMemory, targets: torch.Tensor,
                          hidden_source: torch.Tensor, tokens: torch.Tensor,
                          device: str, sample: int = 4096) -> dict:
    """Step-0 behaviour: rows from `targets`, gates from the real student state."""
    module.eval()
    module = module.to(device)
    take = min(sample, tokens.numel())
    tokens_sample = tokens[-take:].to(device)
    y = hidden_source[-take:].to(device, torch.float32)
    target = targets[-take:].to(device, torch.float32)
    module.context.observe(tokens_sample[None, :])
    out = module(y[None, :, :])[0].float()
    relative = float((out - y).norm() / y.norm())
    cosine = torch.nn.functional.cosine_similarity(out, y, dim=-1).mean()
    # Recompute the internals once to report the gate and magnitude at step 0.
    addresses = memory.support(tokens_sample[None, :], module.sites[0],
                               module.rows)
    values = module.tables[0][addresses].float()          # [1,T,8,VD]
    flat = values.reshape(1, values.shape[1], -1)
    flat = module.conv[0](flat)
    values = flat.reshape(values.shape)[0]                # [T,8,VD]
    key = module.key_proj[0](y)
    score = (values * key[:, None, :]).sum(-1) / (module.value_dim ** 0.5)
    gate = torch.sigmoid(score)
    projected = module.value_proj[0](values)
    read_norms = (gate[..., None] * projected).norm(dim=-1)
    per_head = {}
    for head, order in enumerate(memory.ORDERS):
        addresses = memory.address(tokens_sample[None, :], module.sites[0], order,
                                   head, module.rows)[0]
        row = module.tables[0][addresses].float()
        predicted = module.value_proj[0](row) + module.out_bias[0]
        per_head[f"order{order}_head{head}"] = float(
            (predicted - y).norm() / y.norm())
    mean_row = module.out_bias[0].expand_as(y)
    return {
        "sample_tokens": int(take),
        "hidden_source": "the frozen student's own hidden state at the site",
        "memory_output_relative_frobenius_error": relative,
        "memory_output_mean_cosine": float(cosine),
        "hidden_norm_mean": float(y.norm(dim=-1).mean()),
        "memory_output_norm_mean": float(out.norm(dim=-1).mean()),
        "memory_output_to_hidden_norm_ratio": float(
            out.norm(dim=-1).mean() / y.norm(dim=-1).mean()),
        "gate_mean": float(gate.mean()),
        "gate_min": float(gate.min()),
        "gate_max": float(gate.max()),
        "per_read_output_norm_mean": float(read_norms.mean()),
        "per_head_row_relative_error": per_head,
        "mean_only_relative_error": float((mean_row - y).norm() / y.norm()),
        "note": "per-head error uses one row + the site mean; the memory output "
                "is the gated sum of all 8 reads plus the mean once",
    }


def chunked_relative_error(predicted: torch.Tensor, target: torch.Tensor,
                           chunk: int = 16384) -> float:
    err = 0.0
    ref = 0.0
    for start in range(0, target.shape[0], chunk):
        p = predicted[start:start + chunk].float()
        t = target[start:start + chunk].float()
        err += float(((p - t) ** 2).sum())
        ref += float((t ** 2).sum())
    return (err / ref) ** 0.5


def pca_relative_error(module: memory.ConditionalMemory,
                       coefficients: torch.Tensor, basis: torch.Tensor,
                       target: torch.Tensor, chunk: int = 16384) -> float:
    """Error of `mu + B c` against the mapped teacher state, bounded memory."""
    b = basis.detach().cpu()
    mu = module.out_bias[0].detach().cpu()
    err = 0.0
    ref = 0.0
    for start in range(0, target.shape[0], chunk):
        pred = coefficients[start:start + chunk].float() @ b.T + mu
        t = target[start:start + chunk].float()
        err += float(((pred - t) ** 2).sum())
        ref += float((t ** 2).sum())
    return (err / ref) ** 0.5


# ---------------------------------------------------------------------------
# command


def run(args) -> int:
    started = time.time()
    torch.use_deterministic_algorithms(True)
    device = "cuda"
    if not torch.cuda.is_available():
        raise SystemExit("graft: CUDA is required; run through scripts/court --gpu oracle")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print(f"graft: gpu free={free_bytes / GIB:.2f}GiB total={total_bytes / GIB:.2f}GiB",
          flush=True)
    if args.vram_limit_gib > total_bytes / GIB:
        raise SystemExit("graft: vram limit exceeds the device")

    manifest = json.load(open(args.stream_manifest))
    if manifest.get("schema") != "selvedge/stage2-stream/v1":
        raise SystemExit("graft: unexpected stream schema")
    got = sha256_file(manifest["stream_path"])
    if got != manifest["sha256"]:
        raise SystemExit(f"graft: stream sha mismatch {got} != {manifest['sha256']}")
    stream = load_stream(manifest["stream_path"])
    graft_tokens = int(manifest["graft_tokens"])
    tokens_np = stream[:graft_tokens]
    tokens = torch.from_numpy(tokens_np.astype(np.int64))
    tokenizer_sha = manifest["tokenizer"]["tokenizer_sha256"]
    print(f"graft: {tokens.numel()} tokens from {manifest['stream_path']}", flush=True)

    disk_free = shutil.disk_usage(os.path.dirname(STUDENT_DIR)).free
    disk_report = {
        "cache_filesystem_free_bytes": int(disk_free),
        "teacher_2b_required_bytes": 4_548_221_488,
        "teacher_4b_required_bytes": 9_319_828_096,
        "teacher_4b_cached": os.path.isdir(FOUR_B_DIR),
        "teacher_4b_fits_now": bool(disk_free > 9_319_828_096 + (1 << 30)),
    }

    # -- teacher capture -----------------------------------------------------
    print("graft: loading teacher", args.teacher_dir, flush=True)
    teacher = arms.load_model(args.teacher_dir, dtype=torch.bfloat16, device=device)
    teacher_params = sum(parameter.numel() for parameter in teacher.parameters())
    torch.cuda.reset_peak_memory_stats()
    teacher_hidden = capture(teacher, tokens_np, args.site, args.batch, device,
                             args.vram_limit_gib)
    teacher_peak = torch.cuda.max_memory_allocated() / GIB
    teacher_width = teacher_hidden.shape[1]
    del teacher
    torch.cuda.empty_cache()
    print(f"graft: teacher hidden {tuple(teacher_hidden.shape)} peak={teacher_peak:.2f}GiB",
          flush=True)

    # -- student capture -----------------------------------------------------
    print("graft: loading student", args.student_dir, flush=True)
    student = arms.load_model(args.student_dir, dtype=torch.bfloat16, device=device)
    student_params = sum(parameter.numel() for parameter in student.parameters())
    torch.cuda.reset_peak_memory_stats()
    student_hidden = capture(student, tokens_np, args.site, args.batch, device,
                             args.vram_limit_gib)
    student_peak = torch.cuda.max_memory_allocated() / GIB
    student_width = student_hidden.shape[1]
    print(f"graft: student hidden {tuple(student_hidden.shape)} "
          f"peak={student_peak:.2f}GiB", flush=True)

    # -- projection and PCA --------------------------------------------------
    W, ls_report = fit_ridge(teacher_hidden, student_hidden, args.ridge, device)
    mapped = torch.empty_like(student_hidden)
    for start in range(0, mapped.shape[0], 16384):
        chunk = teacher_hidden[start:start + 16384].to(device, torch.float32)
        mapped[start:start + 16384] = (chunk @ W).to(torch.bfloat16).cpu()
    ls_report["mapped_state_relative_error"] = chunked_relative_error(
        mapped, student_hidden)
    mean, basis, pca_report = fit_pca(mapped, args.value_dim, device)
    del student
    torch.cuda.empty_cache()

    # -- rows ----------------------------------------------------------------
    coefficients = torch.empty(mapped.shape[0], args.value_dim, dtype=torch.bfloat16)
    for start in range(0, mapped.shape[0], 16384):
        y = mapped[start:start + 16384].to(device, torch.float32) - mean
        coefficients[start:start + 16384] = (y @ basis).to(torch.bfloat16).cpu()
    rows, counts = graft_rows(tokens, coefficients, args.rows, args.site, device)
    used = counts > 0
    used_rows = int(used.sum())
    occupancy = counts[used].float()
    row_report = {
        "rows": int(args.rows),
        "rows_with_at_least_one_token": used_rows,
        "empty_rows": int(args.rows - used_rows),
        "occupancy_fraction": used_rows / args.rows,
        "max_count": int(counts.max()),
        "mean_count_nonempty": float(occupancy.mean()),
        "median_count_nonempty": float(occupancy.median()),
        "read_identities": int(tokens.numel() * len(memory.ORDERS)),
        "collision_ratio": float(1.0 - used_rows / (tokens.numel() * len(memory.ORDERS))),
        "row_dtype": "float32 for training; bf16 is the deployment size",
    }

    # -- install -------------------------------------------------------------
    if args.corrective:
        # P19.2b: the representation fit is only an initialisation for the
        # rows and projections. The gate is exactly zero, no constant offset is
        # installed, and the training objective in oracle/frontier/corrective.py
        # supervises the teacher's advantage, not the mapped hidden state.
        module = corrective.CorrectiveMemory(
            hidden=student_width, value_dim=args.value_dim, rows=args.rows,
            sites=(args.site,), conv_kernel=args.conv_kernel,
            table_dtype=torch.float32, seed=args.seed)
        with torch.no_grad():
            module.set_grafted_rows(0, rows, counts)
            module.key_proj[0].weight.copy_(basis.T)
            module.value_proj[0].weight.copy_(basis)
            module.out_bias[0].zero_()
        recon = corrective.corrective_init_report(
            module, student_hidden, tokens, sample=args.recon_sample)
        recon["pca_state_relative_error"] = pca_relative_error(
            module.core, coefficients, basis, mapped)
        recon["supervision"] = (
            "learned by corrective_objective (advantage-weighted KL + base "
            "preservation + gate interference); the rows are only initialised "
            "from the representation fit")
        recon["identity_check"] = (
            "gate alpha = tanh(0) = 0 exactly, so the untrained residual is zero "
            "and the model reproduces the base bit for bit (see the S42 test)")
    else:
        module = memory.ConditionalMemory(
            hidden=student_width, value_dim=args.value_dim, rows=args.rows,
            sites=(args.site,), conv_kernel=args.conv_kernel,
            table_dtype=torch.float32, seed=args.seed)
        with torch.no_grad():
            module.set_grafted_rows(0, rows, counts)
            module.key_proj[0].weight.copy_(basis.T)
            module.value_proj[0].weight.copy_(basis)
            module.out_bias[0].copy_(mean)

        recon = reconstruction_report(module, mapped, student_hidden, tokens,
                                      device, args.recon_sample)
        recon["pca_state_relative_error"] = pca_relative_error(
            module, coefficients, basis, mapped)
        recon["coefficient_relative_error"] = recon["pca_state_relative_error"]

    out_dir = os.path.join(args.out_dir, args.stamp)
    os.makedirs(out_dir, exist_ok=True)
    checkpoint = os.path.join(out_dir, "memory-init.pt")
    torch.save({
        "state_dict": module.state_dict(),
        "config": {
            "hidden": student_width,
            "value_dim": args.value_dim,
            "rows": args.rows,
            "sites": [args.site],
            "conv_kernel": args.conv_kernel,
            "orders": list(memory.ORDERS),
            "table_dtype": "float32",
            "seed": args.seed,
            "variant": "corrective" if args.corrective else "representation",
        },
        "graft": {
            "stream_sha256": manifest["sha256"],
            "graft_tokens": int(tokens.numel()),
            "site": args.site,
            "teacher": args.teacher_dir,
            "student": args.student_dir,
        },
        "graft_counts": counts,
    }, checkpoint)

    receipt = {
        "schema": "selvedge/stage2-graft/v1",
        "kind": "graft",
        "created_utc": utcnow(),
        "stamp": args.stamp,
        "revision": revision(),
        "stream": {
            "manifest": args.stream_manifest,
            "sha256": manifest["sha256"],
            "graft_tokens": int(tokens.numel()),
            "skipped_train_tokens": int(stream.size - tokens.numel()),
            "tokenizer_sha256": tokenizer_sha,
            "contamination": manifest["contamination"],
        },
        "models": {
            "teacher": {"dir": args.teacher_dir,
                        "parameters": int(teacher_params),
                        "hidden_at_site": int(teacher_width)},
            "student": {"dir": args.student_dir,
                        "parameters": int(student_params),
                        "hidden_at_site": int(student_width)},
        },
        "site": args.site,
        "variant": "corrective" if args.corrective else "representation",
        "geometry": module.active_parameter_report(),
        "projection": {"method": "ridge least squares teacher->student",
                       "W": ls_report},
        "pca": pca_report,
        "rows": row_report,
        "reconstruction": recon,
        "disk": disk_report,
        "determinism": {"torch_deterministic": True, "dtype": "bfloat16 capture, "
                        "float32 fits"},
        "vram": {
            "teacher_capture_peak_gib": teacher_peak,
            "student_capture_peak_gib": student_peak,
            "limit_gib": args.vram_limit_gib,
            "source": "torch.cuda.max_memory_allocated",
            "free_at_start_gib": free_bytes / GIB,
            "device_total_gib": total_bytes / GIB,
        },
        "artifacts": {
            "checkpoint": checkpoint,
            "checkpoint_sha256": sha256_file(checkpoint),
            "bytes": os.path.getsize(checkpoint),
        },
        "wall_seconds": round(time.time() - started, 3),
        "container_memory": cgroup_memory(),
        "command": sys.argv,
    }
    os.makedirs(args.receipt_dir, exist_ok=True)
    receipt_path = os.path.join(args.receipt_dir, f"stage2-graft-{args.stamp}.json")
    write_json(receipt_path, receipt)
    print(json.dumps({k: receipt[k] for k in ("projection", "pca", "rows",
                                              "reconstruction")}, indent=2))
    print(f"graft: checkpoint {checkpoint}", flush=True)
    print(f"graft: receipt {receipt_path}", flush=True)
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream-manifest", required=True)
    parser.add_argument("--student-dir", default=STUDENT_DIR)
    parser.add_argument("--teacher-dir", default=TEACHER_DIR)
    parser.add_argument("--site", type=int, default=memory.DEFAULT_SITE)
    parser.add_argument("--rows", type=int, default=memory.DEFAULT_ROWS)
    parser.add_argument("--value-dim", type=int, default=memory.DEFAULT_VALUE_DIM)
    parser.add_argument("--conv-kernel", type=int, default=memory.DEFAULT_CONV_KERNEL)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--corrective", action="store_true",
                        help="build a P19.2b corrective graft (zero gate, "
                             "advantage supervision, no constant offset)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recon-sample", type=int, default=4096)
    parser.add_argument("--vram-limit-gib", type=float, default=12.0)
    parser.add_argument("--out-dir", default="results/stage2/graft")
    parser.add_argument("--receipt-dir", default="results/stage2")
    args = parser.parse_args(argv)
    args.stamp = stamp()
    return run(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
