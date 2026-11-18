# S56 — the vLLM throughput image: what was tried, why it is not built

The evaluation receipt records `engine = transformers` and
`vllm_available = false`. That note is accurate and stays accurate: **no vLLM
throughput number is claimed anywhere in this repository.** This document records
the build attempt so the next session does not repeat the discovery, and names
the exact command that produces the image when the environment allows it.

## 1. Version arithmetic

Every scoring receipt is produced with the pinned anchor stack:

| component | version |
| --- | --- |
| torch | `2.15.0+cu132` |
| transformers | `5.23.0` |
| driver | `620.70.06` |

vLLM's PyPI metadata (fetched 2026-11-18, `requires_dist` of each release) pins
torch exactly:

| vLLM | torch | transformers |
| --- | --- | --- |
| 0.28.0–0.32.0 | `==2.13.0` | `<5.20.0` for 0.32 |
| 0.33.0–0.34.0 | `==2.14.0` | `<5.23.0,>=5.16.1` |

**No released vLLM requires torch 2.15.0**, and 0.34.0 excludes transformers
5.23.0. The newest vLLM that works at all is 0.34.0, on torch 2.14.0+cu132. It
therefore cannot share an image with the scoring stack, even though the driver
and CUDA runtime match.

The separate image (`docker/Dockerfile.vllm`,
`docker/requirements-vllm.txt`) pins:

```
torch==2.14.0+cu132
torchvision==0.29.0+cu132
torchaudio==2.11.0          # 2.11.0+cu132 does not exist; 2.11.1 does not satisfy ==2.11.0
transformers==5.22.2
vllm==0.34.0
```

It is a **throughput instrument only**. No score is produced in it, and its
numbers are never compared with a transformers score.

## 2. The attempt

Attempt 1 failed on the torchaudio pin: `No matching distribution found for
torchaudio==2.11.0+cu132` (the cu132 index has 2.11.0 unlocalised and
2.11.1+cu132). After pinning `torchaudio==2.11.0`, attempt 2 resolved every
version — torch 2.14.0+cu132, torchvision 0.29.0+cu132, torchaudio 2.11.0,
transformers 5.22.2 and vllm 0.34.0 — and then the pip layer ran out of disk:

```
ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device
```

at 217 s, after 220 downloads. The log is
`results/stage2/s56-vllm-build.log` (sha256
`0f74bd21d727fc9557f823c3825ec5e37ea54b0934e0f0b7db196e173022e61b`).

The cause is environmental: Docker's data root is
`/run/media/one/1tb_kingston/docker` on a **26 GiB tmpfs** with 23 GiB already
used by the existing court images and 2.8 GiB free. The vLLM stack is larger
than that. The receipt also records the 11.3 GiB of images not used by the
running job; reclaiming them would be an operator decision, and this session did
not remove another court's image while the GPU job was running.

The receipt is `results/stage2/s56-vllm-build.json` (`outcome: not_built`,
`image: null`, `pip_freeze: null`).

## 3. What was not done

- No vLLM throughput number was invented, estimated from another model, or
  copied from a vendor claim.
- The eval receipts still say `vllm_available: false`; that note was not edited.
- No scoring image was rebuilt or widened: the vLLM attempt is a separate image
  precisely so the anchor stack cannot drift.

## 4. The queued smoke command

`scripts/s56_vllm_smoke.py` loads the post-trained 0.8B and 2B chat checkpoints,
runs eight fixed prompts greedily with a 256-token cap, and measures tokens/s.
It **refuses to report vLLM numbers when vLLM is absent**; `--allow-transformers`
runs the same prompts on the pinned stack and labels the engine `transformers`.

Queued command (run only when the S55 data job has exited and the GPU is free):

```bash
SELVEDGE_MEM=12g SELVEDGE_CPUS=4 SELVEDGE_PIDS=2048 SELVEDGE_TMP=2g \
  scripts/court --gpu vllm -- python3 scripts/s56_vllm_smoke.py
```

Fallback, clearly labelled:

```bash
SELVEDGE_MEM=12g SELVEDGE_CPUS=4 SELVEDGE_PIDS=2048 SELVEDGE_TMP=2g \
  scripts/court --gpu eval -- python3 scripts/s56_vllm_smoke.py --allow-transformers
```

Plan only (CPU, no torch import):

```bash
SELVEDGE_MEM=2g scripts/court python -- python3 scripts/s56_vllm_smoke.py --dry-run
```

## 5. To build it later

1. Free at least 12 GiB in the Docker data root, or move the data root to a
   disk-backed filesystem (this requires restarting the daemon, so it must not
   be done while a court container is running).
2. `scripts/build-courts.sh vllm` — it builds with `--memory`/`--memory-swap`,
   records the image id and digest in `docker/IMAGES.lock`, and writes the full
   `pip freeze` to `docker/vllm-freeze.txt`.
3. Run the queued smoke command and write its receipt; only then may a
   throughput number be quoted, and only beside the engine label.
