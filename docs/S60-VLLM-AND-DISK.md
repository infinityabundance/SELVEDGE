# S60 — vLLM and Docker disk: assessment, not a change

Status: **assessment only.** Nothing was moved, pruned or restarted while the
stage-2b rollout job (`scripts/s59_student_rollouts.py`, container
`selvedge-oracle-court-20261119T030731Z-7004db`) is running. A Docker data-root
move needs a daemon restart, and a daemon restart would kill that job.

All measurements below were taken on 2026-11-19 with the job running.

## 1. What is actually on disk

```
docker info --format 'DockerRootDir={{.DockerRootDir}} Driver={{.Driver}}'
  DockerRootDir=/run/media/one/1tb_kingston/docker  Driver=overlay2

df -h
  /run                    26G   6.7G   19G   27%
  /                      504G   434G   44G   91%   (/dev/nvme1n1p3)
  /mnt/1tb_kingston      1.3T   1.2T   44G   97%   (/dev/nvme1n1p1)
  /mnt/2tb_crucial       1.8T   1.7T   86G   96%   (/dev/nvme0n1p1)

docker system df
  Images        6.887GB   active 1   reclaimable 841.3MB
  Containers         0B   active 1
  Build Cache    6.768GB            reclaimable 6.768GB
```

The path `/run/media/one/1tb_kingston/docker` looks like the 1.3 TiB NVMe but
is **on the 26 GiB `/run` tmpfs**. `/run` has 19 GiB free; Docker's own images
are 6.9 GiB and its build cache is a further 6.8 GiB. The earlier vLLM build
attempt died with `[Errno 28] No space left on device` at 2.8 GiB free
(`docs/S56-VLLM.md`, `results/stage2/s56-vllm-build.json`). The same constraint
is still present.

The repository itself is on `/mnt/1tb_kingston` (44 GiB free). The model cache
is 18 GiB under `~/.cache/selvedge/hf`. Neither is the vLLM bottleneck; the
Docker data root is.

## 2. What can be freed without touching the running job

| action | reclaims | touches the job? | judgement |
| --- | --- | --- | --- |
| `docker builder prune -f` | 6.8 GiB of build cache | no | **safe**, and the first thing to do when the rollout job exits |
| remove `selvedge/python-court` (624 MB) | 624 MB | no | not worth it; it is the CPU test court |
| remove `python:3.12-slim` (119 MB) | 119 MB | no | it is the base layer of both court builds; keep |
| remove `selvedge/eval-court` (6.38 GB) | 6.38 GB | no | **do not**; it is the S54 scoring instrument and will be needed for 0.5M/1M/2M/5M |
| remove `selvedge/oracle-court` (6.05 GB) | 6.05 GB | **yes** | forbidden; it is the running container's image |

Even pruning the build cache leaves ~26 GiB on a tmpfs, and the vLLM image
needs the torch 2.14 + CUDA 13.2 wheel set, the NVIDIA runtime libraries and
vLLM itself. The prior attempt was killed with 2.8 GiB free; the arithmetic does
not support building it in place.

## 3. The safe durable route: a second dockerd on the NVMe

Do **not** move the default data root. That needs
`systemctl stop docker`, which stops the running rollout container and any
scoring job; the task's own rule is not to do this while jobs run.

Instead, when a build is actually wanted:

1. Keep the default daemon and its data root exactly as they are.
2. Start a **second** daemon with an isolated data root and socket on the NVMe
   with the most free space. `/mnt/2tb_crucial` has 86 GiB; `/mnt/1tb_kingston`
   has 44 GiB. Both are real block devices, not tmpfs.
3. Build and run the vLLM image through that socket only
   (`DOCKER_HOST=unix:///run/docker-vllm.sock`). The main daemon, the court
   images and the running job never see the new daemon.
4. Give the second daemon its own bounded build cache and prune it after the
   image is built. The image is a throughput instrument, not a scoring image:
   no score is produced in it and its numbers are never compared with a
   transformers score (`docs/S56-VLLM.md` §1).

Sketch (root required; **not executed in S60**):

```sh
# 1. create the directory outside the project
sudo mkdir -p /mnt/2tb_crucial/docker-vllm
sudo dockerd \
  --data-root /mnt/2tb_crucial/docker-vllm \
  --host unix:///run/docker-vllm.sock \
  --pidfile /run/docker-vllm.pid
# 2. build in the second daemon only
DOCKER_HOST=unix:///run/docker-vllm.sock \
  docker build -f docker/Dockerfile.vllm -t selvedge/vllm:0.34.0 .
# 3. run the smoke through the second daemon, under the same OOM limits
DOCKER_HOST=unix:///run/docker-vllm.sock SELVEDGE_MEM=12g ...
```

The repository's `scripts/docker-court.sh` does not yet take a `DOCKER_HOST`
override; a later amendment must add one so the court receipt records the
second daemon. Until then this is a plan, not a runnable court.

## 4. What S60 actually did

- Measured the data root, the free space, the image sizes and the build cache.
- Did **not** prune, delete, restart or move anything while the rollout job runs.
- Recorded the safe order: `docker builder prune -f` after the job exits, then
  the second-data-root daemon if the vLLM image is still wanted.
- Did not claim a vLLM throughput number. `docs/S56-VLLM.md` and the eval
  receipts still say `vllm_available: false`.
