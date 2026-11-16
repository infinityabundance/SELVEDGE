#!/usr/bin/env bash
# Post-H* pipeline launcher — the GPU window is used only for measurements.
#
#   scripts/post-hstar.sh [--resume|--fresh] [--stage NAME ...] [--dry-run]
#                         [--runner systemd|direct] [--host-probe FILE]
#                         [--repo-root DIR] [--court PATH]
#
# The stage logic, the guards and the state machine live in
# `scripts/post_hstar.py` (CPU-testable, no Docker required to test). This file
# is the operator's entry point and enforces the launch conditions the operator
# directives require:
#
#   * every stage runs through `scripts/court` (directive 1), never on the host;
#   * every GPU stage carries an explicit `SELVEDGE_MEM` derived from that
#     workload's measured peak, and is launched under
#     `systemd-run --user --collect --unit=...` with MemoryMax/MemorySwapMax set
#     to the same value (directive 2); `--runner direct` is the documented
#     exception for a host without a user systemd and for tests, and it is
#     recorded in the state;
#   * the preflight refuses to start while a heavy GPU process or a `selvedge-`
#     container is alive, and refuses when `MemAvailable` cannot cover the
#     largest stage limit plus a reserve;
#   * one stage at a time; state is written after every transition and a resume
#     re-runs only stages that are not `done`.
#
# It never touches a running training container. Poll it read-only with
# `scripts/poll-t2-pilot.sh`.
set -Eeuo pipefail
IFS=$'\n\t'

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
script_repo=$(cd -- "$script_dir/.." && pwd -P)
repo_root="$script_repo"

say() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }

resume=1
runner="systemd"
passthrough=()
dry_run=0
host_probe=0

while (($#)); do
  case "$1" in
    --resume) resume=1; shift ;;
    --fresh) resume=0; shift ;;
    --dry-run) dry_run=1; passthrough+=("--dry-run"); shift ;;
    --host-probe) host_probe=1; passthrough+=("--host-probe" "$2"); shift 2 ;;
    --runner) runner="$2"; passthrough+=("--runner" "$2"); shift 2 ;;
    --repo-root) repo_root="$2"; passthrough+=("--repo-root" "$2"); shift 2 ;;
    --court) passthrough+=("--court" "$2"); shift 2 ;;
    --state) passthrough+=("--state" "$2"); shift 2 ;;
    --stage) passthrough+=("--stage" "$2"); shift 2 ;;
    --force) passthrough+=("--force"); shift ;;
    -h | --help)
      sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      printf 'post-hstar: unknown argument %q\n' "$1" >&2
      exit 64
      ;;
  esac
done

case "$runner" in
  systemd | direct) ;;
  *) printf 'post-hstar: --runner must be systemd or direct\n' >&2; exit 64 ;;
esac

say "post-hstar: repo=$repo_root runner=$runner resume=$resume"

case "$runner" in
  systemd)
    command -v systemd-run >/dev/null 2>&1 ||
      { say "systemd-run is unavailable; refusing to launch unbounded. Pass --runner direct only for a host without a user systemd, and record why."; exit 78; }
    ;;
esac

if ((dry_run == 0 && host_probe == 0)); then
  command -v docker >/dev/null 2>&1 ||
    { say "docker is unavailable; the court cannot run (no host fallback exists)"; exit 69; }
  docker info >/dev/null 2>&1 ||
    { say "the docker daemon is unreachable; refusing to run anything on the host"; exit 69; }
fi

argv=(python3 "$script_repo/scripts/post_hstar.py")
if ((resume)); then
  argv+=(--resume)
fi
if ((${#passthrough[@]} == 0)) && [[ "$repo_root" != "$script_repo" ]]; then
  argv+=(--repo-root "$repo_root")
fi
argv+=("${passthrough[@]}")

say "post-hstar: ${argv[*]}"
exec "${argv[@]}"
