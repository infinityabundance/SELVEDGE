#!/usr/bin/env bash
# S66: select a seed's stage-2a checkpoint, interpolate at the fixed alpha, and
# score it once on full GSM8K, MATH-500 and full MMLU.
#
# The alpha is fixed at 0.375. No coefficient is re-selected on any test split.
# The run is one GPU job at a time and is meant to be launched in the eval court.
#
# Usage:
#   SELVEDGE_MEM=12g scripts/court --gpu eval -- \
#     scripts/s66_seed_pipeline.sh --seed 1
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
cd "$repo_root"

seed=""
run_dir=""
out_dir=""
student_dir="/hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17"
expected_sha=""
while (($#)); do
  case "$1" in
    --seed) seed="${2-}"; shift 2 ;;
    --run-dir) run_dir="${2-}"; shift 2 ;;
    --out-dir) out_dir="${2-}"; shift 2 ;;
    --expected-candidate-sha256) expected_sha="${2-}"; shift 2 ;;
    --student-dir) student_dir="${2-}"; shift 2 ;;
    -h | --help) sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) printf 's66-seed: unknown argument %s\n' "$1" >&2; exit 64 ;;
  esac
done
[[ "$seed" =~ ^[0-9]+$ ]] || { printf 's66-seed: --seed required\n' >&2; exit 64; }
[[ -n "$run_dir" ]] || run_dir="results/stage2/runs/s66-stage2a-seed${seed}"
[[ -n "$out_dir" ]] || out_dir="results/stage2/s66-seeds/seed${seed}"
mkdir -p "$out_dir"

holdout="results/artifacts/stage2/s58-combined/stage2a-combined-holdout.jsonl"
heldout_out="$out_dir/heldout-ce.json"
selection_out="$out_dir/selection.json"
interp_dir="$out_dir/interp"
interp_checkpoint="$interp_dir/interp-0.375-arm.pt"
stage1_out="$out_dir/stage1-gsm8k-math500.json"
paired_out="$out_dir/paired-math-vs-post.json"
mmlu_out="$out_dir/mmlu.json"
retention_out="$out_dir/paired-retention-vs-post.json"

python3 scripts/s55_heldout_ce.py \
  --student-dir "$student_dir" \
  --holdout "$holdout" \
  --run-dir "$run_dir" \
  --seq-len 1250 \
  --vram-limit-gib 12.0 \
  --out "$heldout_out"

python3 - "$heldout_out" "$selection_out" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
selected = payload["selected"]
out = {
    "source": sys.argv[1],
    "checkpoint": selected["checkpoint"],
    "step": selected["step"],
    "ce": selected["ce"],
    "base_ce": selected["base_ce"],
    "delta": selected["delta"],
    "rule": selected["rule"],
}
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    json.dump(out, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(f"s66-seed: selected {out['checkpoint']} step={out['step']} ce={out['ce']:.6f}")
PY
candidate=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$selection_out")
[[ -f "$candidate" ]] || { printf 's66-seed: selected checkpoint %s is absent\n' "$candidate" >&2; exit 66; }

python3 scripts/s66_rebuild_interp.py \
  --model-dir "$student_dir" \
  --checkpoint "$candidate" \
  --coefficient 0.375 \
  --out "$interp_checkpoint"

python3 scripts/s54_post_trained_eval.py \
  --model-dir "$student_dir" \
  --model-id Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --full-checkpoint "$interp_checkpoint" \
  --tasks gsm8k,math500 \
  --modes non-thinking \
  --decodings greedy \
  --batch-size 32 \
  --vram-limit-gib 13.0 \
  --out "$stage1_out"

python3 scripts/s57_paired.py \
  --candidate "$stage1_out" \
  --baseline results/stage1/s54-stage1-post-0.8b-nonthinking-greedy.json \
  --out "$paired_out"

python3 scripts/s43_score_total_cap.py \
  --arm interpolated \
  --full-checkpoint "$interp_checkpoint" \
  --student-dir "$student_dir" \
  --model-id Qwen/Qwen3.5-0.8B \
  --model-revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --tasks mmlu \
  --total-cap-map mmlu=14042 \
  --registration docs/ws19-anchored-subset-amendment-preregistration-v2.json \
  --vram-limit-gib 13.0 \
  --out "$mmlu_out"

raw_mmlu=$(python3 - "$mmlu_out" <<'PY'
import glob, json, os, sys
receipt = json.load(open(sys.argv[1], encoding="utf-8"))
raw = receipt.get("raw") or receipt.get("raw_paths") or {}
if isinstance(raw, dict):
    for value in raw.values():
        if isinstance(value, str) and value.endswith("mmlu.json"):
            print(value)
            raise SystemExit(0)
matches = sorted(glob.glob(os.path.join(os.path.dirname(sys.argv[1]), "raw", "*-mmlu.json")))
if not matches:
    raise SystemExit("no raw mmlu dump found beside the retention receipt")
print(matches[-1])
PY
)
[[ -f "$raw_mmlu" ]] || { printf 's66-seed: raw MMLU dump is absent (%s)\n' "$raw_mmlu" >&2; exit 66; }

python3 scripts/s57_paired.py \
  --lm-eval-candidate "$raw_mmlu" \
  --lm-eval-baseline results/stage2/eval/s58-post-0.8b-baseline-mmlu/raw/20261118T222909Z-mmlu.json \
  --lm-eval-metrics mmlu \
  --out "$retention_out"

python3 - "$out_dir" "$seed" "$candidate" "$interp_checkpoint" "$stage1_out" \
  "$paired_out" "$mmlu_out" "$retention_out" <<'PY'
import hashlib, json, os, sys
out_dir, seed, candidate, interp, stage1, paired, mmlu, retention = sys.argv[1:9]

def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()

payload = {
    "schema": "selvedge/s66-seed-result/v1",
    "seed": int(seed),
    "coefficient": 0.375,
    "selection_used_test_split": False,
    "candidate_checkpoint": {"path": candidate, "sha256": sha(candidate)},
    "interpolated_checkpoint": {"path": interp, "sha256": sha(interp)},
    "receipts": {
        "stage1": {"path": stage1, "sha256": sha(stage1)},
        "paired_math": {"path": paired, "sha256": sha(paired)},
        "mmlu": {"path": mmlu, "sha256": sha(mmlu)},
        "paired_retention": {"path": retention, "sha256": sha(retention)},
    },
}
with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(f"s66-seed: result -> {os.path.join(out_dir, 'result.json')}")
PY

