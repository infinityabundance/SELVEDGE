# S65 checkpoint release — alpha 0.375 (immutable identity)

This is a local release record, not an upload. No artifact was sent to a
registry, model hub or object store. The weights are about 1.0 GB (the selected
full-arm state) and are **not committed** to git.

## Released artifact

| field | value |
|---|---|
| local path | `results/stage2/s65-interp-selected/ckpt-t0.375-full.pt` |
| bytes | `996,328,814` |
| sha256 | `10ba2357beef052a434735c682009726b380866e450661de719eb437a5558e0a` |
| schema | `selvedge/stage2-ckpt/v1` |
| arm | `full` |
| step | `4000` |
| coefficient | `0.375` |
| state elements | `498,113,344` |
| unique active text path | `752,393,024` |
| identity receipt | `results/stage2/s65-interp-selected/identity.json` (stamp `20261119T212853Z`) |

## Source weights

| endpoint | source | identity |
|---|---|---|
| base, `t = 0` | `Qwen/Qwen3.5-0.8B` snapshot `/hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17` | revision `2fc06364715b967f1860aea9cf38778875588b17`; first-shard weight sha256 `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` |
| candidate, `t = 1` | `results/stage2/runs/s58-stage2a/ckpt-004000.pt` | sha256 `33ee31ae5586799b2544ce4b888ae09c4f2654ab11b03301d05720ddb0bb7fc6`; bytes `2,498,786,281`; step 4000; 10,000,000 processed tokens |

The tied embedding/output projection is frozen and identical at both endpoints.
The released state contains the 319 trainable text-tower tensors only.

## Exact rebuild command

The rebuild runs on CPU in the oracle court. It imports the S60 interpolation
routine rather than copying it, so the arithmetic is the same float32 mix:

```sh
SELVEDGE_MEM=10g scripts/court oracle -- \
  python3 scripts/s66_rebuild_interp.py \
    --model-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17 \
    --checkpoint results/stage2/runs/s58-stage2a/ckpt-004000.pt \
    --coefficient 0.375 \
    --out results/stage2/s65-interp-selected/rebuild/interp-0.375-arm.pt \
    --expect-sha256 10ba2357beef052a434735c682009726b380866e450661de719eb437a5558e0a
```

**The output basename matters.** PyTorch's zip serializer uses the output
filename as the root member name inside the `.pt` archive. Rebuilding into
`rebuilt-t0.375-full.pt` produced the same 319 tensors and the same payload but a
different archive hash (`8005207…`). Rebuilding into `interp-0.375-arm.pt`
reproduced the released artifact byte for byte:

```
s66-rebuild: .../rebuild/interp-0.375-arm.pt bytes=996328814
             sha256=10ba2357beef052a434735c682009726b380866e450661de719eb437a5558e0a
s66-rebuild: match=True
```

Rebuild receipt: `results/stage2/s65-interp-selected/rebuild/interp-0.375-arm.pt.json`
(stamp of the run’s Python-court receipt: `20261119T215931Z`). The duplicate
artifact itself was deleted after the hash check to conserve disk; the receipt
and the released copy carry the same bytes.

## Loading contract

1. Load `Qwen/Qwen3.5-0.8B` at revision
   `2fc06364715b967f1860aea9cf38778875588b17`.
2. `torch.load(path, map_location="cpu", weights_only=False)`.
3. `frontier.full.load_full_state_into(model, payload["arm_state"])`.

This is the exact `--full-checkpoint` path used by
`scripts/s54_post_trained_eval.py` and by the `interpolated` arm of
`scripts/s43_score_total_cap.py`.

## Integrity status

* The selected artifact's sha256 matches the grid receipt
  (`results/stage2/s60-interpolation/receipt.json`) and the S65 identity receipt.
* A clean rebuild from the two sources plus `0.375` matches the released bytes.
* No lossy fallback, quantisation or dependency substitution is used.
* No upload has occurred. Publishing requires credentials the operator has not
  provided.
