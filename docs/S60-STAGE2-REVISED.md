# S60 — the revised stage 2 (on-policy): registration summary

The machine-readable registration is
`docs/ws19-post-trained-route-stage2-revised-preregistration-s60-v1.json`. It was
committed before any screen, rollout, interpolation or score named in it.
Nothing on this page is a measurement; the queue is `docs/S60-GPU-QUEUE.md`.

## Why the recipe changes

S58 measured a stage-2a arm that learned short, confident answers: GSM8K
+2.88 points with an interval barely above zero, MATH-500 −11.0 points with the
interval entirely below zero, and MMLU outside the 1.0-point retention bound.
The corpus was teacher text, the objective was completion CE, and the gradient
weighted examples rather than tokens. The revised stage changes the data to the
student's own verified rollouts, the objective to reward-gated reverse KL, the
parameterisation to merged LoRA, and the LR selection to a measured KL gate.

## What the registration fixes

| clause | registered value |
| --- | --- |
| student | post-trained `Qwen/Qwen3.5-0.8B` `2fc06364…`, non-thinking, identity-checked before load |
| rollouts | cap 2,048, K=8, temperature 0.8, top-p 0.95, EOS-only stop |
| pass rate | `p` = verified-correct / 8 per problem; `p=0` and `p=1` carry zero weight and are counted |
| weight | `w(p) = p(1-p)` per problem; the unit of learning is the **difficulty-weighted step** |
| teacher | post-trained 2B `15852e8…`, a **separate pass** over the student's own tokens, one log-prob per token, no top-k cache, ~4 MB per million tokens, never co-resident |
| gate | a verified final answer (train label) admits a rollout to reverse-KL supervision |
| relay | loop-onset cut, 2B continues from the student's own prefix, only verified-correct continuations kept, reverse KL on relay tokens |
| adapter | LoRA rank 64 on **every linear layer**, alpha 32, merged at the end; unique active union stays 752,393,024 |
| LR | gate starts at 1e-4 (10× the full-FT 1e-5); screen {3e-5, 1e-4, 3e-4} by fixed-batch KL, not by accuracy |
| cadence | 0.5M, 1M, 2M, 5M processed tokens, with termination/truncation/loop and MMLU retention at each point |
| retention | MMLU full (14,042) and MMLU-Pro on a 4,000-item total-cap subset; an unresolved interval is labelled `unresolved`, not passed |
| win | > 54.21 GSM8K and > 38.20 MATH-500, each with a problem-level paired 95% interval above zero, and both retention bounds |

The win bar is unchanged from `docs/ws19-post-trained-route-registration-v1.json`.
The retention sample is not: the S58 note that 504 MMLU-Pro items cannot resolve
a 1.0-point bound is the reason the revised registration uses 4,000 items and an
explicit `unresolved` treatment.

## What the literature supports, and what it does not

`docs/S60-LITERATURE.md` records URL, date, claim and verdict per source.

- **Supported:** all-linear LoRA matches full fine-tuning in the low-regret
  regime, and the optimal LoRA LR is ~10× FullFT (LoRA Without Regret,
  VERIFIED); verifier-gated teacher supervision (RG-OPD, VERIFIED);
  `p(1-p)` problem weighting (PACED, VERIFIED); teacher continuation from the
  student's prefix (OLIVE, VERIFIED); failure profiling by repeated sampling
  (FPRT, VERIFIED).
- **Not supported as registered:** the **loop-onset** relay trigger and
  **reverse KL** on relay tokens. Relay-OPD triggers on a failed-prefix
  asymmetry, and OLIVE trains by cross-entropy on teacher tokens. The relay is
  a registered hypothesis with its own control, not a published result.
- **Contrary to a likely reading:** *Solving Without Stopping* reports that
  **non-thinking** students keep stopping; its stopping warning is a thinking-
  mode result. The termination cadence is still required, but the paper does
  not predict a non-thinking stopping regression.

## Controls

- **Gate off:** the same rollouts and recipe with every sampled rollout
  supervised, verified or not. The difference isolates the verified-answer gate.
- **Relay off:** the same rollouts and gate with no relay. The difference
  isolates the loop-onset relay.
- Both are attribution evidence with their own paired intervals; neither is a
  win condition.

## Current state

No screen, rollout, interpolation or score from this registration exists. The
first GPU job is the long-row smoke in `docs/S60-GPU-QUEUE.md` §1, because the
2500-token row width S60 adopts to preserve long MATH traces has not been
measured on this card.
