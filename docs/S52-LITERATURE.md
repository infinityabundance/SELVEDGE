# S52 literature — pre-read routing, counterfactual labels, cost-aware gating

Receipt: `results/stage2/s52-literature-20261117T230200Z.json` (stamp
`20261117T230200Z`). Accessed 2026-11-17. Each claim below names its URL and the
verification level; the receipt carries the full quotations.

Question: is there published evidence for (a) outcome-supervised/ pre-fetch
routing, (b) counterfactual utility labels from paired rollouts, (c) cost-aware
gating, and (d) that deciding to retrieve before the fetch preserves quality?

| id | claim | status |
| --- | --- | --- |
| Jeong et al., Adaptive-RAG, arXiv:2403.14403 (2024-03-21) | a small LM routes to no-retrieval / single-step / iterative retrieval before the strategy runs; labels are collected from actual model outcomes; efficiency and accuracy improve | VERIFIED |
| Asai et al., Self-RAG, arXiv:2310.11511 (2023-10-17) | retrieve on demand; indiscriminate retrieval can hurt; gains over RAG and ChatGPT baselines | VERIFIED |
| Jiang et al., FLARE, arXiv:2305.06983 (2023-05-11) | retrieve/regenerate when an upcoming sentence is low-confidence; superior or competitive on four long-form tasks | VERIFIED |
| Su et al., DRAGIN, arXiv:2403.10081 (2024-03-15) | decide when/what to retrieve during generation; static triggers are a weakness; superior on four datasets | VERIFIED |
| Zou et al., CoBRA, arXiv:2609.00967 (2026-09-01) | paired trajectories with and without tools estimate a reward margin that supervises the tool-use boundary; retrieval is the main tool | VERIFIED |
| He et al., RADEG, arXiv:2608.09168 (2026-08-10) | predicts execution utility before the expensive rollout from matched same-query rollouts; cuts unnecessary executions while preserving verifier reward | VERIFIED |
| Hashemi et al., cost-aware retrieval depth, arXiv:2510.15719 (2025-10-17) | a cost-aware advantage for retrieval depth; ~16–20% lower latency and ~+5% EM over seven QA datasets | VERIFIED |
| Mishra, CA-RAG, arXiv:2606.02581 (2026-03-26) | per-query routing over strategy bundles including retrieval-free; 26% fewer billed tokens, equivalent answer quality | VERIFIED |
| Zhao, when additional retrieval is worthwhile, arXiv:2610.37358 (2026-10-27) | a frozen pre-decision benefit estimate saves retrieval but paired bootstrap does not establish a utility advantage, and no clear answer-quality gain is found | VERIFIED |
| Ong et al., RouteLLM, arXiv:2406.18665 (2024-06-26) | preference-trained routers trade cost for quality between models, >2× cost reduction without quality loss | PARTIAL (model routing, not retrieval) |
| Dong et al., RAFT, arXiv:2304.06767 (2023-04-13) | sampled responses ranked by a reward model are usable supervision | PARTIAL (single-condition ranking, not a paired with/without difference) |

## Reading

**Pre-fetch routing is established; token-level pre-read routing is not.** All
four RAG routers decide before or at retrieval, and Adaptive-RAG's labels come
from actual model outcomes. None of them predicts utility from the causal hidden
state at every token and fetches only the selected memory rows.

**Paired counterfactual labels are published at instance level.** CoBRA's
"reward margin between answering with and without tools" and RADEG's matched
same-query rollouts are the closest precedents. Both label a trajectory or an
instance, not a token. The S52 label is the prompt-level paired outcome
difference broadcast to token rows; the receipt records this as PARTIAL and the
registration does not claim token-level utility identification.

**Cost-aware gating is published, and so is its failure mode.** CA-RAG and the
cost-aware depth paper show measured cost savings at preserved quality. The
2026 evidence-gain study is the counterweight: a routing benefit estimate can
save retrieval without preserving answer quality. The S52 win bar therefore
requires exact-answer accuracy to beat both the base and the matched gated-dense
control, plus fewer measured bytes — not read savings alone.

**No template uses a conv-free sparse read.** The S52 routed read is a
registered mechanism (8-row table gather, key gate and value projection, no
cross-token convolution) because a convolution over retrieved values cannot be
evaluated exactly from a sparse gather whose unselected predecessors were never
fetched. This is a mechanism choice, not a claim of equivalence to the
corrective memory arm.
