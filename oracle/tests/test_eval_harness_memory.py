"""The evaluation instrument must be memory-correct without changing the metric.

The registered distributional metric is KL(teacher || student) against the full
vocabulary at every position. The instrument now computes each model's hidden
states once and streams the vocabulary head over position chunks. These tests
pin the two properties that matter:

* **parity** — the chunked path reproduces the full-logits reference (which is
  the definition the registered metric was written with) for every statistic,
  at several chunk sizes and on every position, including the last one;
* **boundedness** — the head is never applied to more than `chunk` positions at
  a time, so no `positions x vocab` f32 logits tensor can exist.

They also pin the property that made the naive fix wrong: the model's own
forward pass must still run, because the T2 memory arms set their token context
from a forward pre-hook installed on the model. Replacing the whole forward with
a call to `model.model(...)` would leave that context empty.

Named tests, so a receipt can cite them.
"""

from __future__ import annotations

import os
import sys
import types

import pytest
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "eval"))

import harness  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
VOCAB = 97
TOKENS = [3, 11, 5, 90, 7, 42, 1, 63, 22, 8, 17, 95, 4, 33, 60, 12, 19, 77, 2, 25,
          44, 6, 81, 13]


def _tiny_qwen3(vocab: int = VOCAB, seed: int = 0):
    """A real `Qwen3ForCausalLM` at toy size: the shipped head path, not a mock."""
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3Config(
        vocab_size=vocab, hidden_size=32, intermediate_size=37,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, max_position_embeddings=256, tie_word_embeddings=False)
    torch.manual_seed(seed)
    model = transformers.Qwen3ForCausalLM(config)
    model.eval()
    return model


def _parity_tolerance(device):
    """Parity tolerance, from the head's own dtype.

    On CUDA the head runs in bf16 under autocast (the registered instrument's
    `precision`), so a change of head batch shape can move a logit by one bf16
    ULP; on CPU the head is f32 and the only difference is BLAS blocking, which
    was measured at 2.3e-6 relative in KL. Nothing larger is allowed: an
    off-by-one position, a wrong vocabulary slice or a changed reduction shows
    up as a difference of order 1, not 1e-5.
    """
    return (1e-6, 1e-9) if device == "cuda" else (1e-5, 1e-8)


def _assert_parity(got, want, device, what):
    rel, abs_tol = _parity_tolerance(device)
    assert got == pytest.approx(want, rel=rel, abs=abs_tol), what


def _brute_force(student, teacher, tokens, device):
    """The registered definition, computed the way the old instrument did."""
    ids = torch.tensor([tokens], dtype=torch.long, device=device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                enabled=ids.is_cuda):
        student_logits = student(input_ids=ids, use_cache=False).logits.float()
        teacher_logits = teacher(input_ids=ids, use_cache=False).logits.float()
    s = student_logits[:, :-1, :]
    t = teacher_logits[:, :-1, :]
    target = ids[:, 1:]
    logp_s = F.log_softmax(s, dim=-1)
    logp_t = F.log_softmax(t, dim=-1)
    p_t = logp_t.exp()
    kl = (p_t * (logp_t - logp_s)).sum(dim=-1)
    nll = -logp_s.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    top1 = (s.argmax(dim=-1) == target).float()
    return {
        "kl": kl.mean().item(),
        "ppl": float(torch.exp(nll.mean()).item()),
        "nll": nll.mean().item(),
        "top1": top1.mean().item(),
        "positions": int(target.numel()),
        "kl_positions": kl.reshape(-1).tolist(),
        "nll_positions": nll.reshape(-1).tolist(),
    }


# ---------------------------------------------------------------------------
# parity with the full-logits definition


@pytest.mark.parametrize("device", DEVICES)
def test_chunked_window_stats_match_the_full_logits_reference(device):
    student = _tiny_qwen3(seed=11).to(device)
    teacher = _tiny_qwen3(seed=23).to(device)
    reference = _brute_force(student, teacher, TOKENS, device)
    chunked = harness.window_stats(student, teacher, TOKENS, device, chunk=7)
    assert chunked["positions"] == reference["positions"] == len(TOKENS) - 1
    _assert_parity(chunked["kl"], reference["kl"], device, "kl")
    _assert_parity(chunked["nll"], reference["nll"], device, "nll")
    _assert_parity(chunked["ppl"], reference["ppl"], device, "ppl")
    assert chunked["top1"] == pytest.approx(reference["top1"], rel=0.0, abs=1e-9)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("chunk", [1, 2, 3, 7, 64, 1000])
def test_the_metric_is_independent_of_the_chunk_size(device, chunk):
    student = _tiny_qwen3(seed=5).to(device)
    teacher = _tiny_qwen3(seed=9).to(device)
    reference = harness.window_stats(student, teacher, TOKENS, device, chunk=len(TOKENS))
    got = harness.window_stats(student, teacher, TOKENS, device, chunk=chunk)
    for key in ("kl", "nll", "ppl", "top1"):
        _assert_parity(got[key], reference[key], device, key)
    assert got["positions"] == reference["positions"]


@pytest.mark.parametrize("device", DEVICES)
def test_a_two_token_window_scores_exactly_one_position(device):
    student = _tiny_qwen3(seed=13).to(device)
    teacher = _tiny_qwen3(seed=17).to(device)
    stats = harness.window_stats(student, teacher, [5, 9], device, chunk=64)
    assert stats["positions"] == 1
    reference = _brute_force(student, teacher, [5, 9], device)
    _assert_parity(stats["kl"], reference["kl"], device, "kl")


@pytest.mark.parametrize("device", DEVICES)
def test_every_position_is_scored_and_none_is_dropped(device):
    """The chunk boundary must not lose or duplicate a position.

    The last position is the one a naive `range(0, n, chunk)` loop can drop;
    it is checked by comparing the chunked sums against the per-position
    reference values, not only the means.
    """
    student = _tiny_qwen3(seed=31).to(device)
    teacher = _tiny_qwen3(seed=37).to(device)
    reference = _brute_force(student, teacher, TOKENS, device)
    for chunk in (1, 4, 5):
        stats = harness.window_stats(student, teacher, TOKENS, device, chunk=chunk)
        assert stats["positions"] == len(reference["kl_positions"])
        summed = stats["kl"] * stats["positions"]
        _assert_parity(summed, sum(reference["kl_positions"]), device, "kl sum")


# ---------------------------------------------------------------------------
# the plumbing itself is exact


@pytest.mark.parametrize("device", DEVICES)
def test_the_captured_hidden_states_are_the_model_forwards(device):
    """The head stub must capture the same tensor the full forward would use."""
    model = _tiny_qwen3(seed=71).to(device)
    ids = torch.tensor([TOKENS], dtype=torch.long, device=device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                enabled=ids.is_cuda):
        expected = model.model(input_ids=ids, use_cache=False).last_hidden_state
        captured = harness.final_hidden_states(model, ids)
    assert torch.equal(captured, expected)


@pytest.mark.parametrize("device", DEVICES)
def test_a_full_width_streamed_head_is_bitwise_the_model_logits(device):
    """With one chunk over the whole window the streamed path is a no-op."""
    model = _tiny_qwen3(seed=73).to(device)
    ids = torch.tensor([TOKENS], dtype=torch.long, device=device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                enabled=ids.is_cuda):
        expected = model(input_ids=ids, use_cache=False).logits.float()
        hidden = harness.final_hidden_states(model, ids)
        got = harness.logits_chunk(model, hidden, 0, len(TOKENS))
    assert torch.equal(got, expected)


@pytest.mark.parametrize("device", DEVICES)
def test_chunked_logits_are_within_one_step_of_the_full_forward(device):
    """A chunked head may re-order f32/bf16 accumulation by at most one ULP."""
    model = _tiny_qwen3(seed=79).to(device)
    ids = torch.tensor([TOKENS], dtype=torch.long, device=device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                enabled=ids.is_cuda):
        full = model(input_ids=ids, use_cache=False).logits.float()
        hidden = harness.final_hidden_states(model, ids)
        parts = [harness.logits_chunk(model, hidden, start, min(start + 5, len(TOKENS)))
                 for start in range(0, len(TOKENS), 5)]
    got = torch.cat(parts, dim=1)
    step = 2.0 ** -8 if device == "cuda" else 2.0 ** -24
    assert (got - full).abs().max() <= step * (1.0 + full.abs().max())


# ---------------------------------------------------------------------------
# boundedness: the head never sees a whole window


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("chunk", [1, 5, 8])
def test_no_head_call_exceeds_the_chunk(device, chunk):
    student = _tiny_qwen3(seed=41).to(device)
    teacher = _tiny_qwen3(seed=43).to(device)
    widths: dict[str, list[int]] = {"student": [], "teacher": []}

    def spy(which, model):
        head = model.lm_head
        original = head.forward

        def forward(hidden_states, *args, **kwargs):
            widths[which].append(int(hidden_states.shape[1]))
            return original(hidden_states, *args, **kwargs)

        head.forward = forward
        return lambda: setattr(head, "forward", original)

    restore = [spy("student", student), spy("teacher", teacher)]
    try:
        harness.window_stats(student, teacher, TOKENS, device, chunk=chunk)
    finally:
        for undo in restore:
            undo()
    for which, calls in widths.items():
        assert calls, f"{which} head was never applied"
        assert max(calls) <= chunk, f"{which} head saw {max(calls)} positions"
        assert len(calls) == -(-(len(TOKENS) - 1) // chunk)


@pytest.mark.parametrize("device", DEVICES)
def test_the_model_forward_still_runs_so_the_arm_hooks_fire(device):
    student = _tiny_qwen3(seed=47).to(device)
    teacher = _tiny_qwen3(seed=53).to(device)
    seen: list[list[int]] = []

    def hook(_module, args, kwargs):
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        if ids is not None:
            seen.append([int(item) for item in ids.reshape(-1).tolist()])

    handle = student.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        harness.window_stats(student, teacher, TOKENS, device, chunk=6)
    finally:
        handle.remove()
    assert seen == [TOKENS], "the arm's forward pre-hook (token context) did not fire"


def test_a_model_without_an_output_head_fails_closed():
    class NoHead(torch.nn.Module):
        def forward(self, input_ids=None, **kwargs):
            raise AssertionError("must not be called")

    with pytest.raises(RuntimeError, match="output head"):
        harness.window_stats(NoHead(), NoHead(), TOKENS, "cpu")


# ---------------------------------------------------------------------------
# MC1 scoring uses the same streamed path


class _StubTokenizer:
    """Deterministic id assignment; the metric only needs ids, not text."""

    def __init__(self, prompt_ids, choice_ids):
        self.prompt_ids = prompt_ids
        self.choice_ids = choice_ids

    def __call__(self, text, return_tensors="pt", add_special_tokens=True):
        ids = self.prompt_ids if add_special_tokens else self.choice_ids
        return types.SimpleNamespace(
            input_ids=torch.tensor([ids], dtype=torch.long))


def _brute_force_loglikelihood(model, ids, start, device):
    ids = ids.to(device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                enabled=ids.is_cuda):
        logits = model(input_ids=ids, use_cache=False).logits.float()
    log_probs = F.log_softmax(logits[:, start - 1:-1, :], dim=-1)
    return float(log_probs.gather(
        -1, ids[:, start:].unsqueeze(-1)).squeeze(-1).mean().item())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("chunk", [1, 2, 32])
def test_answer_loglikelihood_matches_the_full_logits_reference(device, chunk):
    model = _tiny_qwen3(seed=61).to(device)
    tokenizer = _StubTokenizer([2, 7, 11, 19], [23, 41, 73])
    ids = torch.tensor([[2, 7, 11, 19, 23, 41, 73]], dtype=torch.long)
    reference = _brute_force_loglikelihood(model, ids, start=4, device=device)
    got = harness.answer_loglikelihood(model, tokenizer, "prompt", "choice", device,
                                       chunk=chunk)
    _assert_parity(got, reference, device, "mc1 log-likelihood")
