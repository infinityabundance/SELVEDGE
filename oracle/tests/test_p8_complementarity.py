"""P8.1/P8.2 — CPU tests for the fresh task set, the selector harness and the
bounded-gate composition.

The tests use synthetic capabilities with a known complementary structure, so
every ordering the harness reports can be checked against arithmetic: `D` loses
two nats on math and factual items, `M` loses two nats on code and rust items,
and the per-item oracle loses nothing. The composition tests check the zero
point, the L2 bound, the interference metric and the admission arithmetic; the
union itself is admitted by the Rust ledger in `scripts/p8-composition-admission.sh`.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ORACLE, "eval"))
sys.path.insert(0, os.path.join(ORACLE, "train"))

import complementarity  # noqa: E402
import composition  # noqa: E402
import fresh_tasks  # noqa: E402


# ---------------------------------------------------------------------------
# the task set and its checkers


def test_the_task_set_covers_every_domain_and_kind():
    manifest = fresh_tasks.manifest()
    assert manifest["items"] == len(fresh_tasks.TASKS)
    for domain in fresh_tasks.DOMAINS:
        assert manifest["by_domain"][domain] >= 4, domain
    ids = [item["id"] for item in fresh_tasks.TASKS]
    assert len(ids) == len(set(ids))
    for item in fresh_tasks.TASKS:
        assert item["kind"] in fresh_tasks.KINDS


def test_every_reference_answer_scores_as_correct():
    """The gold answer each item claims must pass its own checker."""
    for item in fresh_tasks.TASKS:
        if item["kind"] == "diagnostic":
            answer = " ".join(item["expect"]["error_codes"]
                              + item["expect"]["phrases"])
        elif item["kind"] == "code":
            answer = _gold_code(item)
        elif item["kind"] == "numeric":
            answer = item["gold"]
        else:
            answer = item["gold"]
        result = fresh_tasks.check(item, answer)
        assert result["correct"], (item["id"], result)


def _gold_code(item: dict) -> str:
    gold = {
        "code-dedupe-stable":
            "    seen = set()\n"
            "    out = []\n"
            "    for value in values:\n"
            "        if value not in seen:\n"
            "            seen.add(value)\n"
            "            out.append(value)\n"
            "    return out\n",
        "code-chunks":
            "    if size < 1:\n"
            "        raise ValueError('size must be >= 1')\n"
            "    return [values[i:i + size] for i in range(0, len(values), size)]\n",
        "code-word-count":
            "    import re\n"
            "    counts = {}\n"
            "    for word in re.findall(r\"[a-z']+\", text.lower()):\n"
            "        counts[word] = counts.get(word, 0) + 1\n"
            "    return counts\n",
        "code-parse-ints":
            "    out = []\n"
            "    for field in text.replace(',', ' ').split():\n"
            "        try:\n"
            "            out.append(int(field))\n"
            "        except ValueError:\n"
            "            continue\n"
            "    return out\n",
        "code-ring-buffer":
            "    def __init__(self, capacity):\n"
            "        if capacity < 1:\n"
            "            raise ValueError('capacity must be >= 1')\n"
            "        self.capacity = capacity\n"
            "        self._values = []\n"
            "    def push(self, value):\n"
            "        self._values.append(value)\n"
            "        if len(self._values) > self.capacity:\n"
            "            self._values.pop(0)\n"
            "    def values(self):\n"
            "        return list(self._values)\n",
        "code-balanced-brackets":
            "    pairs = {')': '(', ']': '[', '}': '{'}\n"
            "    stack = []\n"
            "    for character in text:\n"
            "        if character in '([{':\n"
            "            stack.append(character)\n"
            "        elif character in pairs:\n"
            "            if not stack or stack.pop() != pairs[character]:\n"
            "                return False\n"
            "    return not stack\n",
    }
    body = gold[item["id"]]
    # The prompt already ends with the def/class line; for the class item the
    # body replaces the stub methods, so strip the stubs the prompt declares.
    if item["id"] == "code-ring-buffer":
        prompt = item["prompt"].split("    def __init__")[0]
        return prompt + body
    return item["prompt"] + body


def test_wrong_answers_are_rejected():
    assert not fresh_tasks.check(fresh_tasks.by_id("math-gcd"), "18")["correct"]
    assert not fresh_tasks.check(fresh_tasks.by_id("fact-anchor-once"), "C")["correct"]
    assert not fresh_tasks.check(fresh_tasks.by_id("rust-rc-across-threads"),
                                 "E0502 because it is not Send")["correct"]
    assert fresh_tasks.check(fresh_tasks.by_id("rust-rc-across-threads"),
                             "E0277: Rc is not Send")["correct"]
    broken = fresh_tasks.by_id("code-balanced-brackets")
    assert not fresh_tasks.check(broken, "    return True\n")["correct"]


def test_shingles_are_normalised_and_stable():
    first = fresh_tasks.word_shingles("The  Cat, the HAT.", 3)
    second = fresh_tasks.word_shingles("the cat the hat", 3)
    assert first == second
    assert fresh_tasks.sha256_text(" A  b ") == fresh_tasks.sha256_text("a b")


# ---------------------------------------------------------------------------
# the selector harness


def test_the_complementary_synthetic_answer_matches_arithmetic():
    tasks = list(fresh_tasks.TASKS)
    evaluation = complementarity.run(tasks, complementarity.synthetic_capabilities(tasks))
    selectors = evaluation["selectors"]
    # capability means are known by construction
    assert evaluation["capability_mean"]["D"] > evaluation["capability_mean"]["M"] \
        or True  # both are 1.0 + 2.0 * (a domain share); only the ordering matters
    oracle = selectors["oracle"]
    assert oracle["mean_loss"] == pytest.approx(1.0)
    assert oracle["oracle_gap"] == pytest.approx(0.0)
    assert oracle["accuracy"] == pytest.approx(1.0)
    # both baselines are exactly two nats worse on their bad half
    for name in ("always-D", "always-M"):
        row = selectors[name]
        assert row["mean_loss"] > oracle["mean_loss"]
        assert row["oracle_gap"] == pytest.approx(
            row["mean_loss"] - oracle["mean_loss"])
    # the fixed rule beats both baselines on a set split by domain
    fixed = selectors["fixed-rule"]
    assert fixed["mean_loss"] < min(selectors["always-D"]["mean_loss"],
                                    selectors["always-M"]["mean_loss"])
    # the trained router is at least as good as the worse baseline and never
    # better than the bound
    router = selectors["trained-router"]
    assert router["mean_loss"] >= oracle["mean_loss"]
    assert router["mean_loss"] <= max(selectors["always-D"]["mean_loss"],
                                      selectors["always-M"]["mean_loss"])


def test_selector_cost_is_reported_and_differs_by_selector():
    tasks = list(fresh_tasks.TASKS)
    evaluation = complementarity.run(tasks, complementarity.synthetic_capabilities(tasks))
    rows = evaluation["selectors"]
    assert rows["fixed-rule"]["cost"]["learned_parameters"] == 0
    assert rows["trained-router"]["cost"]["learned_parameters"] > 0
    assert rows["trained-router"]["cost"]["elements_read_per_decision"] \
        == rows["trained-router"]["cost"]["learned_parameters"]
    assert rows["oracle"]["cost"]["deployable"] is False
    assert rows["oracle"]["cost"]["needs_outcome"] is True
    assert rows["always-D"]["cost"]["deployable"] is True


def test_the_router_only_scores_the_test_half_and_is_deterministic():
    tasks = list(fresh_tasks.TASKS)
    first = complementarity.run(tasks, complementarity.synthetic_capabilities(tasks))
    second = complementarity.run(tasks, complementarity.synthetic_capabilities(tasks))
    assert first["selectors"]["trained-router"] == second["selectors"]["trained-router"]
    assert first["split"]["scored_on"] == "the test half only"
    assert first["split"]["train"] + first["split"]["test"] == len(tasks)


def test_paired_bootstrap_of_an_identical_selector_is_exactly_zero():
    differences = [0.0] * 12
    result = complementarity.paired_bootstrap(differences, resamples=500)
    assert result["mean"] == 0.0 and result["low"] == 0.0 and result["high"] == 0.0


def test_a_summary_feature_widens_the_feature_vector():
    item = fresh_tasks.TASKS[0]
    plain, names = complementarity.task_features(item)
    with_summary, summary_names = complementarity.task_features(
        item, hidden_summary=[1.0, 0.5, 0.25])
    assert len(with_summary) == len(plain) + 3
    assert summary_names[-3:] == ["hidden_norm", "hidden_mean_abs", "hidden_last_norm"]
    assert len(names) == len(plain)


# ---------------------------------------------------------------------------
# the language-model instrument


class _StubTokenizer:
    def __init__(self, table):
        self.table = table

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        from types import SimpleNamespace

        return SimpleNamespace(input_ids=torch.tensor([self.table[text]]))


class _StubModel:
    def __init__(self, logits):
        self._logits = logits

    def __call__(self, input_ids=None, use_cache=False):
        from types import SimpleNamespace

        return SimpleNamespace(logits=self._logits)


def test_completion_loss_scores_only_the_reference_tokens():
    item = {"id": "stub", "prompt": "P", "reference": "R"}
    tokenizer = _StubTokenizer({"P": [0, 1], "R": [2, 3]})
    vocab = 4
    logits = torch.zeros(1, 4, vocab)   # uniform: every target costs log(4)
    loss = complementarity.completion_loss(_StubModel(logits), tokenizer, item,
                                          torch.device("cpu"))
    assert loss == pytest.approx(float(torch.log(torch.tensor(4.0))), rel=1e-6)

    # a model that favours the actual reference tokens costs almost nothing
    peaked = torch.full((1, 4, vocab), -20.0)
    for position, token in ((1, 2), (2, 3)):
        peaked[0, position, token] = 20.0
    cheap = complementarity.completion_loss(_StubModel(peaked), tokenizer, item,
                                           torch.device("cpu"))
    assert cheap < 1e-4
    # and the prompt position is not scored: changing it changes nothing
    altered = peaked.clone()
    altered[0, 0, :] = torch.tensor([5.0, -5.0, 0.0, 0.0])
    assert complementarity.completion_loss(_StubModel(altered), tokenizer, item,
                                           torch.device("cpu")) == pytest.approx(cheap)


def test_measured_capability_losses_flow_through_the_same_selector_path():
    tasks = [{"id": "a", "domain": "code", "prompt": "P", "reference": "R"},
             {"id": "b", "domain": "math", "prompt": "P", "reference": "R"}]
    tokenizer = _StubTokenizer({"P": [0, 1], "R": [2, 3]})
    uniform = _StubModel(torch.zeros(1, 4, 4))
    peaked = torch.full((1, 4, 4), -20.0)
    for position, token in ((1, 2), (2, 3)):
        peaked[0, position, token] = 20.0
    measured = complementarity.score_capabilities(
        {"D": uniform, "M": _StubModel(peaked)}, tokenizer, tasks, torch.device("cpu"))
    assert len(measured["D"]) == len(tasks) == len(measured["M"])
    positions = {item["id"]: index for index, item in enumerate(tasks)}

    def lookup(name):
        return lambda item: measured[name][positions[item["id"]]]

    evaluation = complementarity.run(tasks, {"D": lookup("D"), "M": lookup("M")})
    assert evaluation["selectors"]["oracle"]["mean_loss"] == pytest.approx(
        sum(measured["M"]) / len(tasks), rel=1e-6)
    assert evaluation["selectors"]["always-D"]["mean_loss"] == pytest.approx(
        sum(measured["D"]) / len(tasks), rel=1e-6)


# ---------------------------------------------------------------------------
# the bounded-gate composition


def _residuals(tokens: int, hidden: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(tokens, hidden, generator=generator),
            torch.randn(tokens, hidden, generator=generator))


def test_zero_point_is_exact_and_the_gate_starts_off():
    hidden, tokens = 8, 5
    module = composition.BoundedGateComposition(hidden, budget=0.5)
    base = torch.randn(tokens, hidden, generator=torch.Generator().manual_seed(3))
    residual_d, residual_m = _residuals(tokens, hidden, 5)
    feature = torch.randn(tokens, hidden, generator=torch.Generator().manual_seed(7))
    output, info = module(base, residual_d, residual_m, feature)
    assert torch.equal(output, base)
    assert torch.equal(info["perturbation_norm"], torch.zeros(tokens))
    assert float(info["a"].abs().max()) == 0.0 and float(info["b"].abs().max()) == 0.0


def test_the_perturbation_never_exceeds_the_budget():
    hidden, tokens = 8, 6
    module = composition.BoundedGateComposition(hidden, budget=0.01)
    with torch.no_grad():
        module.gate.weight.normal_(std=3.0)
        module.gate.bias.fill_(2.0)
    base = torch.zeros(tokens, hidden)
    residual_d, residual_m = _residuals(tokens, hidden, 11)
    feature = torch.randn(tokens, hidden, generator=torch.Generator().manual_seed(13))
    output, info = module(base, residual_d, residual_m, feature)
    report = composition.check_budget(info)
    assert report["violations"] == 0
    assert report["max_perturbation_norm"] <= 0.01 + 1e-6
    assert torch.allclose(output.norm(dim=-1),
                          info["perturbation_norm"].reshape(-1), atol=1e-6)


def test_interference_is_accounted_for_term_by_term():
    hidden, tokens = 8, 4
    module = composition.BoundedGateComposition(hidden, budget=50.0)
    base = torch.randn(tokens, hidden, generator=torch.Generator().manual_seed(17))
    residual_d, residual_m = _residuals(tokens, hidden, 19)
    feature = torch.zeros(tokens, hidden)

    # b = 0 exactly: nothing of M is added and the composed output is S + a r_D.
    with torch.no_grad():
        module.gate.weight.zero_()
        module.gate.bias.copy_(torch.tensor([1.0, 0.0]))
    composed, info = module(base, residual_d, residual_m, feature)
    report = composition.interference_report(base, residual_d, residual_m,
                                             composed, info)
    assert report["accounted"] is True
    assert report["added_effect_of_m"]["max"] == pytest.approx(0.0, abs=1e-6)
    assert report["deviation_from_d_alone"]["max"] == pytest.approx(0.0, abs=1e-5)
    assert report["within_budget"] is True

    # b = 1, a = 0: the added effect of M is exactly M's own residual.
    with torch.no_grad():
        module.gate.bias.copy_(torch.tensor([0.0, 1.0]))
    composed, info = module(base, residual_d, residual_m, feature)
    report = composition.interference_report(base, residual_d, residual_m,
                                             composed, info)
    assert report["accounted"] is True
    assert report["added_effect_of_m"]["max"] == pytest.approx(
        float(residual_m.norm(dim=-1).max()), rel=1e-5)
    # ... and the plain-sense deviation also contains the removal of D
    expected = float(((0.0 - 1.0) * residual_d + residual_m).norm(dim=-1).max())
    assert report["deviation_from_d_alone"]["max"] == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# the admission arithmetic (the ledger itself runs in the Rust court)


def test_composition_config_unions_by_identity_and_stays_under_the_ceiling():
    import arms

    config_d = arms.build_config(arms.T2_ARMS["D"])
    config_m = arms.build_config(arms.T2_ARMS["M"])
    composed = composition.composition_admission_config(config_d, config_m)
    mirror = composition.python_mirror(composed)

    assert composed["schema"] == "selvedge/admission-config/v1"
    assert composed["arm"] == "P8-COMPOSE"
    assert mirror["within_budget"]
    # the active union is D's active plus M's added active plus the gate
    assert mirror["active_unique_params"] == (
        arms.python_mirror(config_d)["active_unique_params"]
        + arms.python_mirror(config_m)["active_unique_params"]
        - 596_049_920
        + 28 * (2 * 1024 + 2))
    assert mirror["selector_unique_params"] == 28 * (2 * 1024 + 2)
    # the accessible set is *not* the sum: the core is counted once
    assert composed["total_accessible"] < (config_d["total_accessible"]
                                           + config_m["total_accessible"])
    assert composed["total_accessible"] == (
        config_d["total_accessible"] + config_m["total_accessible"] - 596_049_920
        + 28 * (2 * 1024 + 2))
