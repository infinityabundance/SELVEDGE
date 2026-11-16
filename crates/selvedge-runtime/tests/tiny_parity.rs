//! Tiny-fixture parity against the independent NumPy reference.
//!
//! The reference logits are committed (`fixtures/tiny-qwen3/reference-logits.json`)
//! and produced by `oracle/tiny_qwen3_reference.py`, which shares no code with the
//! Rust runtime. This court therefore checks the Rust forward against a second
//! implementation of the Qwen3 equations on the same weights, including two
//! greedy continuation steps so the key/value cache is exercised.
//!
//! Tolerance policy (pre-registered, see `docs/RUNTIME.md`):
//! `|reference - observed| <= abs + rel * |reference|` with `abs = rel = 1e-4`
//! for the tiny fixture. The observed maxima are asserted to be far below that,
//! so a regression shows up as a number, not as a threshold argument.

// Integration courts: a panicking assertion is the failure report.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

mod common;

use selvedge_runtime::{ExecutionMode, Runner, logits_digest};
use serde_json::Value;

/// Absolute tolerance for the tiny fixture.
const TINY_ABS: f64 = 1.0e-4;
/// Relative tolerance for the tiny fixture.
const TINY_REL: f64 = 1.0e-4;

fn reference() -> Value {
    let path = common::fixture_dir().join("reference-logits.json");
    let text = std::fs::read_to_string(&path).expect("committed reference logits");
    serde_json::from_str(&text).expect("reference json")
}

fn reference_logits(document: &Value) -> Vec<Vec<f32>> {
    document["logits"]
        .as_array()
        .expect("logits array")
        .iter()
        .map(|position| {
            position
                .as_array()
                .expect("position array")
                .iter()
                .map(|value| value.as_f64().expect("float") as f32)
                .collect()
        })
        .collect()
}

#[test]
fn rust_forward_matches_the_numpy_reference_on_the_tiny_fixture() {
    let temp = common::TempDir::new("tiny-parity");
    let bundle = common::build_tiny_bundle(temp.path());

    let document = reference();
    let prompt: Vec<u32> = document["prompt_ids"]
        .as_array()
        .expect("prompt ids")
        .iter()
        .map(|value| u32::try_from(value.as_u64().expect("id")).expect("u32"))
        .collect();
    let expected = reference_logits(&document);
    let vocab = document["vocab"].as_u64().expect("vocab") as usize;
    assert_eq!(expected.len(), prompt.len() + 2);

    let mut config = common::tiny_config(&bundle, prompt.clone(), 2);
    config.budget = common::tiny_budget();
    let mut runner = Runner::open(config).expect("open runner");

    // Prefill, then two greedy steps, so the cache path is compared too.
    let mut observed: Vec<Vec<f32>> = Vec::new();
    let mut next = 0_u32;
    for (position, token) in prompt.iter().enumerate() {
        let step = runner
            .forward(*token, position, selvedge_runtime::account::Phase::Prefill)
            .expect("prefill");
        next = selvedge_runtime::model::greedy_token(&step.logits).expect("greedy");
        observed.push(step.logits);
    }
    for index in 0..2_usize {
        let position = prompt.len() + index;
        let step = runner
            .forward(next, position, selvedge_runtime::account::Phase::Decode)
            .expect("decode");
        next = selvedge_runtime::model::greedy_token(&step.logits).expect("greedy");
        observed.push(step.logits);
    }

    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    for (position, (reference, actual)) in expected.iter().zip(observed.iter()).enumerate() {
        assert_eq!(reference.len(), vocab);
        assert_eq!(actual.len(), vocab);
        for (index, (left, right)) in reference.iter().zip(actual.iter()).enumerate() {
            let difference = (f64::from(*left) - f64::from(*right)).abs();
            let relative = if *left == 0.0 {
                difference
            } else {
                difference / f64::from(*left).abs()
            };
            assert!(
                difference <= TINY_ABS + TINY_REL * f64::from(*left).abs(),
                "position {position} index {index}: reference {left} vs {right} \
                 (abs {difference} rel {relative})"
            );
            max_abs = max_abs.max(difference);
            max_rel = max_rel.max(relative);
        }
    }
    // A reference that drifts this far is a bug, not a tolerance question.
    assert!(
        max_abs < 1.0e-5,
        "max abs {max_abs} is larger than expected"
    );
    assert!(
        max_rel < 1.0e-4,
        "max rel {max_rel} is larger than expected"
    );

    // The reference's own greedy continuation must be reproduced exactly.
    let expected_ids: Vec<u32> = document["generated_ids"]
        .as_array()
        .expect("generated ids")
        .iter()
        .map(|value| u32::try_from(value.as_u64().expect("id")).expect("u32"))
        .collect();
    let mut generated = Vec::new();
    next = selvedge_runtime::model::greedy_token(&observed[prompt.len() - 1]).expect("greedy");
    for index in 0..expected_ids.len() {
        generated.push(next);
        next =
            selvedge_runtime::model::greedy_token(&observed[prompt.len() + index]).expect("greedy");
    }
    assert_eq!(generated, expected_ids);

    eprintln!(
        "tiny parity: positions={} vocab={} max_abs={max_abs} max_rel={max_rel}",
        observed.len(),
        vocab
    );
}

#[test]
fn exact_mode_classifies_as_proven_and_approximate_does_not() {
    let temp = common::TempDir::new("tiny-modes");
    let bundle = common::build_tiny_bundle(temp.path());

    let mut exact = common::tiny_config(&bundle, vec![1, 5, 9, 17], 0);
    exact.budget = common::tiny_budget();
    let mut runner = Runner::open(exact).expect("open exact runner");
    let plan = runner.plan().clone();
    let step = runner
        .forward(1, 0, selvedge_runtime::account::Phase::Prefill)
        .expect("exact forward");
    assert!(step.exactness.is_proven(), "exact mode must prove equality");
    let witness = runner
        .mode()
        .last_witness()
        .expect("an exact token leaves a witness");
    assert_eq!(witness.declared_digest(), plan.native.digest());
    assert_eq!(witness.executed_digest(), plan.native.digest());
    assert_eq!(witness.shard_count() as usize, plan.shard_count);
    assert_eq!(step.group_shards as usize, plan.shard_count);

    // The same model in approximate mode must classify as approximate and must
    // produce different logits, because it executes one group per layer instead
    // of four.
    let mut approximate = common::tiny_config(&bundle, vec![1, 5, 9, 17], 0);
    approximate.budget = common::tiny_budget();
    approximate.mode = ExecutionMode::Approximate(common::one_group_selector());
    let mut runner = Runner::open(approximate).expect("open approximate runner");
    let approximate_step = runner
        .forward(1, 0, selvedge_runtime::account::Phase::Prefill)
        .expect("approximate forward");
    assert!(!approximate_step.exactness.is_proven());
    assert_eq!(
        approximate_step.group_shards as usize,
        plan.layers.len() * 3,
        "one group per layer is three shards per layer"
    );
    assert_ne!(
        logits_digest(&step.logits),
        logits_digest(&approximate_step.logits),
        "dropping three of four groups must change the arithmetic"
    );
    assert!(
        approximate_step.accounting.active_unique_params < step.accounting.active_unique_params
    );
}

#[test]
fn tokenizer_round_trips_the_pinned_vocabulary() {
    let model_dir = std::path::Path::new(
        "/hf-cache/models/Qwen/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca",
    );
    if !model_dir.join("tokenizer.json").exists() {
        eprintln!(
            "skipping: no pinned tokenizer at {} (the rust court mounts it read-only)",
            model_dir.display()
        );
        return;
    }
    let tokenizer = selvedge_runtime::GreedyTokenizer::from_file(&model_dir.join("tokenizer.json"))
        .expect("load tokenizer");
    let ids = tokenizer.encode("Hello, world").expect("encode");
    assert_eq!(ids, vec![9707, 11, 1879]);
    assert_eq!(tokenizer.decode(&ids).expect("decode"), "Hello, world");
    let chat = tokenizer.encode_chat("Hi").expect("chat encode");
    // `<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n`
    assert_eq!(chat.first().copied(), Some(151_644));
    assert!(chat.contains(&151_645));
}

#[test]
fn accounting_row_is_valid_jsonl_and_reports_every_required_field() {
    let temp = common::TempDir::new("tiny-accounting");
    let bundle = common::build_tiny_bundle(temp.path());
    let accounting_path = temp.path().join("accounting.jsonl");
    let mut config = common::tiny_config(&bundle, vec![1, 5, 9, 17], 1);
    config.budget = common::tiny_budget();
    config.account = Some(accounting_path.clone());
    let _ = config;
    let mut runner = Runner::open(config).expect("open runner");
    let outcome = runner.run().expect("run");
    assert!(outcome.ok());

    let text = std::fs::read_to_string(&accounting_path).expect("accounting file");
    let rows: Vec<serde_json::Value> = text
        .lines()
        .map(|line| serde_json::from_str(line).expect("json line"))
        .collect();
    assert_eq!(rows.len(), 5, "four prompt tokens plus one generated token");
    for row in &rows {
        assert_eq!(row["schema"], "selvedge/runtime-token-accounting/v1");
        let accounting = &row["accounting"];
        for field in [
            "active_unique_params",
            "resident_unique_params",
            "total_accessible_params",
            "parsed_parameter_reads",
            "learned_bytes_by_tier",
            "flops",
            "controller_flops",
            "quant_dequant_work",
            "within_budget",
            "budget_limit",
        ] {
            assert!(
                !accounting[field].is_null(),
                "accounting row is missing {field}: {row}"
            );
        }
        assert_eq!(accounting["within_budget"], true);
        assert_eq!(row["budget"]["kind"], "within");
        assert!(row["accounting"]["flops"].as_u64().unwrap() > 0);
        assert!(row["read_calls"].as_u64().unwrap() > 0);
    }
}
