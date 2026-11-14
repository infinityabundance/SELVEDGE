//! Court B additions for the Atlas crate.
//!
//! These tests are the executable contract for:
//! - parity with the Python selector arms on exported real fixtures;
//! - deterministic decisions;
//! - interaction-graph invariants;
//! - the no-leakage boundary of `SelectorInput`;
//! - calibration monotonicity where it is meaningful;
//! - independent matched-budget accounting and paired comparison;
//! - the two-view head contract.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::PathBuf;

use selvedge_atlas::{
    AtlasError, AtlasWeights, CompareArm, CompareInput, CompareItem, CostEstimate, CostHead,
    DecisionStatus, FitData, InteractionGraph, QualityObservation, SelectorInput, SelectorKind,
    TwoViewDecision, UnavailableCostHead, ValueEstimate, combine, compare, fit, is_monotone,
    read_fixtures, reliability_curve, select_named,
};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate directory has a parent")
        .parent()
        .expect("crates directory has a parent")
        .to_path_buf()
}

fn weights() -> AtlasWeights {
    let path = repo_root().join("results/selectors/m2-weights.json");
    let text = std::fs::read_to_string(&path).expect("weights fixture is tracked");
    let weights: AtlasWeights = serde_json::from_str(&text).expect("weights parse");
    weights.validate().expect("weights validate");
    weights
}

fn observation(group: u32) -> QualityObservation {
    QualityObservation {
        model_digest: "sha256:test".into(),
        layer: 0,
        group,
        context_id: "doc:0:3".into(),
        base_set: vec![0, 1, 2, 3],
        metric: "teacher_kl".into(),
        delta: 0.5,
        uncertainty: 0.1,
        source_receipt: "results/test.json".into(),
        stamp: "20261114T000000Z".into(),
    }
}

#[test]
fn parity_with_python_selector_arms() {
    let weights = weights();
    let path = repo_root().join("results/selectors/fixtures/selector-inputs.json");
    let fixtures = read_fixtures(&path).expect("fixtures parse");
    assert_eq!(fixtures.schema, "selvedge/atlas/fixtures-v1");
    assert!(!fixtures.records.is_empty());
    let kinds = [
        SelectorKind::Magnitude,
        SelectorKind::Frequency,
        SelectorKind::Random,
        SelectorKind::StaticMagnitude,
        SelectorKind::StaticFrequency,
        SelectorKind::LearnedUnary,
        SelectorKind::LearnedInteraction,
    ];
    for kind in kinds {
        for record in &fixtures.records {
            let input = record.input();
            input
                .validate(weights.layers as u32, weights.groups as u32)
                .expect("input validates");
            let selected = select_named(kind, &input, &weights, 5)
                .unwrap_or_else(|error| panic!("{} failed: {error}", kind.name()));
            let expected = match kind {
                SelectorKind::Magnitude => &record.expected.magnitude,
                SelectorKind::Frequency => &record.expected.frequency,
                SelectorKind::Random => &record.expected.random,
                SelectorKind::StaticMagnitude => &record.expected.static_magnitude,
                SelectorKind::StaticFrequency => &record.expected.static_frequency,
                SelectorKind::LearnedUnary => &record.expected.learned_unary,
                SelectorKind::LearnedInteraction => &record.expected.learned_interaction,
            };
            assert_eq!(
                &selected,
                expected,
                "{} mismatch at layer {} position {}",
                kind.name(),
                record.layer,
                record.position
            );
        }
    }
}

#[test]
fn selection_is_deterministic() {
    let weights = weights();
    let path = repo_root().join("results/selectors/fixtures/selector-inputs.json");
    let fixtures = read_fixtures(&path).expect("fixtures parse");
    let input = fixtures.records[0].input();
    for kind in [
        SelectorKind::Magnitude,
        SelectorKind::Random,
        SelectorKind::LearnedInteraction,
    ] {
        let first = select_named(kind, &input, &weights, 5).expect("select");
        let second = select_named(kind, &input, &weights, 5).expect("select");
        assert_eq!(first, second);
    }
}

#[test]
fn fitted_weights_reproduce_the_python_model_and_decisions() {
    let python = weights();
    let path = repo_root().join("results/selectors/m2-fit-data.json");
    let text = std::fs::read_to_string(&path).expect("fit data is tracked");
    let mut data: FitData = serde_json::from_str(&text).expect("fit data parse");
    data.lambda_mix = python.lambda_mix;
    let fitted = fit(&data).expect("fit");
    fitted.validate().expect("fitted weights validate");
    assert_eq!(fitted.layers, python.layers);
    assert_eq!(fitted.groups, python.groups);
    for layer in 0..python.layers {
        for index in 0..python.unary_weights[layer].len() {
            let difference =
                (fitted.unary_weights[layer][index] - python.unary_weights[layer][index]).abs();
            assert!(
                difference < 1e-9,
                "unary weight differs at layer {layer} index {index}: {difference}"
            );
        }
        for row in 0..5 {
            for column in 0..5 {
                let difference = (fitted.interaction[layer][row][column]
                    - python.interaction[layer][row][column])
                    .abs();
                assert!(
                    difference < 1e-9,
                    "interaction differs at layer {layer} [{row}][{column}]: {difference}"
                );
            }
        }
    }
    let fixtures =
        read_fixtures(&repo_root().join("results/selectors/fixtures/selector-inputs.json"))
            .expect("fixtures parse");
    for record in fixtures.records.iter().take(64) {
        let input = record.input();
        for (kind, expected) in [
            (SelectorKind::LearnedUnary, &record.expected.learned_unary),
            (
                SelectorKind::LearnedInteraction,
                &record.expected.learned_interaction,
            ),
        ] {
            let selected = select_named(kind, &input, &fitted, 5).expect("select");
            assert_eq!(
                &selected,
                expected,
                "fitted {} decision differs",
                kind.name()
            );
        }
    }
}

#[test]
fn selector_input_cannot_carry_future_state() {
    let base = r#"{"layer":0,"layers":28,"position":3,"sequence_key":7,"groups":[]"#;
    let future = format!("{base},\"future_tokens\":[1,2,3]}}");
    let logits = format!("{base},\"teacher_logits\":[0.1]}}");
    let labels = format!("{base},\"gold_label\":4}}");
    assert!(serde_json::from_str::<SelectorInput>(&future).is_err());
    assert!(serde_json::from_str::<SelectorInput>(&logits).is_err());
    assert!(serde_json::from_str::<SelectorInput>(&labels).is_err());
    let valid =
        r#"{"layer":0,"layers":28,"position":3,"sequence_key":7,"groups":[],"hidden":[0.0]}"#;
    let input: SelectorInput = serde_json::from_str(valid).expect("valid input parses");
    let encoded = serde_json::to_string(&input).expect("serialize");
    assert!(!encoded.contains("future"));
    assert!(!encoded.contains("logit"));
    assert!(!encoded.contains("gold"));
}

#[test]
fn interaction_graph_invariants() {
    let mut graph = InteractionGraph::default();
    graph
        .insert_observation(&observation(4), 28, 16)
        .expect("insert");
    let second = QualityObservation {
        delta: 0.7,
        ..observation(4)
    };
    graph.insert_observation(&second, 28, 16).expect("insert");
    let node = graph.nodes.values().next().expect("node exists");
    assert_eq!(node.count, 2);
    assert!((node.mean - 0.6).abs() < 1e-12);
    assert!((node.variance() - 0.02).abs() < 1e-12);
    graph
        .insert_interaction(0, 9, 3, 0.25, 28, 16)
        .expect("edge");
    graph
        .insert_interaction(0, 3, 9, -0.25, 28, 16)
        .expect("edge");
    assert_eq!(graph.edges.len(), 1);
    let edge = graph.edges.values().next().expect("edge exists");
    assert_eq!(edge.count, 2);
    assert_eq!(edge.positive, 1);
    assert_eq!(edge.negative, 1);
    assert!(graph.check_invariants(28, 16).is_ok());
    assert!(graph.insert_interaction(0, 4, 4, 0.0, 28, 16).is_err());
    assert!(graph.insert_interaction(0, 1, 2, f64::NAN, 28, 16).is_err());
    assert!(graph.insert_interaction(0, 100, 2, 0.0, 28, 16).is_err());
    let bad = QualityObservation {
        base_set: vec![1, 4, 9],
        ..observation(4)
    };
    assert!(bad.validate(28, 16).is_err());
    let unsorted = QualityObservation {
        base_set: vec![3, 1],
        ..observation(4)
    };
    assert!(unsorted.validate(28, 16).is_err());
    assert!(matches!(
        graph.insert_observation(&observation(99), 28, 16),
        Err(AtlasError::Invalid { .. })
    ));
}

#[test]
fn calibration_monotonicity() {
    let good = [
        (0.05, false),
        (0.15, false),
        (0.35, true),
        (0.45, false),
        (0.75, true),
        (0.95, true),
    ];
    let points = reliability_curve(&good, 4).expect("curve");
    assert!(is_monotone(&points));
    let bad = [(0.1, true), (0.9, false), (0.95, false)];
    let points = reliability_curve(&bad, 3).expect("curve");
    assert!(!is_monotone(&points));
    assert!(reliability_curve(&[(1.5, true)], 4).is_err());
    assert!(reliability_curve(&[], 4).is_err());
}

fn arm(id: &str, budget: &str, groups: u32, controller: u64, kls: &[f64]) -> CompareArm {
    let mut groups_per_layer = vec![groups; 28];
    if budget == "B142" {
        // B142 is five groups per layer plus two extras chosen by the arm.
        groups_per_layer[0] += 2;
    }
    CompareArm {
        id: id.into(),
        budget: budget.into(),
        groups_per_layer,
        group_width: 384,
        controller_elements: controller,
        items: kls
            .iter()
            .enumerate()
            .map(|(index, value)| CompareItem {
                doc_id: format!("doc-{index}"),
                kl_mean: *value,
            })
            .collect(),
    }
}

#[test]
fn matched_budget_accounting_and_comparison() {
    let base = arm(
        "token-magnitude",
        "B142",
        5,
        868,
        &[6.0, 6.5, 7.0, 6.2, 6.8],
    );
    let same = arm("same", "B142", 5, 868, &[6.1, 6.4, 7.1, 6.3, 6.9]);
    let worse = arm("worse", "B142", 5, 868, &[10.0, 10.5, 11.0, 10.2, 10.8]);
    let input = CompareInput {
        schema: "selvedge/atlas/compare-input-v1".into(),
        ceiling: 1_000_000_000,
        backbone_elements: 663_610_368,
        hidden: 2048,
        intermediate: 6144,
        baseline: "token-magnitude".into(),
        arms: vec![base, same, worse],
    };
    let report = compare(&input, 200).expect("compare");
    assert!(report.ok);
    assert_eq!(report.arms[0].active_unique_elements, 998_631_268);
    assert!(report.arms.iter().all(|arm| arm.matched));
    let worse_entry = report
        .comparisons
        .iter()
        .find(|entry| entry.arm == "worse")
        .expect("comparison exists");
    assert!(worse_entry.available);
    assert!(worse_entry.relative_reduction.unwrap_or(0.0) < 0.0);
    assert!(!worse_entry.passes_five_percent);

    let mut over = input.clone();
    over.arms = vec![CompareArm {
        groups_per_layer: vec![16; 28],
        ..arm("over", "dense", 16, 0, &[0.0])
    }];
    over.baseline = "over".into();
    assert!(matches!(compare(&over, 10), Err(AtlasError::Budget(_))));
}

#[test]
fn two_view_contract_is_explicit_when_cost_is_missing() {
    let value = ValueEstimate::new(0.3, 0.01, 0.2, 0.4, "receipt").expect("value");
    let cost = UnavailableCostHead::new("M3 trace scheduler").cost_to_materialize_now("shard");
    let decision = combine(value.clone(), cost.expect("cost"), 1.0).expect("combine");
    assert_eq!(decision.status, DecisionStatus::CostUnavailable);
    assert!(decision.score.is_none());
    let measured = CostEstimate::measured(0.02, 1024, 0.1).expect("cost");
    let decision = combine(value, measured, 2.0).expect("combine");
    assert_eq!(decision.status, DecisionStatus::Ready);
    assert!((decision.score.unwrap_or(0.0) - 0.26).abs() < 1e-12);
    assert!(matches!(
        combine(
            ValueEstimate::new(0.0, 0.0, 0.0, 0.0, "x").expect("value"),
            CostEstimate::unavailable("none"),
            -1.0
        ),
        Err(AtlasError::Invalid { .. })
    ));
    let _: TwoViewDecision = decision;
}
