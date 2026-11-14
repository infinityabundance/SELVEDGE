//! Court C — budget accounting.
//!
//! Each named test exercises one invariant from the mandate's accounting
//! contract. The scenarios live in `selvedge_core::courts` so the CLI verifier
//! and these tests cannot drift apart.

#![allow(clippy::unwrap_used)]

use selvedge_core::courts::{
    ScenarioOutcome, adapters_memory_router_verifier_counted, boundary_exactly_at_limit_is_within,
    boundary_one_over_limit_is_rejected, fallback_that_remains_over_budget_is_rejected,
    over_budget_allowed_and_labelled_in_research_mode, over_budget_rejected_in_compliance,
    overlapping_slices_deduplicated, plan_fallback_within_budget,
    quantized_metadata_excluded_from_learned_count, recurrent_reuse_unique_vs_reads,
    run_budget_court, tied_embeddings_counted_once, token_accounting_reports_every_required_field,
};

fn assert_passed(outcome: ScenarioOutcome) {
    assert!(
        outcome.passed,
        "scenario {} failed: {}",
        outcome.name, outcome.detail
    );
}

#[test]
fn tied_embeddings_are_counted_once() {
    assert_passed(tied_embeddings_counted_once());
}

#[test]
fn overlapping_and_adjacent_slices_are_deduplicated() {
    assert_passed(overlapping_slices_deduplicated());
}

#[test]
fn recurrent_reuse_counts_unique_params_once_but_reads_every_time() {
    assert_passed(recurrent_reuse_unique_vs_reads());
}

#[test]
fn adapters_memory_router_and_verifier_are_learned() {
    assert_passed(adapters_memory_router_verifier_counted());
}

#[test]
fn over_budget_decision_is_rejected_in_compliance_mode() {
    assert_passed(over_budget_rejected_in_compliance());
}

#[test]
fn over_budget_decision_is_allowed_only_when_labelled_as_teacher_mode() {
    assert_passed(over_budget_allowed_and_labelled_in_research_mode());
}

#[test]
fn exactly_one_billion_active_unique_params_is_within_budget() {
    assert_passed(boundary_exactly_at_limit_is_within());
}

#[test]
fn one_billion_and_one_active_unique_params_is_rejected() {
    assert_passed(boundary_one_over_limit_is_rejected());
}

#[test]
fn quantization_metadata_is_not_counted_as_learned_parameters() {
    assert_passed(quantized_metadata_excluded_from_learned_count());
}

#[test]
fn a_within_budget_fallback_can_be_substituted_before_execution() {
    assert_passed(plan_fallback_within_budget());
}

#[test]
fn a_fallback_that_is_still_over_budget_is_rejected() {
    assert_passed(fallback_that_remains_over_budget_is_rejected());
}

#[test]
fn token_accounting_reports_all_required_counters() {
    assert_passed(token_accounting_reports_every_required_field());
}

#[test]
fn the_whole_court_c_report_passes_and_is_stable() {
    let first = run_budget_court();
    let second = run_budget_court();
    assert_eq!(first, second);
    assert_eq!(first.court, "c");
    for scenario in &first.scenarios {
        assert!(
            scenario.passed,
            "scenario {} failed: {}",
            scenario.name, scenario.detail
        );
    }
    assert!(!first.scenarios.is_empty());
}
