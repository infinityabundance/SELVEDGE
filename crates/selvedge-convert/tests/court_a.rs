//! Named Court A reconstruction tests.
//!
//! Each test runs the full deterministic matrix and asserts that its named
//! scenario passed. Running the matrix repeatedly is cheap because every case
//! is tiny and bounded.

#![allow(clippy::unwrap_used)]

use selvedge_convert::court_a::{CourtAReport, run_court_a};

fn matrix() -> CourtAReport {
    run_court_a(0x5E1A_ED6E_51E1_2026).unwrap()
}

fn assert_scenario(report: &CourtAReport, name: &str) {
    let scenario = report
        .scenarios
        .iter()
        .find(|scenario| scenario.name == name)
        .unwrap_or_else(|| panic!("scenario {name:?} is missing"));
    assert!(scenario.passed, "chapter {name}: {}", scenario.detail);
}

#[test]
fn court_a_all_groups_equal_original() {
    let report = matrix();
    assert_scenario(&report, "all_groups_equal_original");
}

#[test]
fn court_a_group_permutation_invariance() {
    let report = matrix();
    assert_scenario(&report, "group_permutation_invariance");
}

#[test]
fn court_a_single_group_full_span() {
    let report = matrix();
    assert_scenario(&report, "single_group_full_span");
}

#[test]
fn court_a_empty_subset_is_exact_zero() {
    let report = matrix();
    assert_scenario(&report, "empty_subset_is_exact_zero");
}

#[test]
fn court_a_group_sizes_one_two_three_and_prime() {
    let report = matrix();
    for name in [
        "group_size_one",
        "group_size_two",
        "group_size_three",
        "group_size_prime",
    ] {
        assert_scenario(&report, name);
    }
}

#[test]
fn court_a_non_divisible_and_degenerate_dimensions() {
    let report = matrix();
    for name in [
        "non_divisible_grouping",
        "intermediate_one",
        "hidden_one",
        "input_shapes",
    ] {
        assert_scenario(&report, name);
    }
}

#[test]
fn court_a_storage_orders_and_dtypes() {
    let report = matrix();
    for name in [
        "column_major_source",
        "transposed_source",
        "f16_source",
        "bf16_source",
    ] {
        assert_scenario(&report, name);
    }
}

#[test]
fn court_a_twenty_random_seeds() {
    let report = matrix();
    assert_scenario(&report, "twenty_random_seeds");
}

#[test]
fn court_a_determinism_ids_hashes_and_validation() {
    let report = matrix();
    for name in [
        "determinism_same_bytes_and_ids",
        "shard_ids_and_hashes_match_manifest",
        "bundle_validates_all",
    ] {
        assert_scenario(&report, name);
    }
}

#[test]
fn court_a_small_memory_ceiling() {
    let report = matrix();
    assert_scenario(&report, "small_memory_ceiling");
}

#[test]
fn court_a_reports_every_named_scenario_passing() {
    let report = matrix();
    assert!(report.passed, "court a failed: {report:#?}");
    assert_eq!(report.scenario_count, report.scenarios.len() as u64);
    assert!(report.scenario_count >= 21);
    assert!(!report.measurements.is_empty());
    for measurement in &report.measurements {
        assert!(
            measurement.max_abs_f32.is_finite() && measurement.max_rel_f32.is_finite(),
            "non-finite measurement: {measurement:?}"
        );
    }
}
