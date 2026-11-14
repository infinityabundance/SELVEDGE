//! End-to-end CLI tests: doctor, inspect, format validate and the court
//! runners.

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use selvedge_core::testutil::TempDir;
use selvedge_fixtures::{FIXTURE_BUNDLE_ID, publish_tiny_bundle};
use selvedge_format::{ValidationOptions, validate_bundle};
use serde_json::Value;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

fn selvedge() -> Command {
    Command::new(env!("CARGO_BIN_EXE_selvedge"))
}

fn run(args: &[&str]) -> Output {
    selvedge().args(args).output().unwrap()
}

fn stdout_json(output: &Output) -> Value {
    serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "stdout is not JSON ({error}): {}",
            String::from_utf8_lossy(&output.stdout)
        )
    })
}

fn publish_fixture() -> (TempDir, PathBuf) {
    let dir = TempDir::new("selvedge-cli-test").unwrap();
    let published = publish_tiny_bundle(dir.path(), true).unwrap();
    (dir, published.path)
}

#[test]
fn doctor_json_reports_all_checks() {
    let output = run(&["doctor", "--json", "--root", repo_root().to_str().unwrap()]);
    assert!(
        output.status.success(),
        "doctor failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report = stdout_json(&output);
    assert_eq!(report["schema"], "selvedge/doctor/v1");
    assert_eq!(report["ok"], true);
    let checks = report["checks"].as_array().unwrap();
    let names: Vec<&str> = checks
        .iter()
        .filter_map(|check| check["name"].as_str())
        .collect();
    for required in [
        "config",
        "format_schema",
        "court_images",
        "target_dir",
        "disk",
        "gpu",
    ] {
        assert!(
            names.contains(&required),
            "doctor did not report {required}; got {names:?}"
        );
    }
    let gpu = checks.iter().find(|check| check["name"] == "gpu").unwrap();
    assert_eq!(gpu["data"]["cuda_compiled_in"], false);
}

#[test]
fn doctor_human_output_exits_zero() {
    let output = run(&["doctor", "--root", repo_root().to_str().unwrap()]);
    assert!(output.status.success());
    let text = String::from_utf8_lossy(&output.stdout);
    assert!(text.contains("selvedge doctor"));
    assert!(text.contains("format_schema"));
    assert!(text.contains("overall: ok"));
}

#[test]
fn doctor_probe_host_reports_cpu_and_memory() {
    let output = run(&[
        "doctor",
        "--json",
        "--probe-host",
        "--root",
        repo_root().to_str().unwrap(),
    ]);
    // The probe itself may fail in an unusual sandbox, but it must emit JSON
    // and must name the host checks it attempted.
    let report = stdout_json(&output);
    let names: Vec<&str> = report["checks"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|check| check["name"].as_str())
        .collect();
    assert!(names.contains(&"host_cpu"));
    assert!(names.contains(&"host_memory"));
}

#[test]
fn inspect_bundle_prints_a_shard_table() {
    let (_dir, bundle) = publish_fixture();
    let output = run(&["inspect", bundle.to_str().unwrap(), "--json"]);
    assert!(
        output.status.success(),
        "inspect failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report = stdout_json(&output);
    assert_eq!(report["schema"], "selvedge/inspect/v1");
    assert_eq!(report["kind"], "bundle");
    assert_eq!(report["bundle_id"], FIXTURE_BUNDLE_ID);
    assert!(report["shard_count"].as_u64().unwrap() > 0);
    assert!(report["total_shard_bytes"].as_u64().unwrap() > 0);
    assert!(!report["shards"].as_array().unwrap().is_empty());
}

#[test]
fn inspect_accepts_a_bare_manifest() {
    let (dir, _bundle) = publish_fixture();
    let manifest = selvedge_fixtures::tiny_bundle().unwrap().manifest;
    let path = dir.join("manifest-only.json");
    std::fs::write(&path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
    let output = run(&["inspect", path.to_str().unwrap(), "--json"]);
    assert!(output.status.success());
    let report = stdout_json(&output);
    assert_eq!(report["kind"], "manifest");
    assert_eq!(report["bundle_id"], FIXTURE_BUNDLE_ID);
}

#[test]
fn format_validate_accepts_the_fixture() {
    let (_dir, bundle) = publish_fixture();
    let output = run(&["format", "validate", bundle.to_str().unwrap(), "--json"]);
    assert!(
        output.status.success(),
        "validate failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report = stdout_json(&output);
    assert_eq!(report["schema"], "selvedge/validation-report/v1");
    assert!(report["shard_count"].as_u64().unwrap() > 0);
    assert_eq!(report["verify_mode"], "all");
}

#[test]
fn format_validate_structural_mode_is_explicit() {
    let (_dir, bundle) = publish_fixture();
    let output = run(&[
        "format",
        "validate",
        bundle.to_str().unwrap(),
        "--verify",
        "none",
        "--json",
    ]);
    assert!(output.status.success());
    let report = stdout_json(&output);
    assert_eq!(report["verify_mode"], "structural_only");
    assert_eq!(report["verified_shards"].as_array().unwrap().len(), 0);
}

#[test]
fn format_validate_rejects_a_corrupted_payload() {
    let (dir, bundle) = publish_fixture();
    let report = validate_bundle(&bundle, &ValidationOptions::default()).unwrap();
    let mut bytes = std::fs::read(&bundle).unwrap();
    let index = usize::try_from(report.payload_offset + 4).unwrap();
    bytes[index] ^= 0x01;
    let corrupt = dir.join("corrupt.slv");
    std::fs::write(&corrupt, bytes).unwrap();
    let output = run(&["format", "validate", corrupt.to_str().unwrap()]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("invalid"), "stderr: {stderr}");
    assert!(stderr.contains("sha256 mismatch"), "stderr: {stderr}");
}

#[test]
fn court_c_runs_every_budget_scenario() {
    let output = run(&["court", "c", "--json"]);
    assert!(
        output.status.success(),
        "court c failed: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    let report = stdout_json(&output);
    assert_eq!(report["court"], "c");
    assert_eq!(report["passed"], true);
    let names: Vec<&str> = report["scenarios"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|scenario| scenario["name"].as_str())
        .collect();
    assert!(names.contains(&"tied_embeddings_counted_once"));
    assert!(names.contains(&"boundary_exactly_at_limit_is_within"));
    assert!(names.contains(&"fallback_that_remains_over_budget_is_rejected"));
}

#[test]
fn court_d_runs_storage_scenarios_in_a_kept_workdir() {
    let dir = TempDir::new("selvedge-court-d-workdir").unwrap();
    let workdir = dir.join("court-d");
    let output = run(&[
        "court",
        "d",
        "--json",
        "--workdir",
        workdir.to_str().unwrap(),
    ]);
    assert!(
        output.status.success(),
        "court d failed: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    let report = stdout_json(&output);
    assert_eq!(report["court"], "d");
    assert_eq!(report["passed"], true);
    let names: Vec<&str> = report["scenarios"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|scenario| scenario["name"].as_str())
        .collect();
    for required in [
        "valid_bundle_validates",
        "payload_bit_flip_detected",
        "truncated_bundle_detected",
        "cancelled_read_aborts_and_recovers",
        "concurrent_readers_never_see_torn_bundle",
        "hot_reload_rejects_conflicting_bundle_id",
        "sidecar_matches_bundle",
    ] {
        assert!(names.contains(&required), "missing {required}: {names:?}");
    }
    assert!(workdir.join(format!("{FIXTURE_BUNDLE_ID}.slv")).is_file());
}

#[test]
fn version_is_reported() {
    let output = run(&["--version"]);
    assert!(output.status.success());
    let text = String::from_utf8_lossy(&output.stdout);
    assert!(text.contains("selvedge"));
}

fn assert_paths_exist(paths: &[&Path]) {
    for path in paths {
        assert!(path.exists(), "{} should exist", path.display());
    }
}

#[test]
fn helper_paths_exist() {
    let (_dir, bundle) = publish_fixture();
    assert_paths_exist(&[&bundle]);
}
