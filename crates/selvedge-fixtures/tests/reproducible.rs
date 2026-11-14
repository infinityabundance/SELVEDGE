//! The committed fixture must be exactly what the generator produces.

#![allow(clippy::unwrap_used)]

use std::path::PathBuf;

use selvedge_fixtures::{FIXTURE_BUNDLE_ID, tiny_bundle};
use selvedge_format::sidecar::validate_sidecar;
use selvedge_format::{ValidationOptions, validate_bundle};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

#[test]
fn committed_bundle_matches_the_generator() {
    let root = repo_root();
    let path = root
        .join("fixtures")
        .join(format!("{FIXTURE_BUNDLE_ID}.slv"));
    let committed = std::fs::read(&path).unwrap();
    let generated = tiny_bundle().unwrap().encoded().unwrap();
    assert_eq!(
        committed,
        generated,
        "fixtures/{} is stale; run the fixture generator",
        path.file_name().unwrap().to_string_lossy()
    );
    validate_bundle(&path, &ValidationOptions::default()).unwrap();
}

#[test]
fn committed_sidecar_matches_the_bundle() {
    let root = repo_root();
    let bundle = root
        .join("fixtures")
        .join(format!("{FIXTURE_BUNDLE_ID}.slv"));
    let sidecar = root
        .join("fixtures")
        .join(format!("{FIXTURE_BUNDLE_ID}.manifest.json"));
    validate_sidecar(&bundle, &sidecar).unwrap();
}

#[test]
fn committed_artifact_is_small() {
    let root = repo_root();
    let path = root
        .join("fixtures")
        .join(format!("{FIXTURE_BUNDLE_ID}.slv"));
    let length = std::fs::metadata(path).unwrap().len();
    assert!(
        length < 100 * 1024,
        "committed fixture is {length} bytes, above the 100 KiB budget"
    );
}
