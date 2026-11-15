//! Compile-fail cases for the exactness and verdict barriers.
//!
//! Each file under `tests/ui/` must fail to compile with the substring named in
//! its `//@ expect:` line. This is the offline equivalent of trybuild: the test
//! invokes `rustc` directly against the already-built `selvedge_core` rlib, so
//! no extra dependency and no network access is needed. A case that starts
//! compiling is a failure of this test, not a warning.
//!
//! Documented `tests/ui` approach: rationale in `docs/ARCHITECTURE.md`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};
use std::process::Command;

fn workspace_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(Path::parent)
        .expect("crate is inside the workspace")
        .to_path_buf()
}

fn target_dir() -> PathBuf {
    std::env::var_os("CARGO_TARGET_DIR")
        .map_or_else(|| workspace_root().join("target"), PathBuf::from)
}

fn crate_rlibs(crate_name: &str) -> Vec<PathBuf> {
    let deps = target_dir().join("debug/deps");
    let prefix = format!("lib{crate_name}-");
    let mut found: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(&deps)
        .unwrap_or_else(|err| panic!("cannot read {deps:?}: {err}"))
        .filter_map(Result::ok)
        .map(|entry| {
            let modified = entry
                .metadata()
                .and_then(|metadata| metadata.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
            (modified, entry.path())
        })
        .filter(|(_, path)| {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            name.starts_with(&prefix) && name.ends_with(".rlib")
        })
        .collect();
    // Newest first. A stale rlib from an earlier build must never decide the
    // barrier: it can still expose a constructor the current source removed.
    found.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| right.1.cmp(&left.1)));
    found.into_iter().map(|(_, path)| path).collect()
}

fn expected_error(source: &str) -> String {
    source
        .lines()
        .find_map(|line| line.strip_prefix("//@ expect:"))
        .unwrap_or_else(|| panic!("UI case has no `//@ expect:` line"))
        .trim()
        .to_owned()
}

#[test]
fn ui_cases_fail_to_compile() {
    let ui_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/ui");
    let rlibs = crate_rlibs("selvedge_core");
    assert!(
        !rlibs.is_empty(),
        "no selvedge_core rlib under {:?}; run cargo test after cargo build",
        target_dir()
    );
    let store_rlibs = crate_rlibs("selvedge_store");
    assert!(
        !store_rlibs.is_empty(),
        "no selvedge_store rlib under {:?}; the scope-is-not-exactness case needs the real \
         ExecutionScope, so the dev-dependency must be built",
        target_dir()
    );
    let store_rlib = &store_rlibs[0];
    let deps = target_dir().join("debug/deps");
    let out_dir = std::env::temp_dir().join(format!("selvedge-ui-{}", std::process::id()));
    std::fs::create_dir_all(&out_dir).expect("create UI output dir");

    let mut cases: Vec<PathBuf> = std::fs::read_dir(&ui_dir)
        .unwrap_or_else(|err| panic!("cannot read {ui_dir:?}: {err}"))
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("rs"))
        .collect();
    cases.sort();
    assert!(!cases.is_empty(), "no UI cases in {ui_dir:?}");

    for case in cases {
        let source = std::fs::read_to_string(&case).expect("read UI case");
        let expected = expected_error(&source);
        let stem = case.file_stem().unwrap().to_string_lossy().to_string();
        let rlib = rlibs.first().expect("a core rlib was asserted above");
        // Only the newest rlib is admissible. Trying older builds could let a
        // removed constructor appear to exist again.
        let output = Command::new("rustc")
            .arg("--edition=2021")
            .arg("--crate-type=lib")
            .arg("--crate-name")
            .arg(format!("ui_{stem}"))
            .arg("--extern")
            .arg(format!("selvedge_core={}", rlib.display()))
            .arg("--extern")
            .arg(format!("selvedge_store={}", store_rlib.display()))
            .arg("-L")
            .arg(format!("dependency={}", deps.display()))
            .arg("--out-dir")
            .arg(&out_dir)
            .arg(&case)
            .output()
            .expect("rustc must be available in the rust court");
        assert!(
            !output.status.success(),
            "UI case {stem} compiled, but it must be rejected: {expected}"
        );
        let last_stderr = String::from_utf8_lossy(&output.stderr).to_string();
        assert!(
            last_stderr.contains(&expected),
            "UI case {stem} failed for the wrong reason.\nexpected: {expected}\nstderr:\n{last_stderr}"
        );
    }
    let _ = std::fs::remove_dir_all(&out_dir);
}
