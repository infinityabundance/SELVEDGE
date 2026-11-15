//! T2 admission-ledger tests: one per registered arm, plus the over-budget
//! refusal guard for each arm.
//!
//! The tracked configs under `results/train/admission-configs/t2/` are the same
//! bytes `scripts/train-admission-t2.sh` admits. This test re-admits every one
//! through the real `selvedge admit` binary and requires the union to equal the
//! pre-registered design count (docs/EXPERIMENT-CONTRACT.md T2.1). It then
//! mutates each config so the declared union exceeds the ceiling and requires a
//! refusal with no admitted path: the ledger must refuse an over-budget T2 arm
//! before training starts, not merely accept the good case.

#![allow(clippy::unwrap_used)]

use std::path::PathBuf;
use std::process::{Command, Output};

use selvedge_core::testutil::TempDir;
use serde_json::{Value, json};

/// (arm, active unique, total accessible) — T2.1 arithmetic.
const T2_ARMS: &[(&str, u64, u64)] = &[
    ("D", 772_210_688, 772_210_688),
    ("M", 598_153_216, 866_586_624),
    ("X", 772_440_064, 948_600_832),
    ("H", 774_543_360, 1_219_137_536),
    ("H*", 774_543_360, 1_219_137_536),
];

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

fn config_path(arm: &str) -> PathBuf {
    repo_root()
        .join("results/train/admission-configs/t2")
        .join(format!("{arm}.json"))
}

fn selvedge(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_selvedge"))
        .args(args)
        .output()
        .unwrap()
}

fn stdout_json(output: &Output) -> Value {
    serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "stdout is not JSON ({error}): {}",
            String::from_utf8_lossy(&output.stdout)
        )
    })
}

#[test]
fn every_t2_arm_is_admitted_at_its_design_count() {
    for (arm, active, total) in T2_ARMS {
        let path = config_path(arm);
        assert!(path.is_file(), "missing T2 config {}", path.display());
        let config: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(config["arm"], json!(arm));
        assert_eq!(
            config["total_accessible"].as_u64().unwrap(),
            *total,
            "{arm}"
        );
        let output = selvedge(&["admit", "--config", path.to_str().unwrap(), "--json"]);
        assert!(
            output.status.success(),
            "admit refused {arm}: {}",
            String::from_utf8_lossy(&output.stdout)
        );
        let record = stdout_json(&output);
        assert_eq!(record["schema"], "selvedge/admission-record/v1");
        assert_eq!(record["admitted"], true, "{arm}");
        assert_eq!(
            record["active_unique_params"].as_u64().unwrap(),
            *active,
            "{arm}"
        );
        assert_eq!(
            record["total_accessible"].as_u64().unwrap(),
            *total,
            "{arm}"
        );
        assert!(record["active_unique_params"].as_u64().unwrap() <= 1_000_000_000);
        assert_eq!(record["verdict"]["kind"], "within");
    }
}

#[test]
fn every_t2_arm_config_is_refused_when_it_is_pushed_over_the_ceiling() {
    for (arm, active, total) in T2_ARMS {
        let dir = TempDir::new("selvedge-t2-over-budget").unwrap();
        let mut config: Value =
            serde_json::from_slice(&std::fs::read(config_path(arm)).unwrap()).unwrap();
        // A new identity of 1e9 elements cannot overlap the declared slices.
        let over = 1_000_000_000_u64;
        config["execution_uses"]
            .as_array_mut()
            .unwrap()
            .push(json!({
                "identity": {"tensor": "t2.over.budget.probe",
                             "elements": {"start": 0, "end": over}},
                "reads": over,
                "bytes": over * 2,
                "tier": "gpu"
            }));
        let config_path = dir.join(&format!("{arm}-over.json"));
        std::fs::write(&config_path, serde_json::to_vec_pretty(&config).unwrap()).unwrap();
        let record_path = dir.join(&format!("{arm}-record.json"));
        let output = selvedge(&[
            "admit",
            "--config",
            config_path.to_str().unwrap(),
            "--out",
            record_path.to_str().unwrap(),
        ]);
        assert!(
            !output.status.success(),
            "{arm}: over-budget config was admitted"
        );
        let record: Value = serde_json::from_slice(&std::fs::read(&record_path).unwrap()).unwrap();
        assert_eq!(record["admitted"], false, "{arm}");
        // No admitted path exists for an over-budget arm: the verdict is absent.
        assert!(record["verdict"].is_null(), "{arm}");
        assert!(
            record["active_unique_params"].as_u64().unwrap() >= active + over,
            "{arm}"
        );
        assert!(
            record["total_accessible"].as_u64().unwrap() == *total,
            "{arm}: the probe must not change the declared total"
        );
        assert!(record["refusal"].is_string(), "{arm}");
    }
}
