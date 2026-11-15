//! `selvedge admit` — the pre-execution parameter-budget admission record.
//!
//! Training and evaluation runs must not start without a record from this
//! command. The config declares every learned-parameter slice the selector
//! (router) and the execution read for one output token; the command builds a
//! [`selvedge_core::ledger::SelectionPath`], admits it through
//! `ParameterLedger::admit_selection_path` under `ComplianceMode::Compliance`,
//! and writes the admitted path with the selector/execution read counts, their
//! union and the FLOP estimates.
//!
//! An over-budget config produces a refusal record and a non-zero exit; no
//! admitted path exists to execute. The record carries the SHA-256 of the config
//! bytes so a run can prove which declaration admitted it.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Args;
use selvedge_core::{
    Budget, BudgetError, BudgetVerdict, ComplianceMode, FlopEstimate, ParamUse, ParameterLedger,
    SelectionPath, Sha256Digest,
};
use serde::{Deserialize, Serialize};

use crate::CliResult;

/// Schema of the admission config.
pub(crate) const CONFIG_SCHEMA: &str = "selvedge/admission-config/v1";
/// Schema of the admission record.
pub(crate) const RECORD_SCHEMA: &str = "selvedge/admission-record/v1";

#[derive(Debug, Args)]
pub(crate) struct AdmitArgs {
    /// Admission config JSON declaring the selector and execution slices.
    #[arg(long)]
    pub(crate) config: PathBuf,
    /// Write the admission record here (atomic; no partial file is left behind).
    #[arg(long)]
    pub(crate) out: Option<PathBuf>,
    /// Print the admission record to stdout as JSON.
    #[arg(long)]
    pub(crate) json: bool,
}

/// Declared selection path and budget for one arm.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AdmissionConfig {
    /// Must equal [`CONFIG_SCHEMA`].
    pub(crate) schema: String,
    /// Arm identifier (`A`, `A0`, `B`, `C`, `C2`, `D`, `E`).
    pub(crate) arm: String,
    /// Ceiling and compliance mode.
    pub(crate) budget: Budget,
    /// All learned parameters accessible to the model (informational).
    pub(crate) total_accessible: u64,
    /// Router/controller parameter uses.
    pub(crate) selector_uses: Vec<ParamUse>,
    /// Executed parameter uses for one token.
    pub(crate) execution_uses: Vec<ParamUse>,
    /// Router/controller FLOPs for one token.
    pub(crate) selector_flops: FlopEstimate,
    /// Execution FLOPs for one token.
    pub(crate) execution_flops: FlopEstimate,
    /// Free-form notes recorded verbatim.
    #[serde(default)]
    pub(crate) notes: Vec<String>,
    /// Receipt stamp recorded verbatim when present.
    #[serde(default)]
    pub(crate) stamp: Option<String>,
    /// Git revision recorded verbatim when present.
    #[serde(default)]
    pub(crate) revision: Option<String>,
}

/// Machine-readable admission record.
#[derive(Debug, Serialize)]
pub(crate) struct AdmissionRecord {
    pub(crate) schema: &'static str,
    pub(crate) arm: String,
    pub(crate) admitted: bool,
    pub(crate) config_sha256: String,
    pub(crate) budget: Budget,
    pub(crate) total_accessible: u64,
    pub(crate) mode: String,
    pub(crate) active_unique_params: Option<u64>,
    pub(crate) selector_unique_params: Option<u64>,
    pub(crate) execution_unique_params: Option<u64>,
    pub(crate) selector_elements_read: Option<u64>,
    pub(crate) execution_elements_read: Option<u64>,
    pub(crate) selector_flops: FlopEstimate,
    pub(crate) execution_flops: FlopEstimate,
    pub(crate) total_flops: Option<FlopEstimate>,
    pub(crate) verdict: Option<BudgetVerdict>,
    pub(crate) refusal: Option<String>,
    pub(crate) notes: Vec<String>,
    pub(crate) stamp: Option<String>,
    pub(crate) revision: Option<String>,
}

/// Evaluate a parsed config. Returns the record; `Ok(record)` may still carry
/// `admitted = false` for a refusal, which the caller turns into an exit code.
fn evaluate(config: &AdmissionConfig, config_sha256: String) -> Result<AdmissionRecord, String> {
    if config.schema != CONFIG_SCHEMA {
        return Err(format!(
            "unexpected schema {:?} (expected {CONFIG_SCHEMA})",
            config.schema
        ));
    }
    if config.budget.mode != ComplianceMode::Compliance {
        return Err(
            "admission records are only issued in compliance mode; a teacher reference \
             must not have an admission record"
                .to_owned(),
        );
    }
    let path = SelectionPath::new(
        config.selector_uses.clone(),
        config.execution_uses.clone(),
        config.selector_flops,
        config.execution_flops,
    )
    .map_err(|error| format!("invalid selection path: {error}"))?;

    let ledger = ParameterLedger::begin(config.budget, config.total_accessible)
        .map_err(|error| format!("cannot begin ledger: {error}"))?;

    let base = |admitted: bool,
                active: Option<u64>,
                selector_unique: Option<u64>,
                execution_unique: Option<u64>,
                selector_reads: Option<u64>,
                execution_reads: Option<u64>,
                total_flops: Option<FlopEstimate>,
                verdict: Option<BudgetVerdict>,
                refusal: Option<String>| AdmissionRecord {
        schema: RECORD_SCHEMA,
        arm: config.arm.clone(),
        admitted,
        config_sha256: config_sha256.clone(),
        budget: config.budget,
        total_accessible: config.total_accessible,
        mode: config.budget.mode.to_string(),
        active_unique_params: active,
        selector_unique_params: selector_unique,
        execution_unique_params: execution_unique,
        selector_elements_read: selector_reads,
        execution_elements_read: execution_reads,
        selector_flops: config.selector_flops,
        execution_flops: config.execution_flops,
        total_flops,
        verdict,
        refusal,
        notes: config.notes.clone(),
        stamp: config.stamp.clone(),
        revision: config.revision.clone(),
    };

    match ledger.admit_selection_path(path.clone()) {
        Ok(admitted) => {
            let path = admitted.path();
            let active = path
                .active_unique_params()
                .map_err(|error| format!("active union: {error}"))?;
            let selector_unique = path
                .selector_unique_params()
                .map_err(|error| format!("selector union: {error}"))?;
            let execution_unique = path
                .execution_unique_params()
                .map_err(|error| format!("execution union: {error}"))?;
            let selector_reads = path
                .selector_elements_read()
                .map_err(|error| format!("selector reads: {error}"))?;
            let execution_reads = path
                .execution_elements_read()
                .map_err(|error| format!("execution reads: {error}"))?;
            let total_flops = path
                .total_flops()
                .map_err(|error| format!("flops: {error}"))?;
            Ok(base(
                true,
                Some(active),
                Some(selector_unique),
                Some(execution_unique),
                Some(selector_reads),
                Some(execution_reads),
                Some(total_flops),
                Some(admitted.verdict().clone()),
                None,
            ))
        }
        Err(error @ BudgetError::Exceeded { .. }) => {
            let active = path.active_unique_params().ok();
            Ok(base(
                false,
                active,
                path.selector_unique_params().ok(),
                path.execution_unique_params().ok(),
                path.selector_elements_read().ok(),
                path.execution_elements_read().ok(),
                path.total_flops().ok(),
                None,
                Some(error.to_string()),
            ))
        }
        Err(error) => Err(format!("cannot admit selection path: {error}")),
    }
}

fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Run `selvedge admit`.
///
/// # Errors
///
/// Propagates I/O, parse and ledger failures.
pub(crate) fn run(args: &AdmitArgs) -> CliResult {
    let bytes = std::fs::read(&args.config)?;
    let sha = Sha256Digest::of(&bytes).to_hex();
    let config: AdmissionConfig = serde_json::from_slice(&bytes)?;
    let record =
        evaluate(&config, sha).map_err(|reason| -> Box<dyn std::error::Error> { reason.into() })?;
    let rendered = serde_json::to_string_pretty(&record)?;
    if let Some(out) = &args.out {
        write_atomic(out, format!("{rendered}\n").as_bytes())?;
    }
    if args.json || args.out.is_none() {
        println!("{rendered}");
    }
    if record.admitted {
        Ok(ExitCode::SUCCESS)
    } else {
        Ok(ExitCode::from(3))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use selvedge_core::{FlopEstimate, ParamIdentity, ParamTensorId, StorageTier};

    use super::*;

    fn use_of(key: &str, start: u64, end: u64) -> ParamUse {
        let tensor = ParamTensorId::new(key).unwrap();
        let identity = ParamIdentity::new(tensor, start..end).unwrap();
        ParamUse::once(identity, (end - start) * 2, StorageTier::Gpu).unwrap()
    }

    fn config(limit: u64, execution: Vec<ParamUse>) -> AdmissionConfig {
        AdmissionConfig {
            schema: CONFIG_SCHEMA.to_owned(),
            arm: "test".to_owned(),
            budget: Budget::new(limit, ComplianceMode::Compliance).unwrap(),
            total_accessible: 2 * limit,
            selector_uses: vec![use_of("router", 0, 1_000)],
            execution_uses: execution,
            selector_flops: FlopEstimate::matmul(1, 1_000, 64).unwrap(),
            execution_flops: FlopEstimate::matmul(1, 1, 1).unwrap(),
            notes: Vec::new(),
            stamp: None,
            revision: None,
        }
    }

    #[test]
    fn within_budget_is_admitted_and_counts_the_union() {
        let config = config(10_000, vec![use_of("ffn", 0, 8_000)]);
        let record = evaluate(&config, "sha".to_owned()).unwrap();
        assert!(record.admitted);
        assert_eq!(record.active_unique_params, Some(9_000));
        assert_eq!(record.selector_unique_params, Some(1_000));
        assert_eq!(record.execution_unique_params, Some(8_000));
        assert!(record.verdict.unwrap().is_within());
    }

    #[test]
    fn over_budget_is_refused_with_no_admitted_path() {
        let config = config(10_000, vec![use_of("ffn", 0, 9_500)]);
        let record = evaluate(&config, "sha".to_owned()).unwrap();
        assert!(!record.admitted);
        assert!(record.verdict.is_none());
        assert_eq!(record.active_unique_params, Some(10_500));
        assert!(record.refusal.is_some());
    }

    #[test]
    fn teacher_mode_has_no_admission_record() {
        let mut config = config(10_000, vec![use_of("ffn", 0, 8_000)]);
        config.budget = Budget::new(1_000, ComplianceMode::ResearchNoncompliantTeacher).unwrap();
        assert!(evaluate(&config, "sha".to_owned()).is_err());
    }

    #[test]
    fn unknown_field_is_rejected() {
        let json = r#"{"schema":"selvedge/admission-config/v1","arm":"B",
            "budget":{"active_unique_limit":1000000000,"mode":"compliance"},
            "total_accessible":1,"selector_uses":[],"execution_uses":[],
            "selector_flops":{"matmul_flops":0,"other_flops":0,"controller_flops":0},
            "execution_flops":{"matmul_flops":0,"other_flops":0,"controller_flops":0},
            "surprise":true}"#;
        assert!(serde_json::from_str::<AdmissionConfig>(json).is_err());
    }
}
