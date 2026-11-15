//! Executable Court C scenarios (budget accounting).
//!
//! These functions are the shared implementation used by both the integration
//! tests (`crates/selvedge-core/tests/court_c.rs`) and the CLI verifier
//! (`selvedge court c`). Each scenario returns its computed counters as a
//! detail string so a receipt can show the numbers rather than a bare "pass".

use serde::Serialize;

use crate::error::{CoreError, Result};
use crate::hash::Sha256Digest;
use crate::ids::ShardId;
use crate::ledger::{
    AuxiliaryKind, Budget, BudgetError, BudgetVerdict, ComplianceMode, DecisionPlan, FlopEstimate,
    ParamUse, ParameterLedger, QuantDequantWork, ResolvedPlan, TokenAccounting,
};
use crate::residency::StorageTier;
use crate::selection::{SelectionDecision, SelectionProvenance};
use crate::tensor::{ParamIdentity, ParamTensorId};

/// Result of one named verification scenario.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ScenarioOutcome {
    /// Stable scenario name.
    pub name: &'static str,
    /// Whether every assertion in the scenario held.
    pub passed: bool,
    /// Computed counters or the failure reason.
    pub detail: String,
}

/// Result of a complete court run.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CourtReport {
    /// Court identifier.
    pub court: &'static str,
    /// Whether every scenario passed.
    pub passed: bool,
    /// Individual scenarios.
    pub scenarios: Vec<ScenarioOutcome>,
}

/// Run every Court C scenario.
#[must_use]
pub fn run_budget_court() -> CourtReport {
    let scenarios = vec![
        tied_embeddings_counted_once(),
        overlapping_slices_deduplicated(),
        recurrent_reuse_unique_vs_reads(),
        adapters_memory_router_verifier_counted(),
        over_budget_rejected_in_compliance(),
        over_budget_allowed_and_labelled_in_research_mode(),
        boundary_exactly_at_limit_is_within(),
        boundary_one_over_limit_is_rejected(),
        quantized_metadata_excluded_from_learned_count(),
        plan_fallback_within_budget(),
        fallback_that_remains_over_budget_is_rejected(),
        token_accounting_reports_every_required_field(),
    ];
    let passed = scenarios.iter().all(|scenario| scenario.passed);
    CourtReport {
        court: "c",
        passed,
        scenarios,
    }
}

fn scenario(name: &'static str, body: impl FnOnce() -> Result<String>) -> ScenarioOutcome {
    match body() {
        Ok(detail) => ScenarioOutcome {
            name,
            passed: true,
            detail,
        },
        Err(error) => ScenarioOutcome {
            name,
            passed: false,
            detail: error.to_string(),
        },
    }
}

fn require(condition: bool, reason: impl Into<String>) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(CoreError::invalid("court_c", reason))
    }
}

fn identity(key: &str, start: u64, end: u64) -> Result<ParamIdentity> {
    ParamIdentity::new(ParamTensorId::new(key)?, start..end)
}

fn use_once(key: &str, start: u64, end: u64, tier: StorageTier) -> Result<ParamUse> {
    let identity = identity(key, start, end)?;
    let bytes = identity
        .len()
        .checked_mul(4)
        .ok_or_else(|| CoreError::overflow("court fixture bytes"))?;
    ParamUse::once(identity, bytes, tier)
}

fn plan_for(label: &str, uses: Vec<ParamUse>, flops: FlopEstimate) -> Result<DecisionPlan> {
    let shard = ShardId::from_digest(Sha256Digest::of(label.as_bytes()));
    let provenance = SelectionProvenance::approximate(label, Vec::new())?;
    let decision = SelectionDecision::new(vec![shard], provenance)?;
    let mut plan = DecisionPlan::new(decision);
    for use_value in uses {
        plan.push_use(use_value)?;
    }
    plan.flops = flops;
    Ok(plan)
}

fn compliance_budget(limit: u64) -> Result<Budget> {
    Budget::new(limit, ComplianceMode::Compliance)
}

/// Tied embeddings and the LM head share one underlying parameter tensor and
/// therefore count once, while both reads are still reported.
pub fn tied_embeddings_counted_once() -> ScenarioOutcome {
    scenario("tied_embeddings_counted_once", || {
        let vocabulary = 1_000_u64;
        let key = "model.embed_tokens.weight";
        let embed = use_once(key, 0, vocabulary, StorageTier::Nvme)?;
        let head = use_once(key, 0, vocabulary, StorageTier::Nvme)?;
        let plan = plan_for(
            "tied",
            vec![embed.clone(), head.clone()],
            FlopEstimate::default(),
        )?;
        let ledger = ParameterLedger::begin(compliance_budget(1_000_000_000)?, vocabulary * 8)?;
        let verdict = ledger
            .check_decision(&plan)
            .map_err(|err| CoreError::invalid("court_c", err.to_string()))?;
        let plan_active = match verdict {
            BudgetVerdict::Within {
                active_unique_params,
                ..
            } => active_unique_params,
            other => {
                return Err(CoreError::invalid(
                    "court_c",
                    format!("unexpected verdict {other:?}"),
                ));
            }
        };
        let mut ledger = ParameterLedger::begin(compliance_budget(1_000_000_000)?, vocabulary * 8)?;
        ledger.record_use(embed)?;
        ledger.record_use(head)?;
        let snapshot = ledger.snapshot()?;
        require(
            snapshot.active_unique_params == vocabulary,
            format!(
                "tied embeddings counted {} unique elements, expected {vocabulary}",
                snapshot.active_unique_params
            ),
        )?;
        require(plan_active == vocabulary, "plan union did not deduplicate")?;
        require(
            snapshot.parsed_parameter_reads == vocabulary * 2,
            format!(
                "read count {} does not count both reads",
                snapshot.parsed_parameter_reads
            ),
        )?;
        require(
            snapshot
                .learned_bytes_by_tier
                .get(&StorageTier::Nvme)
                .copied()
                .unwrap_or(0)
                == vocabulary * 8,
            "tier byte count does not count both reads",
        )?;
        Ok(format!(
            "active_unique={} reads={} bytes_nvme={} plan_active={}",
            snapshot.active_unique_params,
            snapshot.parsed_parameter_reads,
            snapshot
                .learned_bytes_by_tier
                .get(&StorageTier::Nvme)
                .copied()
                .unwrap_or(0),
            plan_active
        ))
    })
}

/// Overlapping and adjacent fragments of the same tensor are unioned.
pub fn overlapping_slices_deduplicated() -> ScenarioOutcome {
    scenario("overlapping_slices_deduplicated", || {
        let first = use_once("mlp.gate", 0, 100, StorageTier::Nvme)?;
        let second = use_once("mlp.gate", 50, 150, StorageTier::Nvme)?;
        let third = use_once("mlp.gate", 100, 200, StorageTier::Nvme)?;
        let mut ledger = ParameterLedger::begin(compliance_budget(1_000_000_000)?, 1_000)?;
        for use_value in [first, second, third] {
            ledger.record_use(use_value)?;
        }
        let snapshot = ledger.snapshot()?;
        require(
            snapshot.active_unique_params == 200,
            format!(
                "union of 0..100, 50..150, 100..200 is {} not 200",
                snapshot.active_unique_params
            ),
        )?;
        require(
            snapshot.parsed_parameter_reads == 300,
            format!("reads {} not 300", snapshot.parsed_parameter_reads),
        )?;
        Ok(format!(
            "active_unique={} reads={}",
            snapshot.active_unique_params, snapshot.parsed_parameter_reads
        ))
    })
}

/// Recurrent reuse counts the weights once and every repeated read separately;
/// FLOPs accumulate per repeated operation.
pub fn recurrent_reuse_unique_vs_reads() -> ScenarioOutcome {
    scenario("recurrent_reuse_unique_vs_reads", || {
        let repeats = 8_u64;
        let elements = 400_u64;
        let mut ledger = ParameterLedger::begin(compliance_budget(1_000_000_000)?, 10_000)?;
        for _ in 0..repeats {
            ledger.record_use(use_once(
                "recurrent_block",
                0,
                elements,
                StorageTier::HostRam,
            )?)?;
            ledger.record_flops(&FlopEstimate::matmul(1, elements, 1)?)?;
        }
        let snapshot = ledger.snapshot()?;
        require(
            snapshot.active_unique_params == elements,
            format!(
                "recurrent unique count {} not {elements}",
                snapshot.active_unique_params
            ),
        )?;
        require(
            snapshot.parsed_parameter_reads == elements * repeats,
            format!(
                "recurrent reads {} not {}",
                snapshot.parsed_parameter_reads,
                elements * repeats
            ),
        )?;
        require(
            snapshot.flops == 2 * repeats * elements,
            format!("recurrent flops {} unexpected", snapshot.flops),
        )?;
        Ok(format!(
            "active_unique={} reads={} flops={}",
            snapshot.active_unique_params, snapshot.parsed_parameter_reads, snapshot.flops
        ))
    })
}

/// Adapters, learned memory, router and verifier weights all count as learned.
pub fn adapters_memory_router_verifier_counted() -> ScenarioOutcome {
    scenario("adapters_memory_router_verifier_counted", || {
        let mut ledger = ParameterLedger::begin(compliance_budget(1_000_000_000)?, 1_000_000)?;
        let parts = [
            ("adapter.a", 10_u64),
            ("adapter.b", 20),
            ("memory.key", 30),
            ("memory.value", 40),
            ("router", 5),
            ("verifier", 7),
        ];
        for (key, size) in parts {
            ledger.record_use(use_once(key, 0, size, StorageTier::Gpu)?)?;
        }
        let snapshot = ledger.snapshot()?;
        require(
            snapshot.active_unique_params == 112,
            format!(
                "adapter/memory/router/verifier total {} not 112",
                snapshot.active_unique_params
            ),
        )?;
        require(
            snapshot
                .learned_bytes_by_tier
                .get(&StorageTier::Gpu)
                .copied()
                .unwrap_or(0)
                == 448,
            "gpu byte total incorrect",
        )?;
        Ok(format!(
            "active_unique={} gpu_bytes={}",
            snapshot.active_unique_params,
            snapshot
                .learned_bytes_by_tier
                .get(&StorageTier::Gpu)
                .copied()
                .unwrap_or(0)
        ))
    })
}

/// A compliant over-budget decision is an error, not a boolean.
pub fn over_budget_rejected_in_compliance() -> ScenarioOutcome {
    scenario("over_budget_rejected_in_compliance", || {
        let limit = 1_000_000_000_u64;
        let plan = plan_for(
            "over-budget",
            vec![use_once("mlp.gate", 0, limit + 1, StorageTier::Nvme)?],
            FlopEstimate::default(),
        )?;
        let ledger = ParameterLedger::begin(compliance_budget(limit)?, limit * 4)?;
        match ledger.check_decision(&plan) {
            Err(BudgetError::Exceeded {
                active_unique_params,
                limit: reported_limit,
                mode,
            }) => {
                require(active_unique_params == limit + 1, "wrong reported count")?;
                require(reported_limit == limit, "wrong reported limit")?;
                require(mode == ComplianceMode::Compliance, "wrong mode")?;
                Ok(format!(
                    "rejected active_unique={active_unique_params} limit={reported_limit} mode={mode}"
                ))
            }
            Ok(verdict) => Err(CoreError::invalid(
                "court_c",
                format!("over-budget plan was allowed: {verdict:?}"),
            )),
            Err(other) => Err(CoreError::invalid(
                "court_c",
                format!("unexpected error: {other}"),
            )),
        }
    })
}

/// The same plan is allowed in teacher mode only as a labelled verdict.
pub fn over_budget_allowed_and_labelled_in_research_mode() -> ScenarioOutcome {
    scenario("over_budget_allowed_and_labelled_in_research_mode", || {
        let limit = 1_000_000_000_u64;
        let plan = plan_for(
            "teacher",
            vec![use_once("mlp.gate", 0, limit + 1, StorageTier::Nvme)?],
            FlopEstimate::default(),
        )?;
        let ledger = ParameterLedger::begin(
            Budget::new(limit, ComplianceMode::ResearchNoncompliantTeacher)?,
            limit * 4,
        )?;
        match ledger.check_decision(&plan) {
            Ok(BudgetVerdict::NoncompliantTeacher {
                active_unique_params,
                limit: reported_limit,
                label,
            }) => {
                require(active_unique_params == limit + 1, "wrong teacher count")?;
                require(reported_limit == limit, "wrong teacher limit")?;
                require(
                    label == "research_noncompliant_teacher",
                    "teacher label missing",
                )?;
                Ok(format!(
                    "allowed_label={label} active_unique={active_unique_params} limit={reported_limit}"
                ))
            }
            Ok(other) => Err(CoreError::invalid(
                "court_c",
                format!("teacher plan produced unexpected verdict {other:?}"),
            )),
            Err(error) => Err(CoreError::invalid(
                "court_c",
                format!("teacher plan was rejected: {error}"),
            )),
        }
    })
}

/// Exactly at the ceiling is inside the budget.
pub fn boundary_exactly_at_limit_is_within() -> ScenarioOutcome {
    scenario("boundary_exactly_at_limit_is_within", || {
        let limit = 1_000_000_000_u64;
        let plan = plan_for(
            "at-limit",
            vec![use_once("mlp.gate", 0, limit, StorageTier::Nvme)?],
            FlopEstimate::default(),
        )?;
        let ledger = ParameterLedger::begin(compliance_budget(limit)?, limit * 2)?;
        match ledger.check_decision(&plan) {
            Ok(BudgetVerdict::Within {
                active_unique_params,
                limit: reported_limit,
            }) => {
                require(active_unique_params == limit, "boundary count changed")?;
                require(reported_limit == limit, "boundary limit changed")?;
                Ok(format!(
                    "within active_unique={active_unique_params} limit={reported_limit}"
                ))
            }
            other => Err(CoreError::invalid(
                "court_c",
                format!("boundary plan did not pass: {other:?}"),
            )),
        }
    })
}

/// One element over the ceiling is rejected.
pub fn boundary_one_over_limit_is_rejected() -> ScenarioOutcome {
    scenario("boundary_one_over_limit_is_rejected", || {
        let limit = 1_000_000_000_u64;
        let plan = plan_for(
            "over-limit",
            vec![use_once("mlp.gate", 0, limit + 1, StorageTier::Nvme)?],
            FlopEstimate::default(),
        )?;
        let ledger = ParameterLedger::begin(compliance_budget(limit)?, limit * 2)?;
        match ledger.check_decision(&plan) {
            Err(BudgetError::Exceeded {
                active_unique_params,
                ..
            }) => {
                require(active_unique_params == limit + 1, "wrong over-limit count")?;
                Ok(format!(
                    "rejected active_unique={active_unique_params} limit={limit}"
                ))
            }
            other => Err(CoreError::invalid(
                "court_c",
                format!("one-over-limit plan was not rejected: {other:?}"),
            )),
        }
    })
}

/// Scales and zero points are non-learned auxiliary bytes, reported separately.
pub fn quantized_metadata_excluded_from_learned_count() -> ScenarioOutcome {
    scenario("quantized_metadata_excluded_from_learned_count", || {
        let mut ledger = ParameterLedger::begin(compliance_budget(1_000_000_000)?, 10_000)?;
        ledger.record_use(use_once("mlp.gate", 0, 1_000, StorageTier::Nvme)?)?;
        ledger.record_auxiliary(AuxiliaryKind::Scales, 40)?;
        ledger.record_auxiliary(AuxiliaryKind::ZeroPoints, 8)?;
        ledger.record_quant_work(&QuantDequantWork {
            dequantized_elements: 1_000,
            requantized_elements: 0,
        })?;
        let snapshot = ledger.snapshot()?;
        require(
            snapshot.active_unique_params == 1_000,
            format!(
                "learned count includes auxiliary bytes: {}",
                snapshot.active_unique_params
            ),
        )?;
        let auxiliary: u64 = snapshot.auxiliary_bytes.values().sum();
        require(
            auxiliary == 48,
            format!("auxiliary bytes {auxiliary} not 48"),
        )?;
        require(
            snapshot.quant_dequant_work.dequantized_elements == 1_000,
            "dequant work not reported",
        )?;
        Ok(format!(
            "active_unique={} auxiliary_bytes={auxiliary} dequantized={}",
            snapshot.active_unique_params, snapshot.quant_dequant_work.dequantized_elements
        ))
    })
}

/// A fallback hook can substitute an in-budget plan before execution.
pub fn plan_fallback_within_budget() -> ScenarioOutcome {
    scenario("plan_fallback_within_budget", || {
        let limit = 100_u64;
        let original = plan_for(
            "fallback",
            vec![use_once("mlp.gate", 0, 200, StorageTier::Nvme)?],
            FlopEstimate::default(),
        )?;
        let ledger = ParameterLedger::begin(compliance_budget(limit)?, 1_000)?;
        let reduced = {
            let mut plan = original.clone();
            plan.uses = vec![use_once("mlp.gate", 0, limit, StorageTier::Nvme)?];
            plan
        };
        let fallback = |_: &DecisionPlan| Some(reduced.clone());
        let resolved: ResolvedPlan = ledger
            .check_decision_with_fallback(&original, &fallback)
            .map_err(|err| CoreError::invalid("court_c", err.to_string()))?;
        require(resolved.fallback_applied, "fallback was not applied")?;
        require(
            resolved
                .plan
                .active_unique_params()
                .map_err(|err| CoreError::invalid("court_c", err.to_string()))?
                == limit,
            "resolved plan is not the fallback",
        )?;
        require(
            resolved.verdict.is_within(),
            "resolved verdict is not within",
        )?;
        Ok(format!(
            "fallback_applied={} active_unique={} within={}",
            resolved.fallback_applied,
            resolved
                .plan
                .active_unique_params()
                .map_err(|err| { CoreError::invalid("court_c", err.to_string()) })?,
            resolved.verdict.is_within()
        ))
    })
}

/// A fallback that is still over budget cannot be executed.
pub fn fallback_that_remains_over_budget_is_rejected() -> ScenarioOutcome {
    scenario("fallback_that_remains_over_budget_is_rejected", || {
        let limit = 100_u64;
        let original = plan_for(
            "fallback-over",
            vec![use_once("mlp.gate", 0, 200, StorageTier::Nvme)?],
            FlopEstimate::default(),
        )?;
        let ledger = ParameterLedger::begin(compliance_budget(limit)?, 1_000)?;
        let still_over = {
            let mut plan = original.clone();
            plan.uses = vec![use_once("mlp.gate", 0, 150, StorageTier::Nvme)?];
            plan
        };
        let fallback = |_: &DecisionPlan| Some(still_over.clone());
        match ledger.check_decision_with_fallback(&original, &fallback) {
            Err(BudgetError::FallbackExceeded {
                active_unique_params,
                limit: reported_limit,
            }) => {
                require(active_unique_params == 150, "wrong fallback count")?;
                require(reported_limit == limit, "wrong fallback limit")?;
                Ok(format!(
                    "rejected active_unique={active_unique_params} limit={reported_limit}"
                ))
            }
            other => Err(CoreError::invalid(
                "court_c",
                format!("over-budget fallback was not rejected: {other:?}"),
            )),
        }
    })
}

/// The snapshot exposes every field the mandate requires.
pub fn token_accounting_reports_every_required_field() -> ScenarioOutcome {
    scenario("token_accounting_reports_every_required_field", || {
        let accessible = 5_000_000_u64;
        let mut ledger = ParameterLedger::begin(compliance_budget(1_000_000_000)?, accessible)?;
        ledger.record_use(use_once("attn.q", 0, 100, StorageTier::Nvme)?)?;
        ledger.record_use(use_once("attn.k", 0, 100, StorageTier::Gpu)?)?;
        ledger.record_resident(identity("attn.q", 0, 100)?);
        ledger.record_resident(identity("attn.k", 0, 100)?);
        ledger.record_resident(identity("attn.k", 50, 100)?);
        ledger.record_flops(&FlopEstimate {
            matmul_flops: 1_000,
            other_flops: 10,
            controller_flops: 3,
        })?;
        ledger.record_auxiliary(AuxiliaryKind::Scales, 4)?;
        ledger.record_quant_work(&QuantDequantWork {
            dequantized_elements: 5,
            requantized_elements: 0,
        })?;
        let snapshot: TokenAccounting = ledger.snapshot()?;
        require(snapshot.active_unique_params == 200, "active unique")?;
        require(snapshot.resident_unique_params == 200, "resident union")?;
        require(snapshot.total_accessible_params == accessible, "accessible")?;
        require(snapshot.parsed_parameter_reads == 200, "reads")?;
        require(snapshot.flops == 1_010, "model flops")?;
        require(snapshot.controller_flops == 3, "controller flops")?;
        require(snapshot.within_budget, "within budget")?;
        require(snapshot.mode == ComplianceMode::Compliance, "mode")?;
        require(snapshot.learned_bytes_by_tier.len() == 2, "tier bytes")?;
        require(snapshot.auxiliary_bytes.len() == 1, "aux bytes")?;
        Ok(format!(
            "active_unique={} resident_unique={} accessible={} reads={} flops={} controller_flops={} tiers={} auxiliary_kinds={}",
            snapshot.active_unique_params,
            snapshot.resident_unique_params,
            snapshot.total_accessible_params,
            snapshot.parsed_parameter_reads,
            snapshot.flops,
            snapshot.controller_flops,
            snapshot.learned_bytes_by_tier.len(),
            snapshot.auxiliary_bytes.len()
        ))
    })
}
