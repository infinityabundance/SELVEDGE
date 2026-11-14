//! Court A: executable reconstruction matrix.
//!
//! Each scenario converts a deterministic synthetic SwiGLU checkpoint, publishes
//! a real bundle, re-reads it from disk through the format validator, and
//! compares the grouped FFN against independent dense f32 and f64 reference
//! paths. No scenario claims bitwise equality.

use std::path::PathBuf;

use selvedge_core::testutil::TempDir;
use selvedge_core::{ModelDigest, Sha256Builder, Sha256Digest, StorageOrder};
use selvedge_format::{BundleFile, ValidationOptions, VerifyMode, validate_bundle};
use serde::Serialize;

use crate::convert::{ConversionOptions, convert, plan_conversion};
use crate::error::{ConvertError, Result};
use crate::ffn::{GroupSpec, ffn_grouped_f32};
use crate::synthetic::{
    DeterministicRng, SYNTHETIC_SEED, SyntheticDType, SyntheticModel, swiglu_model,
};
use crate::verify::{Tolerance, VerificationOptions, verify_bundle};

/// One named scenario.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ScenarioResult {
    /// Scenario name.
    pub name: String,
    /// Whether it passed.
    pub passed: bool,
    /// Human-readable detail, including measured values where relevant.
    pub detail: String,
}

/// A measured row for the report and the generated-number receipts.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MeasurementRow {
    /// Scenario name.
    pub scenario: String,
    /// Source dtype tested.
    pub source_dtype: String,
    /// Grouping description.
    pub grouping: String,
    /// Maximum f32 absolute error.
    pub max_abs_f32: f64,
    /// Maximum f32 relative error.
    pub max_rel_f32: f64,
    /// Maximum f64 absolute error.
    pub max_abs_f64: f64,
    /// Maximum f64 relative error.
    pub max_rel_f64: f64,
    /// Number of compared f32 components.
    pub count: u64,
    /// Logical shards in the converted bundle.
    pub shard_count: u64,
    /// Payload bytes in the converted bundle.
    pub payload_bytes: u64,
    /// Peak in-flight bytes used by the conversion.
    pub peak_inflight_bytes: u64,
    /// Absolute tolerance.
    pub tolerance_abs: f64,
    /// Relative tolerance.
    pub tolerance_rel: f64,
}

/// Stable numbers selected for the generated-number manifest.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CourtAReceiptNumbers {
    /// All-groups f32 maximum absolute error.
    pub all_groups_max_abs_f32: Option<f64>,
    /// All-groups f32 maximum relative error.
    pub all_groups_max_rel_f32: Option<f64>,
    /// All-groups f64 maximum absolute error.
    pub all_groups_max_abs_f64: Option<f64>,
    /// All-groups f64 maximum relative error.
    pub all_groups_max_rel_f64: Option<f64>,
    /// f16-source f32 maximum absolute error.
    pub f16_max_abs_f32: Option<f64>,
    /// bf16-source f32 maximum absolute error.
    pub bf16_max_abs_f32: Option<f64>,
    /// Absolute tolerance.
    pub tolerance_abs: f64,
    /// Relative tolerance.
    pub tolerance_rel: f64,
    /// Number of scenarios.
    pub scenario_count: u64,
    /// Compared f32 components in the all-groups case.
    pub compared_components: Option<u64>,
    /// Shards in the all-groups bundle.
    pub shard_count: Option<u64>,
    /// Payload bytes in the all-groups bundle.
    pub payload_bytes: Option<u64>,
    /// Peak in-flight bytes in the all-groups conversion.
    pub peak_inflight_bytes: Option<u64>,
}

/// The complete Court A report.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CourtAReport {
    /// Report schema.
    pub schema: String,
    /// Court name.
    pub court: String,
    /// Whether every scenario passed.
    pub passed: bool,
    /// Seed used for the base model.
    pub seed: u64,
    /// Absolute tolerance.
    pub tolerance_abs: f64,
    /// Relative tolerance.
    pub tolerance_rel: f64,
    /// Optional receipt stamp supplied by the caller.
    pub stamp: Option<String>,
    /// Optional git revision supplied by the caller.
    pub revision: Option<String>,
    /// Producing tool.
    pub generated_by: String,
    /// Scenario results.
    pub scenarios: Vec<ScenarioResult>,
    /// Measured reconstruction rows.
    pub measurements: Vec<MeasurementRow>,
    /// Stable numbers selected for `docs/perf-numbers.json`.
    pub receipt_numbers: CourtAReceiptNumbers,
    /// Number of scenarios.
    pub scenario_count: u64,
}

#[derive(Clone, Debug)]
struct CaseOutcome {
    max_abs_f32: f64,
    max_rel_f32: f64,
    max_abs_f64: f64,
    max_rel_f64: f64,
    count: u64,
    source_dtype: String,
    shard_count: u64,
    payload_bytes: u64,
    peak_inflight_bytes: u64,
}

/// Run the full Court A matrix with the default seed.
///
/// # Errors
///
/// Propagates any conversion or verification failure.
pub fn run_court_a(seed: u64) -> Result<CourtAReport> {
    run_court_a_with_provenance(seed, None, None)
}

/// Run the full Court A matrix with optional receipt provenance.
///
/// # Errors
///
/// Propagates any conversion or verification failure.
pub fn run_court_a_with_provenance(
    seed: u64,
    stamp: Option<String>,
    revision: Option<String>,
) -> Result<CourtAReport> {
    let tolerance = Tolerance::default();
    let mut scenarios = Vec::new();
    let mut measurements = Vec::new();

    let base = run_case(
        CaseRequest {
            name: "all_groups_equal_original",
            hidden: 8,
            intermediate: 10,
            spec: GroupSpec::ByCount(4),
            dtype: SyntheticDType::F32,
            order: StorageOrder::RowMajor,
            seed: SYNTHETIC_SEED ^ 1,
            batch_sizes: vec![1, 4, 17],
            max_inflight_bytes: 8 * 1024 * 1024,
        },
        tolerance,
    );
    push_case(
        &mut scenarios,
        &mut measurements,
        base,
        "all_groups_equal_original",
        "count 4",
    );

    let permutation = scenario("group_permutation_invariance", || {
        let model = swiglu_model(8, 10, seed ^ 2)?;
        let mut source = model.to_source(SyntheticDType::F32, StorageOrder::RowMajor)?;
        let first = GroupSpec::Explicit(vec![(0..4).collect(), (4..8).collect(), vec![8, 9]]);
        let second = GroupSpec::Explicit(vec![vec![8, 9], (0..4).collect(), (4..8).collect()]);
        let input = DeterministicRng::new(seed ^ 3).fill(2 * 8);
        let mut groups_first = Vec::new();
        let mut groups_second = Vec::new();
        for spec in [&first, &second] {
            let groups = crate::ffn::resolve_groups(spec, 10, true)?;
            let channel_groups: Vec<Vec<u64>> =
                groups.iter().map(|group| group.indices.clone()).collect();
            let mut budget = crate::tensor_source::MemoryBudget::new(8 * 1024 * 1024)?;
            let built = crate::verify::source_groups(
                &mut source,
                model.map().ok_or(ConvertError::NoFfnTensors)?,
                crate::ffn::FfnDims {
                    hidden: 8,
                    intermediate: 10,
                },
                &channel_groups,
                &mut budget,
            )?;
            if groups_first.is_empty() {
                groups_first = built;
            } else {
                groups_second = built;
            }
        }
        let left = ffn_grouped_f32(8, &groups_first, 2, &input)?;
        let right = ffn_grouped_f32(8, &groups_second, 2, &input)?;
        let max_abs = left
            .iter()
            .zip(right.iter())
            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
            .fold(0.0_f64, f64::max);
        if max_abs > tolerance.abs + tolerance.rel * 10.0 {
            return Err(ConvertError::ToleranceExceeded {
                max_abs,
                max_rel: max_abs,
                abs_tol: tolerance.abs,
                rel_tol: tolerance.rel,
            });
        }
        Ok(format!(
            "two group orders differ by max abs {max_abs:.3e} (not bitwise required)"
        ))
    });
    scenarios.push(permutation);

    let single = run_case(
        CaseRequest {
            name: "single_group_full_span",
            hidden: 6,
            intermediate: 9,
            spec: GroupSpec::ByCount(1),
            dtype: SyntheticDType::F32,
            order: StorageOrder::RowMajor,
            seed: seed ^ 4,
            batch_sizes: vec![1, 4, 17],
            max_inflight_bytes: 8 * 1024 * 1024,
        },
        tolerance,
    );
    push_case(
        &mut scenarios,
        &mut measurements,
        single,
        "single_group_full_span",
        "single group",
    );

    scenarios.push(scenario("empty_subset_is_exact_zero", || {
        let input = DeterministicRng::new(seed ^ 5).fill(3 * 4);
        let empty = ffn_grouped_f32(4, &[], 3, &input)?;
        if empty.iter().any(|value| *value != 0.0) {
            return Err(ConvertError::inconsistent(
                "empty group set did not produce exact zero",
            ));
        }
        Ok(format!(
            "{} components are exactly +0.0; this is a mathematical zero, not a skip proof",
            empty.len()
        ))
    }));

    for size in [1_usize, 2, 3] {
        let intermediate =
            u64::try_from(size * 4).map_err(|_| ConvertError::plan("group size overflow"))?;
        let groups: Vec<Vec<u64>> = (0..intermediate)
            .collect::<Vec<_>>()
            .chunks(size)
            .map(<[u64]>::to_vec)
            .collect();
        let outcome = run_case(
            CaseRequest {
                name: match size {
                    1 => "group_size_one",
                    2 => "group_size_two",
                    _ => "group_size_three",
                },
                hidden: 5,
                intermediate,
                spec: GroupSpec::Explicit(groups),
                dtype: SyntheticDType::F32,
                order: StorageOrder::RowMajor,
                seed: seed ^ (100 + size as u64),
                batch_sizes: vec![1, 4],
                max_inflight_bytes: 8 * 1024 * 1024,
            },
            tolerance,
        );
        push_case(
            &mut scenarios,
            &mut measurements,
            outcome,
            match size {
                1 => "group_size_one",
                2 => "group_size_two",
                _ => "group_size_three",
            },
            &format!("explicit size {size}"),
        );
    }

    let prime = run_case(
        CaseRequest {
            name: "group_size_prime",
            hidden: 5,
            intermediate: 14,
            spec: GroupSpec::Explicit(vec![(0..7).collect(), (7..14).collect()]),
            dtype: SyntheticDType::F32,
            order: StorageOrder::RowMajor,
            seed: seed ^ 6,
            batch_sizes: vec![1, 4],
            max_inflight_bytes: 8 * 1024 * 1024,
        },
        tolerance,
    );
    push_case(
        &mut scenarios,
        &mut measurements,
        prime,
        "group_size_prime",
        "prime size 7",
    );

    let uneven = run_case(
        CaseRequest {
            name: "non_divisible_grouping",
            hidden: 5,
            intermediate: 10,
            spec: GroupSpec::Explicit(vec![(0..4).collect(), (4..8).collect(), (8..10).collect()]),
            dtype: SyntheticDType::F32,
            order: StorageOrder::RowMajor,
            seed: seed ^ 7,
            batch_sizes: vec![1, 4],
            max_inflight_bytes: 8 * 1024 * 1024,
        },
        tolerance,
    );
    push_case(
        &mut scenarios,
        &mut measurements,
        uneven,
        "non_divisible_grouping",
        "4/4/2",
    );

    let one_intermediate = run_case(
        CaseRequest {
            name: "intermediate_one",
            hidden: 5,
            intermediate: 1,
            spec: GroupSpec::ByCount(1),
            dtype: SyntheticDType::F32,
            order: StorageOrder::RowMajor,
            seed: seed ^ 8,
            batch_sizes: vec![1, 4],
            max_inflight_bytes: 8 * 1024 * 1024,
        },
        tolerance,
    );
    push_case(
        &mut scenarios,
        &mut measurements,
        one_intermediate,
        "intermediate_one",
        "intermediate=1",
    );

    let one_hidden = run_case(
        CaseRequest {
            name: "hidden_one",
            hidden: 1,
            intermediate: 4,
            spec: GroupSpec::ByCount(2),
            dtype: SyntheticDType::F32,
            order: StorageOrder::RowMajor,
            seed: seed ^ 9,
            batch_sizes: vec![1, 4],
            max_inflight_bytes: 8 * 1024 * 1024,
        },
        tolerance,
    );
    push_case(
        &mut scenarios,
        &mut measurements,
        one_hidden,
        "hidden_one",
        "hidden=1",
    );

    let shapes = run_case(
        CaseRequest {
            name: "input_shapes",
            hidden: 5,
            intermediate: 7,
            spec: GroupSpec::ByCount(3),
            dtype: SyntheticDType::F32,
            order: StorageOrder::RowMajor,
            seed: seed ^ 10,
            // [1,1] -> 1, [4,17] -> 68, [2,3,5] -> 30.
            batch_sizes: vec![1, 68, 30],
            max_inflight_bytes: 8 * 1024 * 1024,
        },
        tolerance,
    );
    push_case(
        &mut scenarios,
        &mut measurements,
        shapes,
        "input_shapes",
        "batch [1,1],[4,17],[2,3,5]",
    );

    let column = run_case(
        CaseRequest {
            name: "column_major_source",
            hidden: 7,
            intermediate: 9,
            spec: GroupSpec::ByCount(3),
            dtype: SyntheticDType::F32,
            order: StorageOrder::ColumnMajor,
            seed: seed ^ 11,
            batch_sizes: vec![1, 4],
            max_inflight_bytes: 8 * 1024 * 1024,
        },
        tolerance,
    );
    push_case(
        &mut scenarios,
        &mut measurements,
        column,
        "column_major_source",
        "column-major source",
    );

    let transposed = run_case(
        CaseRequest {
            name: "transposed_source",
            hidden: 7,
            intermediate: 9,
            spec: GroupSpec::ByCount(3),
            dtype: SyntheticDType::F32,
            order: StorageOrder::Transposed { axes: vec![1, 0] },
            seed: seed ^ 12,
            batch_sizes: vec![1, 4],
            max_inflight_bytes: 8 * 1024 * 1024,
        },
        tolerance,
    );
    push_case(
        &mut scenarios,
        &mut measurements,
        transposed,
        "transposed_source",
        "transposed source",
    );

    let f16 = run_case(
        CaseRequest {
            name: "f16_source",
            hidden: 6,
            intermediate: 8,
            spec: GroupSpec::ByCount(2),
            dtype: SyntheticDType::F16,
            order: StorageOrder::RowMajor,
            seed: seed ^ 13,
            batch_sizes: vec![1, 4],
            max_inflight_bytes: 8 * 1024 * 1024,
        },
        tolerance,
    );
    push_case(
        &mut scenarios,
        &mut measurements,
        f16,
        "f16_source",
        "f16 source",
    );

    let bf16 = run_case(
        CaseRequest {
            name: "bf16_source",
            hidden: 6,
            intermediate: 8,
            spec: GroupSpec::ByCount(2),
            dtype: SyntheticDType::BF16,
            order: StorageOrder::RowMajor,
            seed: seed ^ 14,
            batch_sizes: vec![1, 4],
            max_inflight_bytes: 8 * 1024 * 1024,
        },
        tolerance,
    );
    push_case(
        &mut scenarios,
        &mut measurements,
        bf16,
        "bf16_source",
        "bf16 source",
    );

    scenarios.push(scenario("twenty_random_seeds", || {
        let mut worst = 0.0_f64;
        for index in 0..20_u64 {
            let outcome = run_case(
                CaseRequest {
                    name: "seed",
                    hidden: 6,
                    intermediate: 9,
                    spec: GroupSpec::ByCount(3),
                    dtype: SyntheticDType::F32,
                    order: StorageOrder::RowMajor,
                    seed: seed.wrapping_add(1000 + index),
                    batch_sizes: vec![1, 4],
                    max_inflight_bytes: 8 * 1024 * 1024,
                },
                tolerance,
            )?;
            if outcome.max_abs_f32 > tolerance.abs + tolerance.rel * 10.0 {
                return Err(ConvertError::ToleranceExceeded {
                    max_abs: outcome.max_abs_f32,
                    max_rel: outcome.max_rel_f32,
                    abs_tol: tolerance.abs,
                    rel_tol: tolerance.rel,
                });
            }
            worst = worst.max(outcome.max_abs_f32);
        }
        Ok(format!("20 seeds; worst f32 max abs {worst:.3e}"))
    }));

    scenarios.push(scenario("determinism_same_bytes_and_ids", || {
        let model = swiglu_model(6, 9, seed ^ 15)?;
        let first = convert_case(
            &model,
            SyntheticDType::F32,
            StorageOrder::RowMajor,
            &GroupSpec::ByCount(3),
            tolerance,
            2,
        )?;
        let second = convert_case(
            &model,
            SyntheticDType::F32,
            StorageOrder::RowMajor,
            &GroupSpec::ByCount(3),
            tolerance,
            2,
        )?;
        if first.bundle_bytes != second.bundle_bytes {
            return Err(ConvertError::inconsistent(
                "two identical conversions produced different bundle bytes",
            ));
        }
        if first.shard_ids != second.shard_ids {
            return Err(ConvertError::inconsistent(
                "two identical conversions produced different shard ids",
            ));
        }
        Ok(format!(
            "identical bytes ({} bytes) and {} ids",
            first.bundle_bytes.len(),
            first.shard_ids.len()
        ))
    }));

    scenarios.push(scenario("shard_ids_and_hashes_match_manifest", || {
        let model = swiglu_model(6, 9, seed ^ 16)?;
        let converted = convert_case(
            &model,
            SyntheticDType::F32,
            StorageOrder::RowMajor,
            &GroupSpec::ByCount(3),
            tolerance,
            2,
        )?;
        let mut bundle = BundleFile::open(&converted.bundle_path)?;
        let mut checked = 0_u64;
        for record in bundle.manifest().shards.clone() {
            let recomputed = record.recompute_id(&bundle.manifest().model)?;
            if recomputed != record.shard_id {
                return Err(ConvertError::inconsistent(format!(
                    "shard {} does not recompute",
                    record.shard_id.short()
                )));
            }
            let bytes = bundle.read_shard(&record.shard_id, 64 * 1024 * 1024)?;
            if Sha256Digest::of(&bytes) != record.sha256 {
                return Err(ConvertError::inconsistent(format!(
                    "shard {} content hash disagrees with the manifest",
                    record.shard_id.short()
                )));
            }
            checked += 1;
        }
        Ok(format!("{checked} ids and content hashes recomputed"))
    }));

    scenarios.push(scenario("bundle_validates_all", || {
        let model = swiglu_model(6, 9, seed ^ 17)?;
        let converted = convert_case(
            &model,
            SyntheticDType::F32,
            StorageOrder::RowMajor,
            &GroupSpec::ByCount(3),
            tolerance,
            2,
        )?;
        let report = validate_bundle(
            &converted.bundle_path,
            &ValidationOptions {
                verify: VerifyMode::All,
                ..ValidationOptions::default()
            },
        )?;
        Ok(format!(
            "{} shards verified under VerifyMode::All",
            report.verified_shards.len()
        ))
    }));

    scenarios.push(scenario("small_memory_ceiling", || {
        let model = swiglu_model(8, 10, seed ^ 18)?;
        // The largest contiguous group extent is ceil(10/3)=4 rows of 8 f32
        // values, i.e. 128 bytes; both rows and down use the same size.
        let outcome = convert_case_with_budget(
            &model,
            SyntheticDType::F32,
            StorageOrder::RowMajor,
            &GroupSpec::ByCount(3),
            tolerance,
            vec![1],
            128,
        )?;
        if outcome.peak_inflight_bytes > 128 {
            return Err(ConvertError::MemoryBudgetExceeded {
                requested: outcome.peak_inflight_bytes,
                used: 0,
                limit: 128,
            });
        }
        Ok(format!(
            "converted with a 128-byte ceiling; peak {} bytes",
            outcome.peak_inflight_bytes
        ))
    }));

    let passed = scenarios.iter().all(|scenario| scenario.passed);
    let receipt_numbers = receipt_numbers(&measurements, tolerance, scenarios.len());
    Ok(CourtAReport {
        schema: "selvedge/court-a/v1".to_owned(),
        court: "a".to_owned(),
        passed,
        seed,
        tolerance_abs: tolerance.abs,
        tolerance_rel: tolerance.rel,
        stamp,
        revision,
        generated_by: format!("selvedge-convert {}", env!("CARGO_PKG_VERSION")),
        scenario_count: scenarios.len() as u64,
        scenarios,
        measurements,
        receipt_numbers,
    })
}

fn receipt_numbers(
    measurements: &[MeasurementRow],
    tolerance: Tolerance,
    scenario_count: usize,
) -> CourtAReceiptNumbers {
    let find = |name: &str| measurements.iter().find(|row| row.scenario == name);
    let all = find("all_groups_equal_original");
    let f16 = find("f16_source");
    let bf16 = find("bf16_source");
    CourtAReceiptNumbers {
        all_groups_max_abs_f32: all.map(|row| row.max_abs_f32),
        all_groups_max_rel_f32: all.map(|row| row.max_rel_f32),
        all_groups_max_abs_f64: all.map(|row| row.max_abs_f64),
        all_groups_max_rel_f64: all.map(|row| row.max_rel_f64),
        f16_max_abs_f32: f16.map(|row| row.max_abs_f32),
        bf16_max_abs_f32: bf16.map(|row| row.max_abs_f32),
        tolerance_abs: tolerance.abs,
        tolerance_rel: tolerance.rel,
        scenario_count: scenario_count as u64,
        compared_components: all.map(|row| row.count),
        shard_count: all.map(|row| row.shard_count),
        payload_bytes: all.map(|row| row.payload_bytes),
        peak_inflight_bytes: all.map(|row| row.peak_inflight_bytes),
    }
}

fn push_case(
    scenarios: &mut Vec<ScenarioResult>,
    measurements: &mut Vec<MeasurementRow>,
    outcome: Result<CaseOutcome>,
    name: &str,
    grouping: &str,
) {
    match outcome {
        Ok(case) => {
            let detail = format!(
                "source={} shards={} payload={} bytes peak={} bytes; f32 max abs {:.3e} rel {:.3e}; f64 max abs {:.3e} rel {:.3e}",
                case.source_dtype,
                case.shard_count,
                case.payload_bytes,
                case.peak_inflight_bytes,
                case.max_abs_f32,
                case.max_rel_f32,
                case.max_abs_f64,
                case.max_rel_f64
            );
            measurements.push(MeasurementRow {
                scenario: name.to_owned(),
                source_dtype: case.source_dtype,
                grouping: grouping.to_owned(),
                max_abs_f32: case.max_abs_f32,
                max_rel_f32: case.max_rel_f32,
                max_abs_f64: case.max_abs_f64,
                max_rel_f64: case.max_rel_f64,
                count: case.count,
                shard_count: case.shard_count,
                payload_bytes: case.payload_bytes,
                peak_inflight_bytes: case.peak_inflight_bytes,
                tolerance_abs: Tolerance::default().abs,
                tolerance_rel: Tolerance::default().rel,
            });
            scenarios.push(ScenarioResult {
                name: name.to_owned(),
                passed: true,
                detail,
            });
        }
        Err(error) => scenarios.push(ScenarioResult {
            name: name.to_owned(),
            passed: false,
            detail: error.to_string(),
        }),
    }
}

struct CaseRequest<'a> {
    name: &'a str,
    hidden: u64,
    intermediate: u64,
    spec: GroupSpec,
    dtype: SyntheticDType,
    order: StorageOrder,
    seed: u64,
    batch_sizes: Vec<u64>,
    max_inflight_bytes: u64,
}

fn run_case(request: CaseRequest<'_>, tolerance: Tolerance) -> Result<CaseOutcome> {
    let CaseRequest {
        name,
        hidden,
        intermediate,
        spec,
        dtype,
        order,
        seed,
        batch_sizes,
        max_inflight_bytes,
    } = request;
    let model = swiglu_model(hidden, intermediate, seed)?;
    let outcome = convert_case_with_budget(
        &model,
        dtype,
        order,
        &spec,
        tolerance,
        batch_sizes,
        max_inflight_bytes,
    )?;
    let _ = name;
    Ok(outcome)
}

fn convert_case(
    model: &SyntheticModel,
    dtype: SyntheticDType,
    order: StorageOrder,
    spec: &GroupSpec,
    tolerance: Tolerance,
    max_inflight_mib: u64,
) -> Result<ConvertedCase> {
    convert_case_detailed(
        model,
        dtype,
        order,
        spec,
        tolerance,
        vec![1, 4],
        max_inflight_mib * 1024 * 1024,
    )
}

fn convert_case_with_budget(
    model: &SyntheticModel,
    dtype: SyntheticDType,
    order: StorageOrder,
    spec: &GroupSpec,
    tolerance: Tolerance,
    batch_sizes: Vec<u64>,
    max_inflight_bytes: u64,
) -> Result<CaseOutcome> {
    let detailed = convert_case_detailed(
        model,
        dtype,
        order,
        spec,
        tolerance,
        batch_sizes,
        max_inflight_bytes,
    )?;
    Ok(detailed.outcome)
}

struct ConvertedCase {
    bundle_path: PathBuf,
    bundle_bytes: Vec<u8>,
    shard_ids: Vec<String>,
    outcome: CaseOutcome,
    #[allow(dead_code)]
    temp: TempDir,
}

fn convert_case_detailed(
    model: &SyntheticModel,
    dtype: SyntheticDType,
    order: StorageOrder,
    spec: &GroupSpec,
    tolerance: Tolerance,
    batch_sizes: Vec<u64>,
    max_inflight_bytes: u64,
) -> Result<ConvertedCase> {
    let temp = TempDir::new("selvedge-court-a")?;
    let mut source = model.to_source(dtype, order)?;
    model.validate(&source)?;
    let digest = synthetic_digest(model)?;
    let options = ConversionOptions {
        alignment: 512,
        max_inflight_bytes,
        complete: true,
        bundle_id: None,
        created_by: "selvedge-court-a".to_owned(),
        force: false,
        config_hash: String::new(),
    };
    let plan = plan_conversion(&source, &model.maps, spec, digest, &options)?;
    let (published, _report) = convert(&mut source, &plan, temp.path(), &options)?;
    let verification = verify_bundle(
        &published.path,
        &mut source,
        &model.maps,
        &VerificationOptions {
            tolerance,
            batch_sizes,
            seed: model.seed ^ 0xC0DE,
            max_shard_bytes: 64 * 1024 * 1024,
            max_inflight_bytes: 8 * 1024 * 1024,
        },
    )?;
    let layer = verification
        .layers
        .first()
        .ok_or_else(|| ConvertError::inconsistent("verification produced no layers"))?;
    let bundle_bytes = std::fs::read(&published.path)?;
    let bundle = BundleFile::open(&published.path)?;
    let shard_ids = bundle
        .manifest()
        .shards
        .iter()
        .map(|record| record.shard_id.to_string())
        .collect();
    Ok(ConvertedCase {
        bundle_path: published.path,
        bundle_bytes,
        shard_ids,
        outcome: CaseOutcome {
            max_abs_f32: layer.full_f32.max_abs,
            max_rel_f32: layer.full_f32.max_rel,
            max_abs_f64: layer.full_f64.max_abs,
            max_rel_f64: layer.full_f64.max_rel,
            count: layer.full_f32.count,
            source_dtype: layer.source_dtype.clone(),
            shard_count: verification.validated_shards,
            payload_bytes: _report.payload_bytes,
            peak_inflight_bytes: _report.peak_inflight_bytes,
        },
        temp,
    })
}

fn synthetic_digest(model: &SyntheticModel) -> Result<ModelDigest> {
    let mut builder = Sha256Builder::new();
    for (key, tensor) in &model.tensors {
        builder.update(key.as_bytes());
        for dimension in &tensor.shape {
            builder.update(&dimension.to_le_bytes());
        }
        for value in &tensor.values {
            builder.update(&value.to_le_bytes());
        }
    }
    ModelDigest::from_parts(
        builder.finish(),
        Some(format!("synthetic-seed-{:#018x}", model.seed)),
        None,
    )
    .map_err(ConvertError::from)
}

fn scenario(name: &str, body: impl FnOnce() -> Result<String>) -> ScenarioResult {
    match body() {
        Ok(detail) => ScenarioResult {
            name: name.to_owned(),
            passed: true,
            detail,
        },
        Err(error) => ScenarioResult {
            name: name.to_owned(),
            passed: false,
            detail: error.to_string(),
        },
    }
}
