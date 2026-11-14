//! Decomposition, grouping, orientation and memory-bound tests.

#![allow(clippy::unwrap_used)]

use selvedge_convert::error::ConvertError;
use selvedge_convert::tensor_source::{MemoryBudget, MemoryTensorSource};
use selvedge_convert::{
    ConversionOptions, GroupSpec, MemoryTensor, SyntheticDType, TensorSource, Tolerance,
    VerificationOptions, convert, plan_conversion, resolve_groups, swiglu_model, validate_ffn_map,
    verify_bundle,
};
use selvedge_core::testutil::TempDir;
use selvedge_core::{DType, ModelDigest, Sha256Digest, StorageOrder};

fn digest(seed: u64) -> ModelDigest {
    ModelDigest::from_parts(
        Sha256Digest::of(b"selvedge-test-model"),
        Some(format!("seed-{seed}")),
        None,
    )
    .unwrap()
}

fn options(limit: u64) -> ConversionOptions {
    ConversionOptions {
        alignment: 512,
        max_inflight_bytes: limit,
        complete: true,
        bundle_id: None,
        created_by: "selvedge-test".to_owned(),
        force: false,
        config_hash: "test-config".to_owned(),
    }
}

fn convert_and_verify(
    hidden: u64,
    intermediate: u64,
    spec: &GroupSpec,
    dtype: SyntheticDType,
    order: StorageOrder,
    limit: u64,
) -> Result<(), ConvertError> {
    let model = swiglu_model(
        hidden,
        intermediate,
        0x51E1_0000 + hidden * 31 + intermediate,
    )?;
    let mut source = model.to_source(dtype, order)?;
    model.validate(&source)?;
    let temp = TempDir::new("selvedge-decomp").unwrap();
    let options = options(limit);
    let plan = plan_conversion(&source, &model.maps, spec, digest(model.seed), &options)?;
    let (published, _) = convert(&mut source, &plan, temp.path(), &options)?;
    let report = verify_bundle(
        &published.path,
        &mut source,
        &model.maps,
        &VerificationOptions {
            tolerance: Tolerance::default(),
            batch_sizes: vec![1, 4, 17],
            seed: model.seed,
            max_shard_bytes: 64 * 1024 * 1024,
            max_inflight_bytes: 8 * 1024 * 1024,
        },
    )?;
    assert!(report.ok, "{report:?}");
    Ok(())
}

#[test]
fn group_count_above_intermediate_is_rejected() {
    assert!(resolve_groups(&GroupSpec::ByCount(11), 10, true).is_err());
    assert!(resolve_groups(&GroupSpec::ByCount(10), 10, true).is_ok());
}

#[test]
fn complete_coverage_is_enforced() {
    let partial = GroupSpec::Explicit(vec![vec![0, 1], vec![2, 3]]);
    assert!(resolve_groups(&partial, 5, false).is_ok());
    assert!(resolve_groups(&partial, 5, true).is_err());
    let complete = GroupSpec::Explicit(vec![vec![0, 1], vec![2, 3, 4]]);
    assert!(resolve_groups(&complete, 5, true).is_ok());
}

#[test]
fn disjointness_is_enforced() {
    let overlap = GroupSpec::Explicit(vec![vec![0, 1, 2], vec![2, 3]]);
    assert!(resolve_groups(&overlap, 4, false).is_err());
    let duplicate_inside = GroupSpec::Explicit(vec![vec![0, 0, 1]]);
    assert!(resolve_groups(&duplicate_inside, 4, false).is_err());
    let out_of_range = GroupSpec::Explicit(vec![vec![0, 9]]);
    assert!(resolve_groups(&out_of_range, 4, false).is_err());
}

#[test]
fn non_contiguous_explicit_groups_convert_and_verify() {
    let spec = GroupSpec::Explicit(vec![vec![0, 2, 4], vec![1, 3, 5], vec![6, 7, 8, 9]]);
    convert_and_verify(
        7,
        10,
        &spec,
        SyntheticDType::F32,
        StorageOrder::RowMajor,
        8 * 1024 * 1024,
    )
    .unwrap();
}

#[test]
fn column_major_and_transposed_sources_convert() {
    for order in [
        StorageOrder::ColumnMajor,
        StorageOrder::Transposed { axes: vec![1, 0] },
    ] {
        convert_and_verify(
            6,
            9,
            &GroupSpec::ByCount(3),
            SyntheticDType::F32,
            order,
            8 * 1024 * 1024,
        )
        .unwrap();
    }
}

#[test]
fn f16_and_bf16_sources_convert() {
    for dtype in [SyntheticDType::F16, SyntheticDType::BF16] {
        convert_and_verify(
            6,
            8,
            &GroupSpec::ByCount(2),
            dtype,
            StorageOrder::RowMajor,
            8 * 1024 * 1024,
        )
        .unwrap();
    }
}

#[test]
fn memory_ceiling_is_enforced() {
    let model = swiglu_model(8, 10, 0xABC).unwrap();
    let mut source = model
        .to_source(SyntheticDType::F32, StorageOrder::RowMajor)
        .unwrap();
    let options = options(4);
    let plan = plan_conversion(
        &source,
        &model.maps,
        &GroupSpec::ByCount(3),
        digest(model.seed),
        &options,
    )
    .unwrap();
    let temp = TempDir::new("selvedge-decomp-oom").unwrap();
    match convert(&mut source, &plan, temp.path(), &options) {
        Err(ConvertError::MemoryBudgetExceeded { limit, .. }) => assert_eq!(limit, 4),
        other => panic!("expected a budget failure, got {other:?}"),
    }
}

#[test]
fn map_validation_rejects_wrong_orientation_and_dtype() {
    let model = swiglu_model(5, 6, 0xDEF).unwrap();
    let source = model
        .to_source(SyntheticDType::F32, StorageOrder::RowMajor)
        .unwrap();
    let good = model.map().unwrap();
    assert!(validate_ffn_map(&source, good).is_ok());

    let mut wrong = good.clone();
    // Orientation is a caller declaration, not something the shapes can
    // disambiguate: swapping hidden and intermediate maps a rank-2 gate/up/down
    // triple to another consistent interpretation. A genuinely inconsistent
    // map is one whose down tensor does not match the declared orientation.
    wrong.down = wrong.gate.clone();
    assert!(validate_ffn_map(&source, &wrong).is_err());

    let mut ints = MemoryTensorSource::new();
    for key in source.tensor_keys() {
        let descriptor = source.descriptor(&key).unwrap();
        let elements = descriptor.elements;
        let data = vec![0_u8; elements as usize];
        ints.insert(
            MemoryTensor::from_raw(
                key,
                DType::I8,
                descriptor.shape.clone(),
                StorageOrder::RowMajor,
                data,
            )
            .unwrap(),
        )
        .unwrap();
    }
    assert!(matches!(
        validate_ffn_map(&ints, good),
        Err(ConvertError::SourceDTypeUnsupported { .. })
    ));
}

#[test]
fn memory_budget_refuses_overcommit() {
    let mut budget = MemoryBudget::new(8).unwrap();
    budget.reserve(8).unwrap();
    assert!(matches!(
        budget.reserve(1),
        Err(ConvertError::MemoryBudgetExceeded { .. })
    ));
    budget.release(8).unwrap();
    budget.reserve(1).unwrap();
    assert_eq!(budget.peak(), 8);
}
