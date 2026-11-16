//! Exact SwiGLU FFN decomposition, repacking and reconstruction proof.
//!
//! This crate owns the model path for SELVEDGE phase S2:
//!
//! - [`safetensors`] is a safe, streaming reader for the real safetensors
//!   container (no `unsafe`, no memory mapping, typed rejection of unknown
//!   dtypes and malformed layouts);
//! - [`ffn`] holds the typed [`ffn::FfnTensorMap`], deterministic grouping rules
//!   and independent f32/f64 reference arithmetic;
//! - [`convert`] streams one channel group at a time into a spool file and
//!   publishes a format-v1 bundle through the atomic writer;
//! - [`verify`] re-reads the published bytes and measures reconstruction error
//!   against the source checkpoint;
//! - [`synthetic`] provides the deterministic checkpoint and safetensors writer
//!   used by the fixture and the offline demo.
//!
//! The grouped identity `FFN(x) = sum_g FFN_g(x)` is exact in real arithmetic.
//! Floating-point reduction order changes, so this crate measures error and
//! never claims bitwise equality.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod accounting;
pub mod accounting_qwen35;
pub mod convert;
pub mod court_a;
pub mod error;
pub mod ffn;
pub mod masked;
pub mod real_parity;
pub mod safetensors;
pub mod synthetic;
pub mod tensor_source;
pub mod verify;

pub use accounting::{
    AccountingOptions, BudgetReport, CategoryTotals, DEFAULT_ACTIVE_CEILING, DEFAULT_GROUP_COUNT,
    ModelAccountingReport, ModelIdentity, TensorAccount, TieReport, account_model, category_totals,
};
pub use accounting_qwen35::{
    ActivePath, ActivePathReport, CertificationReport, QWEN35_MANDATE_CEILING, QWEN35_SCHEMA,
    Qwen35AccountingReport, Qwen35Options, TieAssertion, Tower, TowerTensor, account_model_qwen35,
    detect_model_type,
};
pub use convert::{
    ConversionOptions, ConversionPlan, ConversionReport, LayerPlan, PlannedShard, ShardReport,
    TRANSFORM_DOWN, TRANSFORM_GATE, TRANSFORM_UP, convert, model_digest_for_path, plan_conversion,
    report_plan,
};
pub use error::{ConvertError, Result};
pub use ffn::{
    DiscoveredFfn, FfnDims, FfnOrientation, FfnTensorMap, Group, GroupMatrices, GroupSpec,
    discover_ffn_maps, ffn_dense_f32, ffn_dense_f64, ffn_grouped_f32, ffn_grouped_f64,
    resolve_groups, validate_ffn_map,
};
pub use masked::{
    MAX_INPUT_BYTES, MaskedOptions, MaskedReport, MaskedRequest, masked_grouped_from_bundle,
};
pub use real_parity::{
    BF16_PARITY_ABS, BF16_PARITY_REL, BF16_PARITY_SCALE, F32_PARITY_ABS, F32_PARITY_REL,
    F32_PARITY_SCALE, LayerParity, ParityErrors, RealParityOptions, RealParityReport,
    TOLERANCE_POLICY, verify_real_parity,
};
pub use safetensors::{
    CheckpointSource, HeaderIndex, MAX_HEADER_BYTES, MultiSafetensors, PositionalRead, Safetensors,
    TensorEntry, parse_bytes,
};
pub use synthetic::{
    DeterministicRng, SYNTHETIC_HIDDEN, SYNTHETIC_INTERMEDIATE, SYNTHETIC_SEED, SyntheticDType,
    SyntheticModel, SyntheticTensor, encode_safetensors, swiglu_model, write_safetensors,
};
pub use tensor_source::{
    MemoryBudget, MemoryTensor, MemoryTensorSource, TensorDescriptor, TensorSource,
};
pub use verify::{
    ErrorMeasurement, GroupMeasurement, LayerVerification, Tolerance, VerificationOptions,
    VerificationReport, verify_bundle, verify_with_defaults,
};
