//! `selvedge-atlas` — causal quality observations, interaction graphs and
//! selection.
//!
//! The crate owns three boundaries:
//!
//! - [`QualityObservation`] records one measured intervention and names the
//!   receipt that produced it;
//! - [`InteractionGraph`] aggregates unary and pairwise observations with
//!   typed invariants;
//! - [`SelectorInput`] is the only record a deployed selector may read, and it
//!   cannot carry teacher logits, future tokens or gold labels.
//!
//! [`TwoViewDecision`] fixes the `value_if_executed` /
//! `cost_to_materialize_now` contract. The cost head is explicitly unavailable
//! until the M3 trace scheduler fills it; a missing cost model is never
//! reported as zero.
//!
//! The oracle is not part of this crate: it needs the per-layer group
//! contributions and is not deployable.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod calibration;
pub mod compare;
pub mod error;
pub mod fit;
pub mod fixtures;
pub mod graph;
pub mod input;
pub mod observation;
pub mod selector;
pub mod twoview;
pub mod weights;

pub use calibration::{
    CalibrationReport, ReliabilityPoint, is_monotone, reliability_curve, report,
};
pub use compare::{
    CompareArm, CompareArmReport, CompareInput, CompareItem, CompareReport, CompareReportEntry,
    active_unique_elements, compare,
};
pub use error::{AtlasError, Result};
pub use fit::{FitData, FitLayer, fit, solve_ridge};
pub use fixtures::{FixtureExpected, FixtureRecord, Fixtures, read_fixtures};
pub use graph::{EdgeKey, EdgeStats, HyperEdge, InteractionGraph, NodeKey, NodeStats};
pub use input::{GROUP_FEATURE_COUNT, GroupFeature, SelectorInput};
pub use observation::QualityObservation;
pub use selector::{
    LearnedInteractionSelector, LearnedUnarySelector, MagnitudeSelector, RandomSelector, SEED,
    Selector, SelectorKind, StaticKind, StaticSelector, select_named,
};
pub use twoview::{
    CostEstimate, CostHead, DecisionStatus, TwoViewDecision, UnavailableCostHead, ValueEstimate,
    ValueHead, combine,
};
pub use weights::{AtlasWeights, Rankings};
