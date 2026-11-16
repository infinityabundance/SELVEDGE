//! Single-device inference for SELVEDGE: Qwen3 with bundle-backed MLP groups.
//!
//! This crate executes the pinned Qwen3-0.6B configuration on one device with
//! two clearly separated execution modes:
//!
//! - [`config::ExecutionMode::Exact`] executes the bundle's **complete** native
//!   shard set for every layer and produces a
//!   [`selvedge_core::ExactSelection`] only after
//!   [`selvedge_core::NativeSetWitness::compare`] proves the executed set equal
//!   to the declaration derived from the validated manifest;
//! - [`config::ExecutionMode::Approximate`] executes a selector-provided subset
//!   and records both [`selvedge_core::CausalAvailability`] and
//!   [`selvedge_core::PhysicalCompliance`] for the run. It never claims
//!   exactness.
//!
//! Everything that can change bytes on the wire (cache policy, cache size,
//! prefetch, injected transfer delay, read order) is confined to
//! [`config::StreamConfig`]. None of those knobs can change the executed shard
//! set or the arithmetic order, which is what makes the Court E parity claim
//! testable by construction rather than by inspection: the group execution
//! order is always ascending `(layer, group index)` and each dense reduction is
//! a fixed-order dot product (see [`kernels`]).
//!
//! The crate contains no `unsafe`, no floating-point environment changes, no
//! thread-count-dependent reduction order and no wall-clock-dependent behaviour.
//! Sampling is greedy only; there is no RNG in the execution path.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod account;
pub mod bundle;
pub mod config;
pub mod error;
pub mod kernels;
pub mod model;
pub mod modes;
pub mod prefetch;
pub mod residency;
pub mod run;
pub mod tokenizer;
pub mod weights;

pub use account::{AccountingRow, AccountingWriter, RunSummary};
pub use bundle::{
    BundlePlan, GroupPlan, GroupReader, GroupTensors, NativeDeclaration, ShardFragment,
};
pub use config::{
    ExecutionMode, ModelDescriptor, PrefillMode, PromptSource, Qwen3Config, RunConfig,
    SelectorSpec, StreamConfig, Tolerance,
};
pub use error::{Result, RuntimeError};
pub use model::{KvCache, Model, TokenTrace};
pub use modes::{ExactState, GroupChoice, ModeState};
pub use prefetch::{PrefetchAccounting, PrefetchOutcome, StallAccounting};
pub use residency::{DecodeDtype, DecodeKey, DecodeResidency, ResidencyCounters, ResidencyStats};
pub use run::{LogitsComparison, RunOutcome, Runner, TokenStep, logits_digest};
pub use tokenizer::GreedyTokenizer;
pub use weights::Backbone;
