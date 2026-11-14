//! Parity fixtures exported by the Python harness.
//!
//! The fixture file contains real calibration contexts with their pre-decision
//! group features and the Python selector decisions. The Rust Court B test
//! replays every selector on the same records and requires exact equality.

use std::path::Path;

use serde::Deserialize;

use crate::error::Result;
use crate::input::{GroupFeature, SelectorInput};

/// Expected Python per-layer decisions.
#[derive(Clone, Debug, Deserialize)]
pub struct FixtureExpected {
    /// Activation magnitude.
    pub magnitude: Vec<u32>,
    /// Per-token frequency.
    pub frequency: Vec<u32>,
    /// Deterministic random.
    pub random: Vec<u32>,
    /// Static magnitude ranking.
    #[serde(rename = "static-magnitude")]
    pub static_magnitude: Vec<u32>,
    /// Static frequency ranking.
    #[serde(rename = "static-frequency")]
    pub static_frequency: Vec<u32>,
    /// Learned unary.
    #[serde(rename = "learned-unary")]
    pub learned_unary: Vec<u32>,
    /// Learned interaction-aware.
    #[serde(rename = "learned-interaction")]
    pub learned_interaction: Vec<u32>,
}

/// One fixture record.
#[derive(Clone, Debug, Deserialize)]
pub struct FixtureRecord {
    /// Layer.
    pub layer: u32,
    /// Layer count.
    pub layers: u32,
    /// Position.
    pub position: u32,
    /// Sequence key.
    pub sequence_key: u64,
    /// Group features.
    pub groups: Vec<GroupFeature>,
    /// Optional hidden state.
    #[serde(default)]
    pub hidden: Option<Vec<f32>>,
    /// Expected decisions.
    pub expected: FixtureExpected,
}

impl FixtureRecord {
    /// Convert to the strict selector input (drops the expected block).
    #[must_use]
    pub fn input(&self) -> SelectorInput {
        SelectorInput {
            layer: self.layer,
            layers: self.layers,
            position: self.position,
            sequence_key: self.sequence_key,
            groups: self.groups.clone(),
            hidden: self.hidden.clone(),
        }
    }
}

/// The fixture file.
#[derive(Clone, Debug, Deserialize)]
pub struct Fixtures {
    /// Schema identifier.
    pub schema: String,
    /// Records.
    pub records: Vec<FixtureRecord>,
}

/// Read the fixture file.
///
/// # Errors
///
/// Returns an IO or JSON error.
pub fn read_fixtures(path: &Path) -> Result<Fixtures> {
    let text = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}
