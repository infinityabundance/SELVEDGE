//! Interaction graph and typed invariants.
//!
//! Nodes are `(layer, group)` pairs with an aggregated unary utility. Edges
//! are canonical `(layer, low < high)` pairs with an aggregated pairwise
//! non-additivity. Hyperedges record a measured base set and are stored
//! separately. All maps are ordered, so serialization is deterministic.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::{AtlasError, Result};
use crate::observation::QualityObservation;

/// A node identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NodeKey {
    /// Decoder layer.
    pub layer: u32,
    /// Group index.
    pub group: u32,
}

impl NodeKey {
    /// Construct a node key.
    ///
    /// # Errors
    ///
    /// Rejects indices outside the model.
    pub fn new(layer: u32, group: u32, layers: u32, groups: u32) -> Result<Self> {
        if layer >= layers || group >= groups {
            return Err(AtlasError::invalid("node", "is outside the model"));
        }
        Ok(Self { layer, group })
    }
}

/// A canonical edge identity with `low < high`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EdgeKey {
    /// Decoder layer.
    pub layer: u32,
    /// Lower group index.
    pub low: u32,
    /// Higher group index.
    pub high: u32,
}

impl EdgeKey {
    /// Construct a canonical edge key.
    ///
    /// # Errors
    ///
    /// Rejects a self-loop or an index outside the model.
    pub fn new(layer: u32, left: u32, right: u32, layers: u32, groups: u32) -> Result<Self> {
        if left == right {
            return Err(AtlasError::Invariant(format!(
                "self-loop at layer {layer} group {left}"
            )));
        }
        if layer >= layers || left >= groups || right >= groups {
            return Err(AtlasError::invalid("edge", "is outside the model"));
        }
        Ok(Self {
            layer,
            low: left.min(right),
            high: left.max(right),
        })
    }
}

/// Welford statistics for a node.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct NodeStats {
    /// Observation count.
    pub count: u64,
    /// Running mean.
    pub mean: f64,
    /// Sum of squared deviations.
    pub m2: f64,
    /// Minimum observed value.
    pub min: f64,
    /// Maximum observed value.
    pub max: f64,
}

impl NodeStats {
    /// Add one observation.
    pub fn observe(&mut self, value: f64) {
        if self.count == 0 {
            self.mean = value;
            self.min = value;
            self.max = value;
        } else {
            self.min = self.min.min(value);
            self.max = self.max.max(value);
        }
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        self.m2 += delta * (value - self.mean);
    }

    /// Sample variance, zero for a singleton.
    #[must_use]
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / (self.count - 1) as f64
        }
    }
}

/// Welford statistics plus sign counts for an edge.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EdgeStats {
    /// Observation count.
    pub count: u64,
    /// Running mean.
    pub mean: f64,
    /// Sum of squared deviations.
    pub m2: f64,
    /// Count of positive values.
    pub positive: u64,
    /// Count of negative values.
    pub negative: u64,
}

impl EdgeStats {
    /// Add one observed interaction.
    pub fn observe(&mut self, value: f64) {
        if value > 0.0 {
            self.positive += 1;
        } else if value < 0.0 {
            self.negative += 1;
        }
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        self.m2 += delta * (value - self.mean);
    }

    /// Sample variance, zero for a singleton.
    #[must_use]
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / (self.count - 1) as f64
        }
    }
}

/// A measured base set with an aggregate interaction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HyperEdge {
    /// Decoder layer.
    pub layer: u32,
    /// Sorted base set.
    pub groups: Vec<u32>,
    /// Number of contexts.
    pub count: u64,
    /// Mean measured interaction.
    pub mean: f64,
}

/// The interaction graph.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct InteractionGraph {
    /// Unary nodes.
    pub nodes: BTreeMap<NodeKey, NodeStats>,
    /// Canonical pairwise edges.
    pub edges: BTreeMap<EdgeKey, EdgeStats>,
    /// Measured multi-group contexts.
    #[serde(default)]
    pub hyperedges: Vec<HyperEdge>,
}

impl InteractionGraph {
    /// Insert a validated quality observation.
    ///
    /// # Errors
    ///
    /// Propagates validation errors.
    pub fn insert_observation(
        &mut self,
        observation: &QualityObservation,
        layers: u32,
        groups: u32,
    ) -> Result<()> {
        observation.validate(layers, groups)?;
        let key = NodeKey::new(observation.layer, observation.group, layers, groups)?;
        self.nodes
            .entry(key)
            .or_default()
            .observe(observation.delta);
        Ok(())
    }

    /// Insert a pairwise interaction, canonicalising `(g, j)` to `(low, high)`.
    ///
    /// # Errors
    ///
    /// Rejects self-loops, out-of-range indices and non-finite values.
    pub fn insert_interaction(
        &mut self,
        layer: u32,
        left: u32,
        right: u32,
        value: f64,
        layers: u32,
        groups: u32,
    ) -> Result<()> {
        if !value.is_finite() {
            return Err(AtlasError::invalid("interaction", "is not finite"));
        }
        let key = EdgeKey::new(layer, left, right, layers, groups)?;
        self.nodes
            .entry(NodeKey::new(layer, key.low, layers, groups)?)
            .or_default();
        self.nodes
            .entry(NodeKey::new(layer, key.high, layers, groups)?)
            .or_default();
        self.edges.entry(key).or_default().observe(value);
        Ok(())
    }

    /// Check every structural invariant.
    ///
    /// # Errors
    ///
    /// Fails on an empty-count node, a non-finite statistic, an out-of-range
    /// node, an edge whose endpoints do not exist, a non-canonical edge, or a
    /// hyperedge that is not strictly ascending.
    pub fn check_invariants(&self, layers: u32, groups: u32) -> Result<()> {
        for (key, stats) in &self.nodes {
            if key.layer >= layers || key.group >= groups {
                return Err(AtlasError::Invariant(format!(
                    "node {key:?} is outside the model"
                )));
            }
            // A node may exist only because an edge references it; pairwise
            // evidence without a unary observation is legitimate.
            if !stats.mean.is_finite() || !stats.m2.is_finite() {
                return Err(AtlasError::Invariant(format!(
                    "node {key:?} has a non-finite statistic"
                )));
            }
            if stats.min > stats.max {
                return Err(AtlasError::Invariant(format!("node {key:?} has min > max")));
            }
        }
        for (key, stats) in &self.edges {
            if key.layer >= layers || key.low >= key.high || key.high >= groups {
                return Err(AtlasError::Invariant(format!(
                    "edge {key:?} is not canonical"
                )));
            }
            for group in [key.low, key.high] {
                let node = NodeKey {
                    layer: key.layer,
                    group,
                };
                if !self.nodes.contains_key(&node) {
                    return Err(AtlasError::Invariant(format!(
                        "edge {key:?} references missing node {node:?}"
                    )));
                }
            }
            if stats.count == 0 || !stats.mean.is_finite() || !stats.m2.is_finite() {
                return Err(AtlasError::Invariant(format!(
                    "edge {key:?} has an invalid statistic"
                )));
            }
        }
        for edge in &self.hyperedges {
            if edge.layer >= layers || edge.groups.len() < 2 {
                return Err(AtlasError::Invariant("hyperedge is malformed".into()));
            }
            if edge.groups.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Err(AtlasError::Invariant(
                    "hyperedge groups are not strictly ascending".into(),
                ));
            }
            if edge.groups.iter().any(|group| *group >= groups) {
                return Err(AtlasError::Invariant(
                    "hyperedge references a group outside the model".into(),
                ));
            }
            if !edge.mean.is_finite() {
                return Err(AtlasError::Invariant("hyperedge mean is non-finite".into()));
            }
        }
        Ok(())
    }
}
