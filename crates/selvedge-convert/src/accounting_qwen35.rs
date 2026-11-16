//! Exact parameter accounting for a Qwen3.5 checkpoint (accounting schema v2).
//!
//! Qwen3.5 is not a Qwen3 checkpoint with a renamed config. Three things make
//! the v1 auditor in [`crate::accounting`] structurally unable to account it,
//! and this module exists to account them:
//!
//! 1. the language model is **nested** (`text_config`) and its tensors live under
//!    `model.language_model.*`, so there is no top-level `hidden_size`;
//! 2. it is **multimodal**: a vision tower (`model.visual.*`) is a real, trained
//!    part of the checkpoint, and a text-only execution never reads it;
//! 3. it carries a **multi-token-prediction head** (`mtp.*`) that is trained and
//!    shipped, executed only when drafting is enabled.
//!
//! Consequences for the numbers, stated as facts:
//!
//! - the element sum of the *text tower* is the model a text-only runtime
//!   executes, and it is strictly smaller than the checkpoint;
//! - therefore any claim of the form "N unique active parameters" must name the
//!   declared execution path, and the auditor refuses a declaration that is
//!   inconsistent with the tensors;
//! - a tie is verified, never trusted: a tied config with an `lm_head.weight`
//!   whose bytes differ from the embedding is a hard failure, and a declared
//!   tie against such a pair is refused.
//!
//! Every count here is derived from the safetensors header (shape products) and
//! cross-checked three ways: shape product against the declared element count,
//! element count times dtype width against the declared byte span, and the
//! summed dtype-aware bytes against the Hub's `model.safetensors.index.json`
//! `metadata.total_size`. A mismatch in any of the three fails closed.
//!
//! No FLOP number is produced here. Qwen3.5's Gated DeltaNet layers are a scan,
//! not a matmul stack, so the v1 FLOP formulas do not apply; inventing one would
//! be a fabricated number. `flops` is deliberately absent from this schema.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use selvedge_core::DType;
use serde::{Deserialize, Serialize};

use crate::accounting::TieReport;
use crate::error::{ConvertError, Result};
use crate::safetensors::CheckpointSource;
use crate::tensor_source::{TensorDescriptor, TensorSource};

/// Schema identifier written into every v2 report.
pub const QWEN35_SCHEMA: &str = "selvedge/model-accounting/v2";

/// The mandate's unique-active-elements ceiling, repeated here because a v2
/// certificate cites it by name.
pub const QWEN35_MANDATE_CEILING: u64 = 1_000_000_000;

/// Rows per chunk when verifying a tie. Bounded for a fixed allocation.
const TIE_CHECK_ROWS: u64 = 8_192;

/// Which tower a tensor belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tower {
    /// The language model: embeddings, hybrid layers, final norm, tied head.
    Text,
    /// The vision encoder.
    Vision,
    /// The multi-token-prediction head.
    Mtp,
}

impl Tower {
    /// Stable lowercase name used in reports.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Vision => "vision",
            Self::Mtp => "mtp",
        }
    }

    fn namespace_prefix(self) -> &'static str {
        match self {
            Self::Text => "model.language_model.",
            Self::Vision => "model.visual.",
            Self::Mtp => "mtp.",
        }
    }
}

/// A declared execution path. The declaration is what the auditor tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ActivePath {
    /// Language model only: the vision tower and the MTP head are not read.
    #[default]
    TextOnly,
    /// Language model plus the MTP head (multi-token drafting).
    TextAndMtp,
}

impl ActivePath {
    /// Stable lowercase name used in reports and on the command line.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TextOnly => "text-only",
            Self::TextAndMtp => "text-and-mtp",
        }
    }

    /// Parse the command-line spelling.
    ///
    /// # Errors
    ///
    /// Rejects every other spelling; there is no silent default.
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "text-only" => Ok(Self::TextOnly),
            "text-and-mtp" => Ok(Self::TextAndMtp),
            other => Err(ConvertError::plan(format!(
                "unknown active path {other:?}; accepted: text-only, text-and-mtp"
            ))),
        }
    }

    /// Towers the declared path executes.
    #[must_use]
    pub fn towers(self) -> &'static [Tower] {
        match self {
            Self::TextOnly => &[Tower::Text],
            Self::TextAndMtp => &[Tower::Text, Tower::Mtp],
        }
    }
}

/// How the tie between the embedding and the output projection is decided.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TieAssertion {
    /// Use `tie_word_embeddings` from the config and verify it against bytes.
    #[default]
    FromConfig,
    /// The caller declares the weights tied; refused if the bytes contradict it.
    Tied,
    /// The caller declares the weights untied; refused if no `lm_head.weight`
    /// exists to read.
    Untied,
}

/// Options for [`account_model_qwen35`].
#[derive(Clone, Debug)]
pub struct Qwen35Options {
    /// Ceiling the certificate is evaluated against.
    pub ceiling: u64,
    /// The declared execution path.
    pub active_path: ActivePath,
    /// How the tie is decided.
    pub tie_assertion: TieAssertion,
    /// When set, additionally certify the active set against this ceiling.
    pub certify_ceiling: Option<u64>,
    /// `(name, elements)` pairs the observed totals must equal exactly.
    pub expect_totals: Vec<(String, u64)>,
    /// Include the streaming checkpoint digest.
    pub compute_digest: bool,
    /// Receipt stamp, recorded verbatim.
    pub stamp: Option<String>,
    /// Git revision, recorded verbatim.
    pub revision: Option<String>,
}

impl Default for Qwen35Options {
    fn default() -> Self {
        Self {
            ceiling: QWEN35_MANDATE_CEILING,
            active_path: ActivePath::TextOnly,
            tie_assertion: TieAssertion::FromConfig,
            certify_ceiling: None,
            expect_totals: Vec::new(),
            compute_digest: true,
            stamp: None,
            revision: None,
        }
    }
}

impl Qwen35Options {
    /// Validate the options.
    ///
    /// # Errors
    ///
    /// Rejects a zero ceiling or a zero certification ceiling.
    pub fn validate(&self) -> Result<()> {
        if self.ceiling == 0 {
            return Err(ConvertError::plan("accounting ceiling must be non-zero"));
        }
        if self.certify_ceiling == Some(0) {
            return Err(ConvertError::plan(
                "certification ceiling must be non-zero when requested",
            ));
        }
        Ok(())
    }
}

/// One tensor's exact contribution, tagged with its tower.
#[derive(Clone, Debug, Serialize)]
pub struct TowerTensor {
    /// Checkpoint tensor key.
    pub tensor: String,
    /// Tower the tensor belongs to.
    pub tower: String,
    /// Logical component inside the tower.
    pub component: String,
    /// Logical shape.
    pub shape: Vec<u64>,
    /// Logical element count.
    pub elements: u64,
    /// Byte span declared by the header.
    pub bytes: u64,
    /// Dtype as recorded in the header.
    pub dtype: String,
    /// Text layer index, when the tensor belongs to one.
    pub layer: Option<u32>,
    /// Whether the tensor is part of the FFN (the removable set in a sharded
    /// design).
    pub ffn: bool,
}

/// Per-tower totals.
#[derive(Clone, Debug, Serialize)]
pub struct TowerSummary {
    /// Tower name.
    pub tower: String,
    /// Tensor count.
    pub tensors: u64,
    /// Elements present in the checkpoint for this tower.
    pub checkpoint_elements: u64,
    /// Unique elements after the tie is applied (text only; other towers are
    /// disjoint from the text tower and from each other).
    pub unique_elements: u64,
    /// Bytes present in the checkpoint for this tower.
    pub bytes: u64,
    /// Elements in FFN tensors.
    pub ffn_elements: u64,
    /// Elements that are always active once the tower executes.
    pub always_active_elements: u64,
    /// Whether the declared path executes this tower.
    pub executed_by_declared_path: bool,
}

/// Fixed facts read from the nested text config.
#[derive(Clone, Debug, Serialize)]
pub struct TextFacts {
    /// `text_config.model_type`.
    pub model_type: String,
    /// Hidden width.
    pub hidden: u64,
    /// FFN intermediate width.
    pub intermediate: u64,
    /// Layer count.
    pub layers: u64,
    /// Vocabulary size.
    pub vocab: u64,
    /// Number of Gated DeltaNet layers.
    pub linear_layers: u64,
    /// Number of gated-attention layers.
    pub attention_layers: u64,
    /// Indices of the attention layers.
    pub attention_layer_indices: Vec<u32>,
    /// Attention query heads.
    pub attention_heads: u64,
    /// Attention KV heads.
    pub attention_kv_heads: u64,
    /// Attention head dimension.
    pub attention_head_dim: u64,
    /// Whether the attention output gate is fused into `q_proj`.
    pub attn_output_gate: bool,
    /// DeltaNet key heads.
    pub linear_key_heads: u64,
    /// DeltaNet key head dimension.
    pub linear_key_head_dim: u64,
    /// DeltaNet value heads.
    pub linear_value_heads: u64,
    /// DeltaNet value head dimension.
    pub linear_value_head_dim: u64,
    /// Depthwise short-convolution kernel width.
    pub conv_kernel: u64,
    /// RoPE theta.
    pub rope_theta: f64,
    /// Fraction of each head that is rotated.
    pub partial_rotary_factor: f64,
    /// Rotated dimensions per head (`head_dim * partial_rotary_factor`).
    pub rotary_dim: u64,
    /// mRoPE section widths.
    pub mrope_section: Vec<u64>,
    /// Whether mRoPE is interleaved.
    pub mrope_interleaved: bool,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f64,
    /// MTP layer count.
    pub mtp_layers: u64,
    /// Whether the MTP head owns dedicated embeddings.
    pub mtp_use_dedicated_embeddings: bool,
    /// Config-declared tie.
    pub tie_word_embeddings: bool,
}

/// Fixed facts read from `vision_config`.
#[derive(Clone, Debug, Serialize)]
pub struct VisionFacts {
    /// Encoder depth.
    pub depth: u64,
    /// Encoder hidden width.
    pub hidden: u64,
    /// Encoder FFN width.
    pub intermediate: u64,
    /// Encoder attention heads.
    pub heads: u64,
    /// Projection width into the language model.
    pub out_hidden: u64,
    /// Patch edge.
    pub patch_size: u64,
    /// Spatial merge factor.
    pub spatial_merge_size: u64,
    /// Temporal patch size.
    pub temporal_patch_size: u64,
    /// Input channels.
    pub in_channels: u64,
    /// Learned position table rows.
    pub position_embeddings: u64,
    /// Activation.
    pub hidden_act: String,
}

/// Model identity, plus the tower facts.
#[derive(Clone, Debug, Serialize)]
pub struct Qwen35Identity {
    /// Path the auditor read.
    pub path: String,
    /// Top-level `model_type`.
    pub model_type: String,
    /// Declared architectures.
    pub architectures: Vec<String>,
    /// SHA-256 of `config.json`.
    pub config_sha256: String,
    /// Streaming digest of the checkpoint files, when requested.
    pub checkpoint_digest: Option<String>,
    /// Number of checkpoint files read.
    pub checkpoint_files: u64,
    /// Text tower facts.
    pub text: TextFacts,
    /// Vision tower facts, absent when the config declares no vision tower.
    pub vision: Option<VisionFacts>,
}

/// Checkpoint-level totals.
#[derive(Clone, Debug, Serialize)]
pub struct CheckpointTotals {
    /// Tensors classified.
    pub tensors: u64,
    /// Element sum over every tensor.
    pub tensor_elements: u64,
    /// Byte sum over every tensor, dtype-aware.
    pub tensor_bytes: u64,
    /// Unique elements after the tie is applied.
    pub unique_elements: u64,
    /// `metadata.total_size` from `model.safetensors.index.json`, when present.
    pub index_total_bytes: Option<u64>,
    /// Whether the index total equals the dtype-aware byte sum.
    pub index_total_matches: Option<bool>,
    /// Per-dtype element and byte sums, so a reader can check the total.
    pub by_dtype: Vec<DtypeTotal>,
}

/// Elements and bytes for one dtype.
#[derive(Clone, Debug, Serialize)]
pub struct DtypeTotal {
    /// Dtype name.
    pub dtype: String,
    /// Tensors of this dtype.
    pub tensors: u64,
    /// Elements of this dtype.
    pub elements: u64,
    /// Bytes of this dtype.
    pub bytes: u64,
}

/// A tensor read by more than one executed tower.
#[derive(Clone, Debug, Serialize)]
pub struct SharedRead {
    /// Tensor key.
    pub tensor: String,
    /// Towers that read it.
    pub towers: Vec<String>,
    /// Why it is shared.
    pub reason: String,
}

/// The declared-path report.
#[derive(Clone, Debug, Serialize)]
pub struct ActivePathReport {
    /// The declared path.
    pub declared: String,
    /// Towers the declaration executes.
    pub executed_towers: Vec<String>,
    /// Towers the declaration excludes.
    pub excluded_towers: Vec<String>,
    /// Unique elements the declared path reads.
    pub unique_active_elements: u64,
    /// Elements excluded by the declaration, per tower.
    pub excluded_elements: Vec<ExcludedTower>,
    /// Tensors the declared path reads from a tower it attributes elsewhere.
    pub shared_reads: Vec<SharedRead>,
    /// Verdict on the declaration: `consistent` or `inconsistent`.
    pub verdict: String,
}

/// One tower excluded by the declared path.
#[derive(Clone, Debug, Serialize)]
pub struct ExcludedTower {
    /// Tower name.
    pub tower: String,
    /// Elements excluded.
    pub elements: u64,
    /// Why it is excluded.
    pub reason: String,
}

/// One term of the certification arithmetic.
#[derive(Clone, Debug, Serialize)]
pub struct CertificationTerm {
    /// Human label.
    pub label: String,
    /// Elements.
    pub elements: u64,
    /// `+` or `-`, so the printed arithmetic can be re-added by a reader.
    pub sign: String,
}

/// The ceiling certificate.
#[derive(Clone, Debug, Serialize)]
pub struct CertificationReport {
    /// Whether a certificate was requested.
    pub requested: bool,
    /// Ceiling certified against.
    pub ceiling: u64,
    /// Unique active elements of the declared path.
    pub active_unique_elements: u64,
    /// Ceiling minus active; negative means the ceiling is exceeded.
    pub headroom_elements: i64,
    /// `certified` or `exceeds_ceiling`.
    pub verdict: String,
    /// The arithmetic, term by term.
    pub arithmetic: Vec<CertificationTerm>,
    /// Plain-language statement of the verdict.
    pub explanation: String,
}

/// One expected-total check.
#[derive(Clone, Debug, Serialize)]
pub struct ExpectationCheck {
    /// Name the caller used (`text`, `vision`, `mtp`, `checkpoint`).
    pub name: String,
    /// Elements the caller asserted.
    pub expected: u64,
    /// Elements the auditor observed.
    pub observed: u64,
    /// Whether they agree.
    pub matches: bool,
}

/// The v2 accounting report.
#[derive(Clone, Debug, Serialize)]
pub struct Qwen35AccountingReport {
    /// Schema identifier.
    pub schema: String,
    /// Whether every claim in this report held.
    pub ok: bool,
    /// Receipt stamp.
    pub stamp: Option<String>,
    /// Git revision.
    pub revision: Option<String>,
    /// Producer identity.
    pub generated_by: String,
    /// Model identity.
    pub model: Qwen35Identity,
    /// Tie verification.
    pub tie: TieReport,
    /// Every classified tensor.
    pub components: Vec<TowerTensor>,
    /// Totals per tower.
    pub towers: BTreeMap<String, TowerSummary>,
    /// Checkpoint totals.
    pub checkpoint: CheckpointTotals,
    /// The declared path.
    pub active_path: ActivePathReport,
    /// The certificate.
    pub certification: CertificationReport,
    /// Requested-vs-observed totals.
    pub expectations: Vec<ExpectationCheck>,
    /// Declarations the tensors contradict. A non-empty list sets `ok = false`.
    pub refusals: Vec<String>,
}

// ---------------------------------------------------------------------------
// config
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct RopeParameters {
    rope_theta: f64,
    #[serde(default)]
    partial_rotary_factor: Option<f64>,
    #[serde(default)]
    mrope_section: Vec<u64>,
    #[serde(default)]
    mrope_interleaved: bool,
}

#[derive(Debug, Deserialize)]
struct TextConfig {
    model_type: String,
    hidden_size: u64,
    intermediate_size: u64,
    num_hidden_layers: u64,
    num_attention_heads: u64,
    num_key_value_heads: u64,
    head_dim: u64,
    vocab_size: u64,
    #[serde(default)]
    layer_types: Vec<String>,
    #[serde(default)]
    full_attention_interval: Option<u64>,
    #[serde(default)]
    mlp_only_layers: Vec<u64>,
    #[serde(default)]
    linear_conv_kernel_dim: u64,
    linear_key_head_dim: u64,
    linear_num_key_heads: u64,
    linear_num_value_heads: u64,
    linear_value_head_dim: u64,
    rms_norm_eps: f64,
    #[serde(default)]
    attn_output_gate: bool,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
    #[serde(default)]
    mtp_num_hidden_layers: u64,
    #[serde(default)]
    mtp_use_dedicated_embeddings: bool,
    #[serde(default)]
    rope_parameters: Option<RopeParameters>,
}

#[derive(Debug, Deserialize)]
struct VisionConfig {
    depth: u64,
    hidden_size: u64,
    intermediate_size: u64,
    num_heads: u64,
    out_hidden_size: u64,
    patch_size: u64,
    spatial_merge_size: u64,
    temporal_patch_size: u64,
    in_channels: u64,
    num_position_embeddings: u64,
    #[serde(default)]
    hidden_act: String,
}

#[derive(Debug, Deserialize)]
struct Qwen35Config {
    model_type: String,
    #[serde(default)]
    architectures: Vec<String>,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
    text_config: TextConfig,
    #[serde(default)]
    vision_config: Option<VisionConfig>,
}

struct ExpectedV2 {
    tensor: String,
    tower: Tower,
    component: String,
    shape: Vec<u64>,
    layer: Option<u32>,
    ffn: bool,
}

fn product(shape: &[u64], tensor: &str) -> Result<u64> {
    let mut total = 1_u64;
    for dim in shape {
        total = total.checked_mul(*dim).ok_or_else(|| {
            ConvertError::inconsistent(format!("{tensor}: element count overflows u64"))
        })?;
    }
    Ok(total)
}

fn dtype_width(dtype: &DType) -> Result<u64> {
    match dtype {
        DType::F32 => Ok(4),
        DType::F16 | DType::BF16 => Ok(2),
        other => Err(ConvertError::inconsistent(format!(
            "accounting accepts f32/f16/bf16 tensors, saw {}",
            other.stable_name()
        ))),
    }
}

fn config_path(path: &Path) -> Result<PathBuf> {
    let metadata = std::fs::metadata(path)?;
    if metadata.is_dir() {
        Ok(path.join("config.json"))
    } else {
        path.parent()
            .map(|parent| parent.join("config.json"))
            .ok_or_else(|| {
                ConvertError::inconsistent(format!(
                    "{} has no parent directory to find config.json",
                    path.display()
                ))
            })
    }
}

/// Read the top-level `model_type` of a checkpoint without accounting it.
///
/// # Errors
///
/// Rejects a missing or unreadable `config.json` and a config without a string
/// `model_type`.
pub fn detect_model_type(path: &Path) -> Result<String> {
    let config_path = config_path(path)?;
    let bytes = std::fs::read(&config_path)?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| {
        ConvertError::inconsistent(format!(
            "{} is not readable JSON: {error}",
            config_path.display()
        ))
    })?;
    value
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| {
            ConvertError::inconsistent(format!(
                "{} has no string `model_type`",
                config_path.display()
            ))
        })
}

fn load_config(path: &Path) -> Result<(Qwen35Config, String)> {
    let config_path = config_path(path)?;
    let bytes = std::fs::read(&config_path)?;
    let config: Qwen35Config = serde_json::from_slice(&bytes).map_err(|error| {
        ConvertError::inconsistent(format!(
            "{} is not a readable Qwen3.5 config: {error}",
            config_path.display()
        ))
    })?;
    let mut builder = selvedge_core::Sha256Builder::new();
    builder.update(&bytes);
    Ok((config, builder.finish().to_hex()))
}

fn resolved_tie(config: &Qwen35Config) -> Result<bool> {
    let top = config.tie_word_embeddings;
    let nested = config.text_config.tie_word_embeddings;
    match (top, nested) {
        (Some(a), Some(b)) if a != b => Err(ConvertError::inconsistent(format!(
            "config declares tie_word_embeddings={a} at the top level and {b} inside \
             text_config; the auditor will not guess"
        ))),
        (Some(a), _) => Ok(a),
        (None, Some(b)) => Ok(b),
        (None, None) => Ok(false),
    }
}

fn text_facts(text: &TextConfig, attention_indices: &[u32]) -> Result<TextFacts> {
    let rope = text.rope_parameters.as_ref().ok_or_else(|| {
        ConvertError::inconsistent("text_config has no rope_parameters; RoPE shape is unknown")
    })?;
    let factor = rope.partial_rotary_factor.unwrap_or(1.0);
    if !(factor > 0.0 && factor <= 1.0) {
        return Err(ConvertError::inconsistent(format!(
            "partial_rotary_factor {factor} is outside (0, 1]"
        )));
    }
    let rotary_dim = ((text.head_dim as f64) * factor) as u64;
    if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) {
        return Err(ConvertError::inconsistent(format!(
            "rotary dimension {rotary_dim} is not a positive even number"
        )));
    }
    let layers = text.num_hidden_layers;
    if layers == 0 || text.hidden_size == 0 || text.intermediate_size == 0 || text.vocab_size == 0 {
        return Err(ConvertError::inconsistent(
            "text_config has a zero hidden/intermediate/layer/vocab dimension",
        ));
    }
    if text.head_dim == 0
        || text.num_attention_heads == 0
        || text.num_key_value_heads == 0
        || text.linear_num_key_heads == 0
        || text.linear_num_value_heads == 0
        || text.linear_key_head_dim == 0
        || text.linear_value_head_dim == 0
    {
        return Err(ConvertError::inconsistent(
            "text_config has a zero head dimension or head count",
        ));
    }
    Ok(TextFacts {
        model_type: text.model_type.clone(),
        hidden: text.hidden_size,
        intermediate: text.intermediate_size,
        layers,
        vocab: text.vocab_size,
        linear_layers: layers - attention_indices.len() as u64,
        attention_layers: attention_indices.len() as u64,
        attention_layer_indices: attention_indices.to_vec(),
        attention_heads: text.num_attention_heads,
        attention_kv_heads: text.num_key_value_heads,
        attention_head_dim: text.head_dim,
        attn_output_gate: text.attn_output_gate,
        linear_key_heads: text.linear_num_key_heads,
        linear_key_head_dim: text.linear_key_head_dim,
        linear_value_heads: text.linear_num_value_heads,
        linear_value_head_dim: text.linear_value_head_dim,
        conv_kernel: text.linear_conv_kernel_dim,
        rope_theta: rope.rope_theta,
        partial_rotary_factor: factor,
        rotary_dim,
        mrope_section: rope.mrope_section.clone(),
        mrope_interleaved: rope.mrope_interleaved,
        rms_norm_eps: text.rms_norm_eps,
        mtp_layers: text.mtp_num_hidden_layers,
        mtp_use_dedicated_embeddings: text.mtp_use_dedicated_embeddings,
        tie_word_embeddings: text.tie_word_embeddings.unwrap_or(false),
    })
}

/// Which layers are full attention, cross-checked against the interval.
fn attention_layers(text: &TextConfig) -> Result<Vec<u32>> {
    if !text.mlp_only_layers.is_empty() {
        return Err(ConvertError::inconsistent(format!(
            "text_config declares mlp_only_layers {:?}; this auditor has no verified \
             shape rule for a layer without attention",
            text.mlp_only_layers
        )));
    }
    if text.layer_types.len() as u64 != text.num_hidden_layers {
        return Err(ConvertError::inconsistent(format!(
            "text_config lists {} layer_types for {} layers",
            text.layer_types.len(),
            text.num_hidden_layers
        )));
    }
    let interval = text
        .full_attention_interval
        .ok_or_else(|| ConvertError::inconsistent("text_config has no full_attention_interval"))?;
    if interval == 0 {
        return Err(ConvertError::inconsistent(
            "text_config full_attention_interval is zero",
        ));
    }
    let mut indices = Vec::new();
    for (index, kind) in text.layer_types.iter().enumerate() {
        let index_u32 = u32::try_from(index)
            .map_err(|_| ConvertError::inconsistent("layer index does not fit in u32"))?;
        let declared_attention = kind == "full_attention";
        let interval_attention = (index_u32 + 1)
            % u32::try_from(interval).map_err(|_| {
                ConvertError::inconsistent("full_attention_interval does not fit in u32")
            })?
            == 0;
        if kind != "full_attention" && kind != "linear_attention" {
            return Err(ConvertError::inconsistent(format!(
                "layer {index} has unknown layer_type {kind:?}"
            )));
        }
        if declared_attention != interval_attention {
            return Err(ConvertError::inconsistent(format!(
                "layer {index} is {kind:?} but full_attention_interval {interval} implies \
                 {}",
                if interval_attention {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            )));
        }
        if declared_attention {
            indices.push(index_u32);
        }
    }
    Ok(indices)
}

fn attention_layer_tensors(prefix: &str, text: &TextFacts, layer: u32) -> Vec<ExpectedV2> {
    let hidden = text.hidden;
    let head_dim = text.attention_head_dim;
    let q_out = text.attention_heads * head_dim * if text.attn_output_gate { 2 } else { 1 };
    let kv_out = text.attention_kv_heads * head_dim;
    let rows: Vec<(&str, &str, Vec<u64>)> = vec![
        (
            "self_attn.q_proj.weight",
            "attention.q",
            vec![q_out, hidden],
        ),
        (
            "self_attn.k_proj.weight",
            "attention.k",
            vec![kv_out, hidden],
        ),
        (
            "self_attn.v_proj.weight",
            "attention.v",
            vec![kv_out, hidden],
        ),
        (
            "self_attn.o_proj.weight",
            "attention.o",
            vec![hidden, text.attention_heads * head_dim],
        ),
        (
            "self_attn.q_norm.weight",
            "attention.q_norm",
            vec![head_dim],
        ),
        (
            "self_attn.k_norm.weight",
            "attention.k_norm",
            vec![head_dim],
        ),
    ];
    let mut out = Vec::new();
    for (suffix, component, shape) in rows {
        out.push(ExpectedV2 {
            tensor: format!("{prefix}.{suffix}"),
            tower: Tower::Text,
            component: component.to_owned(),
            shape,
            layer: Some(layer),
            ffn: false,
        });
    }
    out
}

fn ffn_layer_tensors(prefix: &str, text: &TextFacts, layer: u32) -> Vec<ExpectedV2> {
    let hidden = text.hidden;
    let intermediate = text.intermediate;
    let rows: Vec<(&str, &str, Vec<u64>)> = vec![
        (
            "mlp.gate_proj.weight",
            "mlp.gate",
            vec![intermediate, hidden],
        ),
        ("mlp.up_proj.weight", "mlp.up", vec![intermediate, hidden]),
        (
            "mlp.down_proj.weight",
            "mlp.down",
            vec![hidden, intermediate],
        ),
    ];
    rows.into_iter()
        .map(|(suffix, component, shape)| ExpectedV2 {
            tensor: format!("{prefix}.{suffix}"),
            tower: Tower::Text,
            component: component.to_owned(),
            shape,
            layer: Some(layer),
            ffn: true,
        })
        .collect()
}

fn norm_layer_tensors(prefix: &str, text: &TextFacts, layer: u32) -> Vec<ExpectedV2> {
    let hidden = text.hidden;
    vec![
        ExpectedV2 {
            tensor: format!("{prefix}.input_layernorm.weight"),
            tower: Tower::Text,
            component: "norm".to_owned(),
            shape: vec![hidden],
            layer: Some(layer),
            ffn: false,
        },
        ExpectedV2 {
            tensor: format!("{prefix}.post_attention_layernorm.weight"),
            tower: Tower::Text,
            component: "norm".to_owned(),
            shape: vec![hidden],
            layer: Some(layer),
            ffn: false,
        },
    ]
}

fn expected_text(
    config: &Qwen35Config,
    facts: &TextFacts,
    attention: &[u32],
) -> Result<Vec<ExpectedV2>> {
    let text = &config.text_config;
    let hidden = facts.hidden;
    let mut out = vec![ExpectedV2 {
        tensor: "model.language_model.embed_tokens.weight".to_owned(),
        tower: Tower::Text,
        component: "embedding".to_owned(),
        shape: vec![facts.vocab, hidden],
        layer: None,
        ffn: false,
    }];
    let conv_dim = 2 * text.linear_num_key_heads * text.linear_key_head_dim
        + text.linear_num_value_heads * text.linear_value_head_dim;
    let value_width = text.linear_num_value_heads * text.linear_value_head_dim;
    for layer in 0..facts.layers {
        let layer_u32 = u32::try_from(layer)
            .map_err(|_| ConvertError::inconsistent("layer index does not fit in u32"))?;
        let prefix = format!("model.language_model.layers.{layer}");
        out.extend(norm_layer_tensors(&prefix, facts, layer_u32));
        if attention.contains(&layer_u32) {
            out.extend(attention_layer_tensors(&prefix, facts, layer_u32));
        } else {
            let rows: Vec<(&str, &str, Vec<u64>)> = vec![
                (
                    "linear_attn.in_proj_qkv.weight",
                    "linear_attention.in_proj_qkv",
                    vec![conv_dim, hidden],
                ),
                (
                    "linear_attn.in_proj_z.weight",
                    "linear_attention.in_proj_z",
                    vec![value_width, hidden],
                ),
                (
                    "linear_attn.in_proj_a.weight",
                    "linear_attention.in_proj_a",
                    vec![text.linear_num_value_heads, hidden],
                ),
                (
                    "linear_attn.in_proj_b.weight",
                    "linear_attention.in_proj_b",
                    vec![text.linear_num_value_heads, hidden],
                ),
                (
                    "linear_attn.conv1d.weight",
                    "linear_attention.conv1d",
                    vec![conv_dim, 1, text.linear_conv_kernel_dim],
                ),
                (
                    "linear_attn.A_log",
                    "linear_attention.a_log",
                    vec![text.linear_num_value_heads],
                ),
                (
                    "linear_attn.dt_bias",
                    "linear_attention.dt_bias",
                    vec![text.linear_num_value_heads],
                ),
                (
                    "linear_attn.norm.weight",
                    "linear_attention.norm",
                    vec![text.linear_value_head_dim],
                ),
                (
                    "linear_attn.out_proj.weight",
                    "linear_attention.out_proj",
                    vec![hidden, value_width],
                ),
            ];
            for (suffix, component, shape) in rows {
                out.push(ExpectedV2 {
                    tensor: format!("{prefix}.{suffix}"),
                    tower: Tower::Text,
                    component: component.to_owned(),
                    shape,
                    layer: Some(layer_u32),
                    ffn: false,
                });
            }
        }
        out.extend(ffn_layer_tensors(&prefix, facts, layer_u32));
    }
    out.push(ExpectedV2 {
        tensor: "model.language_model.norm.weight".to_owned(),
        tower: Tower::Text,
        component: "norm".to_owned(),
        shape: vec![hidden],
        layer: None,
        ffn: false,
    });
    Ok(out)
}

fn expected_mtp(_config: &Qwen35Config, facts: &TextFacts) -> Result<Vec<ExpectedV2>> {
    let count = facts.mtp_layers;
    if count == 0 {
        return Ok(Vec::new());
    }
    if facts.mtp_use_dedicated_embeddings {
        return Err(ConvertError::inconsistent(
            "text_config declares mtp_use_dedicated_embeddings=true; this auditor has no \
             verified shape rule for a dedicated MTP embedding and refuses to guess",
        ));
    }
    let hidden = facts.hidden;
    let mut out = vec![
        ExpectedV2 {
            tensor: "mtp.fc.weight".to_owned(),
            tower: Tower::Mtp,
            component: "mtp.fc".to_owned(),
            shape: vec![hidden, 2 * hidden],
            layer: None,
            ffn: false,
        },
        ExpectedV2 {
            tensor: "mtp.pre_fc_norm_embedding.weight".to_owned(),
            tower: Tower::Mtp,
            component: "norm".to_owned(),
            shape: vec![hidden],
            layer: None,
            ffn: false,
        },
        ExpectedV2 {
            tensor: "mtp.pre_fc_norm_hidden.weight".to_owned(),
            tower: Tower::Mtp,
            component: "norm".to_owned(),
            shape: vec![hidden],
            layer: None,
            ffn: false,
        },
        ExpectedV2 {
            tensor: "mtp.norm.weight".to_owned(),
            tower: Tower::Mtp,
            component: "norm".to_owned(),
            shape: vec![hidden],
            layer: None,
            ffn: false,
        },
    ];
    for layer in 0..count {
        let layer_u32 = u32::try_from(layer)
            .map_err(|_| ConvertError::inconsistent("MTP layer index does not fit in u32"))?;
        let prefix = format!("mtp.layers.{layer}");
        for mut item in norm_layer_tensors(&prefix, facts, layer_u32) {
            item.tower = Tower::Mtp;
            out.push(item);
        }
        for mut item in attention_layer_tensors(&prefix, facts, layer_u32) {
            item.tower = Tower::Mtp;
            out.push(item);
        }
        for mut item in ffn_layer_tensors(&prefix, facts, layer_u32) {
            item.tower = Tower::Mtp;
            out.push(item);
        }
    }
    Ok(out)
}

fn expected_vision(vision: &VisionConfig) -> Vec<ExpectedV2> {
    let hidden = vision.hidden_size;
    let intermediate = vision.intermediate_size;
    let merged = hidden * vision.spatial_merge_size * vision.spatial_merge_size;
    let mut out = vec![
        ExpectedV2 {
            tensor: "model.visual.patch_embed.proj.weight".to_owned(),
            tower: Tower::Vision,
            component: "patch_embed.weight".to_owned(),
            shape: vec![
                hidden,
                vision.in_channels,
                vision.temporal_patch_size,
                vision.patch_size,
                vision.patch_size,
            ],
            layer: None,
            ffn: false,
        },
        ExpectedV2 {
            tensor: "model.visual.patch_embed.proj.bias".to_owned(),
            tower: Tower::Vision,
            component: "patch_embed.bias".to_owned(),
            shape: vec![hidden],
            layer: None,
            ffn: false,
        },
        ExpectedV2 {
            tensor: "model.visual.pos_embed.weight".to_owned(),
            tower: Tower::Vision,
            component: "position_embedding".to_owned(),
            shape: vec![vision.num_position_embeddings, hidden],
            layer: None,
            ffn: false,
        },
        ExpectedV2 {
            tensor: "model.visual.merger.norm.weight".to_owned(),
            tower: Tower::Vision,
            component: "merger.norm".to_owned(),
            shape: vec![hidden],
            layer: None,
            ffn: false,
        },
        ExpectedV2 {
            tensor: "model.visual.merger.norm.bias".to_owned(),
            tower: Tower::Vision,
            component: "merger.norm".to_owned(),
            shape: vec![hidden],
            layer: None,
            ffn: false,
        },
        ExpectedV2 {
            tensor: "model.visual.merger.linear_fc1.weight".to_owned(),
            tower: Tower::Vision,
            component: "merger.fc1".to_owned(),
            shape: vec![merged, merged],
            layer: None,
            ffn: false,
        },
        ExpectedV2 {
            tensor: "model.visual.merger.linear_fc1.bias".to_owned(),
            tower: Tower::Vision,
            component: "merger.fc1".to_owned(),
            shape: vec![merged],
            layer: None,
            ffn: false,
        },
        ExpectedV2 {
            tensor: "model.visual.merger.linear_fc2.weight".to_owned(),
            tower: Tower::Vision,
            component: "merger.fc2".to_owned(),
            shape: vec![vision.out_hidden_size, merged],
            layer: None,
            ffn: false,
        },
        ExpectedV2 {
            tensor: "model.visual.merger.linear_fc2.bias".to_owned(),
            tower: Tower::Vision,
            component: "merger.fc2".to_owned(),
            shape: vec![vision.out_hidden_size],
            layer: None,
            ffn: false,
        },
    ];
    for block in 0..vision.depth {
        let prefix = format!("model.visual.blocks.{block}");
        let rows: Vec<(String, &str, Vec<u64>)> = vec![
            (format!("{prefix}.norm1.weight"), "norm", vec![hidden]),
            (format!("{prefix}.norm1.bias"), "norm", vec![hidden]),
            (format!("{prefix}.norm2.weight"), "norm", vec![hidden]),
            (format!("{prefix}.norm2.bias"), "norm", vec![hidden]),
            (
                format!("{prefix}.attn.qkv.weight"),
                "attn.qkv",
                vec![3 * hidden, hidden],
            ),
            (
                format!("{prefix}.attn.qkv.bias"),
                "attn.qkv",
                vec![3 * hidden],
            ),
            (
                format!("{prefix}.attn.proj.weight"),
                "attn.proj",
                vec![hidden, hidden],
            ),
            (
                format!("{prefix}.attn.proj.bias"),
                "attn.proj",
                vec![hidden],
            ),
            (
                format!("{prefix}.mlp.linear_fc1.weight"),
                "mlp.fc1",
                vec![intermediate, hidden],
            ),
            (
                format!("{prefix}.mlp.linear_fc1.bias"),
                "mlp.fc1",
                vec![intermediate],
            ),
            (
                format!("{prefix}.mlp.linear_fc2.weight"),
                "mlp.fc2",
                vec![hidden, intermediate],
            ),
            (
                format!("{prefix}.mlp.linear_fc2.bias"),
                "mlp.fc2",
                vec![hidden],
            ),
        ];
        for (tensor, component, shape) in rows {
            out.push(ExpectedV2 {
                tensor,
                tower: Tower::Vision,
                component: component.to_owned(),
                shape,
                layer: None,
                ffn: false,
            });
        }
    }
    out
}

fn count_tensor_v2(source: &impl TensorSource, want: &ExpectedV2) -> Result<TowerTensor> {
    let descriptor: TensorDescriptor = source.descriptor(&want.tensor)?;
    if descriptor.shape != want.shape {
        return Err(ConvertError::inconsistent(format!(
            "{} has shape {:?}, expected {:?} from config.json",
            want.tensor, descriptor.shape, want.shape
        )));
    }
    let elements = product(&descriptor.shape, &want.tensor)?;
    if elements != descriptor.elements {
        return Err(ConvertError::inconsistent(format!(
            "{} declares {} elements but its shape implies {elements}",
            want.tensor, descriptor.elements
        )));
    }
    let expected_bytes = elements
        .checked_mul(dtype_width(&descriptor.dtype)?)
        .ok_or_else(|| {
            ConvertError::inconsistent(format!("{}: byte count overflows", want.tensor))
        })?;
    if descriptor.byte_len != expected_bytes {
        return Err(ConvertError::inconsistent(format!(
            "{} declares {} bytes, expected {expected_bytes} for {elements} elements of {}",
            want.tensor,
            descriptor.byte_len,
            descriptor.dtype.stable_name()
        )));
    }
    Ok(TowerTensor {
        tensor: want.tensor.clone(),
        tower: want.tower.as_str().to_owned(),
        component: want.component.clone(),
        shape: descriptor.shape.clone(),
        elements,
        bytes: descriptor.byte_len,
        dtype: descriptor.dtype.stable_name().to_owned(),
        layer: want.layer,
        ffn: want.ffn,
    })
}

fn tensors_identical(
    source: &mut impl TensorSource,
    left: &TensorDescriptor,
    right: &TensorDescriptor,
) -> Result<bool> {
    if left.shape != right.shape || left.dtype != right.dtype {
        return Ok(false);
    }
    if left.rank() != 2 || right.rank() != 2 {
        return Ok(false);
    }
    let rows = left.shape.first().copied().unwrap_or(0);
    let cols = left.shape.get(1).copied().unwrap_or(0);
    let mut start = 0_u64;
    while start < rows {
        let end = (start + TIE_CHECK_ROWS).min(rows);
        let left_rows = source.read_region_f32(&left.key, start..end, 0..cols)?;
        let right_rows = source.read_region_f32(&right.key, start..end, 0..cols)?;
        if left_rows.len() != right_rows.len() {
            return Ok(false);
        }
        for (a, b) in left_rows.iter().zip(right_rows.iter()) {
            if a.to_bits() != b.to_bits() {
                return Ok(false);
            }
        }
        start = end;
    }
    Ok(true)
}

/// Read `metadata.total_size` from `model.safetensors.index.json` if present.
fn index_total_bytes(path: &Path) -> Result<Option<u64>> {
    let metadata = std::fs::metadata(path)?;
    let dir = if metadata.is_dir() {
        path.to_path_buf()
    } else {
        match path.parent() {
            Some(parent) => parent.to_path_buf(),
            None => return Ok(None),
        }
    };
    let index_path = dir.join("model.safetensors.index.json");
    if !index_path.is_file() {
        return Ok(None);
    }
    let bytes = std::fs::read(&index_path)?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| {
        ConvertError::inconsistent(format!(
            "{} is not readable JSON: {error}",
            index_path.display()
        ))
    })?;
    let total = value
        .get("metadata")
        .and_then(|meta| meta.get("total_size"))
        .and_then(serde_json::Value::as_u64);
    match total {
        Some(total) => Ok(Some(total)),
        None => Err(ConvertError::inconsistent(format!(
            "{} has no metadata.total_size; the Hub total cannot be checked",
            index_path.display()
        ))),
    }
}

fn tower_summary(
    tower: Tower,
    accounts: &[TowerTensor],
    unique_elements: u64,
    executed: bool,
) -> TowerSummary {
    let mut summary = TowerSummary {
        tower: tower.as_str().to_owned(),
        tensors: 0,
        checkpoint_elements: 0,
        unique_elements,
        bytes: 0,
        ffn_elements: 0,
        always_active_elements: 0,
        executed_by_declared_path: executed,
    };
    for account in accounts {
        summary.tensors += 1;
        summary.checkpoint_elements += account.elements;
        summary.bytes += account.bytes;
        if account.ffn {
            summary.ffn_elements += account.elements;
            // FFN is the removable set, so it is not always active unless the
            // whole tower runs dense; the text path's own always-active set
            // excludes it by construction.
        } else {
            summary.always_active_elements += account.elements;
        }
    }
    summary
}

fn certify(
    options: &Qwen35Options,
    path: &ActivePath,
    towers: &BTreeMap<String, TowerSummary>,
    active_unique: u64,
    tie: &TieReport,
) -> Result<CertificationReport> {
    let ceiling = options.certify_ceiling.unwrap_or(options.ceiling);
    let headroom = i64::try_from(ceiling)
        .map_err(|_| ConvertError::inconsistent("ceiling does not fit in i64"))?
        - i64::try_from(active_unique)
            .map_err(|_| ConvertError::inconsistent("active count does not fit in i64"))?;
    let certified = active_unique <= ceiling;
    let mut arithmetic = Vec::new();
    let text = towers
        .get(Tower::Text.as_str())
        .ok_or_else(|| ConvertError::inconsistent("the text tower summary is missing"))?;
    arithmetic.push(CertificationTerm {
        label: "text tower unique (tie-aware)".to_owned(),
        elements: text.unique_elements,
        sign: "+".to_owned(),
    });
    if path.towers().contains(&Tower::Mtp) {
        if let Some(mtp) = towers.get(Tower::Mtp.as_str()) {
            arithmetic.push(CertificationTerm {
                label: "mtp head (executed by the declared path)".to_owned(),
                elements: mtp.checkpoint_elements,
                sign: "+".to_owned(),
            });
        }
    } else if let Some(mtp) = towers.get(Tower::Mtp.as_str()) {
        arithmetic.push(CertificationTerm {
            label: "mtp head (excluded by the declared path)".to_owned(),
            elements: mtp.checkpoint_elements,
            sign: "-".to_owned(),
        });
    }
    if let Some(vision) = towers.get(Tower::Vision.as_str()) {
        arithmetic.push(CertificationTerm {
            label: "vision tower (never read by this path)".to_owned(),
            elements: vision.checkpoint_elements,
            sign: "-".to_owned(),
        });
    }
    let explanation = if certified {
        format!(
            "the declared path {} reads {active_unique} unique active elements, which is at or \
             below the {ceiling} ceiling; headroom {headroom}. The tie is {}",
            path.as_str(),
            if tie.treated_as_shared {
                "verified (the embedding is the output projection)"
            } else {
                "not claimed"
            }
        )
    } else {
        format!(
            "the declared path {} reads {active_unique} unique active elements, above the \
             {ceiling} ceiling by {}. No sharding of this checkpoint makes the declaration \
             compliant; the gap is structural",
            path.as_str(),
            -headroom
        )
    };
    Ok(CertificationReport {
        requested: options.certify_ceiling.is_some(),
        ceiling,
        active_unique_elements: active_unique,
        headroom_elements: headroom,
        verdict: if certified {
            "certified".to_owned()
        } else {
            "exceeds_ceiling".to_owned()
        },
        arithmetic,
        explanation,
    })
}

/// Account a Qwen3.5 checkpoint, including its towers and a declared path.
///
/// # Errors
///
/// Rejects a missing or inconsistent config, an unknown `model_type`, a tensor
/// set the auditor cannot classify, a shape/dtype/byte mismatch, a tie the bytes
/// contradict, a Hub total that disagrees with the header sum, and every reader
/// failure. A *declaration* the tensors contradict is reported in
/// [`Qwen35AccountingReport::refusals`] with `ok = false` rather than as an
/// error, so the arithmetic that led to the refusal is still printed.
pub fn account_model_qwen35(
    path: &Path,
    options: &Qwen35Options,
) -> Result<Qwen35AccountingReport> {
    options.validate()?;
    let (config, config_sha256) = load_config(path)?;
    if config.model_type != "qwen3_5" {
        return Err(ConvertError::inconsistent(format!(
            "the qwen3.5 auditor is for model_type \"qwen3_5\", config says {:?}",
            config.model_type
        )));
    }
    let attention = attention_layers(&config.text_config)?;
    let facts = text_facts(&config.text_config, &attention)?;
    if facts.model_type != "qwen3_5_text" {
        return Err(ConvertError::inconsistent(format!(
            "text_config.model_type is {:?}; expected \"qwen3_5_text\"",
            facts.model_type
        )));
    }
    let config_tied = resolved_tie(&config)?;

    let mut expected = expected_text(&config, &facts, &attention)?;
    expected.extend(expected_mtp(&config, &facts)?);
    if let Some(vision) = &config.vision_config {
        expected.extend(expected_vision(vision));
    }
    // The declared tie does not change which tensors are *claimed* at this
    // point; `lm_head.weight` presence is handled below so the claim can be
    // checked against the bytes.
    let mut source = CheckpointSource::open(path)?;
    let keys: BTreeSet<String> = source.tensor_keys().into_iter().collect();
    let mut expected_keys: BTreeSet<String> =
        expected.iter().map(|item| item.tensor.clone()).collect();
    let lm_head_key = "lm_head.weight".to_owned();
    let lm_head_present = keys.contains(&lm_head_key);
    if lm_head_present {
        expected_keys.insert(lm_head_key.clone());
    }
    let unexpected: Vec<String> = keys.difference(&expected_keys).cloned().collect();
    if !unexpected.is_empty() {
        return Err(ConvertError::inconsistent(format!(
            "checkpoint contains {} tensor(s) the accounting does not classify: {}",
            unexpected.len(),
            unexpected.join(", ")
        )));
    }
    let missing: Vec<String> = expected_keys.difference(&keys).cloned().collect();
    if !missing.is_empty() {
        return Err(ConvertError::inconsistent(format!(
            "checkpoint is missing {} expected tensor(s): {}",
            missing.len(),
            missing.join(", ")
        )));
    }

    let mut accounts = Vec::with_capacity(expected.len() + 1);
    for want in &expected {
        accounts.push(count_tensor_v2(&source, want)?);
    }

    // ---- the tie, verified against bytes -------------------------------
    let mut refusals: Vec<String> = Vec::new();
    let mut tie = TieReport {
        config_tie_word_embeddings: config_tied,
        lm_head_present,
        lm_head_identical_to_embedding: false,
        treated_as_shared: false,
        compared_elements: 0,
    };
    if lm_head_present {
        let embed = source.descriptor("model.language_model.embed_tokens.weight")?;
        let head = source.descriptor(&lm_head_key)?;
        if head.shape != embed.shape {
            return Err(ConvertError::inconsistent(format!(
                "lm_head.weight has shape {:?} but the embedding has {:?}",
                head.shape, embed.shape
            )));
        }
        let identical = tensors_identical(&mut source, &embed, &head)?;
        tie.lm_head_identical_to_embedding = identical;
        tie.compared_elements = head.elements;
        accounts.push(count_tensor_v2(
            &source,
            &ExpectedV2 {
                tensor: lm_head_key.clone(),
                tower: Tower::Text,
                component: "lm_head".to_owned(),
                shape: head.shape.clone(),
                layer: None,
                ffn: false,
            },
        )?);
        if config_tied && !identical {
            return Err(ConvertError::inconsistent(
                "config declares tie_word_embeddings=true but lm_head.weight is not byte-identical \
                 to model.language_model.embed_tokens.weight",
            ));
        }
        tie.treated_as_shared = identical && config_tied;
    } else {
        // No lm_head tensor: the embedding is the output projection, but only
        // if the config says so. Absence alone is not a tie.
        tie.treated_as_shared = config_tied;
        tie.lm_head_identical_to_embedding = config_tied;
    }
    match options.tie_assertion {
        TieAssertion::FromConfig => {}
        TieAssertion::Tied => {
            if lm_head_present && !tie.lm_head_identical_to_embedding {
                refusals.push(
                    "the caller declared the weights tied, but lm_head.weight is present and its \
                     bytes differ from model.language_model.embed_tokens.weight; a tied claim here \
                     would under-count the checkpoint"
                        .to_owned(),
                );
            }
            if !config_tied {
                refusals.push(
                    "the caller declared the weights tied, but config.json says \
                     tie_word_embeddings=false; the auditor will not override the config"
                        .to_owned(),
                );
            }
        }
        TieAssertion::Untied => {
            if !lm_head_present {
                refusals.push(
                    "the caller declared the weights untied, but no lm_head.weight exists; the \
                     only output projection is the embedding"
                        .to_owned(),
                );
            }
        }
    }

    accounts.sort_by(|a, b| a.tensor.cmp(&b.tensor));

    // ---- per-tower totals ----------------------------------------------
    let mut by_dtype: BTreeMap<String, DtypeTotal> = BTreeMap::new();
    let mut checkpoint_elements = 0_u64;
    let mut checkpoint_bytes = 0_u64;
    for account in &accounts {
        checkpoint_elements = checkpoint_elements
            .checked_add(account.elements)
            .ok_or_else(|| ConvertError::inconsistent("element count overflows u64"))?;
        checkpoint_bytes = checkpoint_bytes
            .checked_add(account.bytes)
            .ok_or_else(|| ConvertError::inconsistent("byte count overflows u64"))?;
        let entry = by_dtype.entry(account.dtype.clone()).or_insert(DtypeTotal {
            dtype: account.dtype.clone(),
            tensors: 0,
            elements: 0,
            bytes: 0,
        });
        entry.tensors += 1;
        entry.elements += account.elements;
        entry.bytes += account.bytes;
    }

    let text_accounts: Vec<TowerTensor> = accounts
        .iter()
        .filter(|a| a.tower == Tower::Text.as_str())
        .cloned()
        .collect();
    let vision_accounts: Vec<TowerTensor> = accounts
        .iter()
        .filter(|a| a.tower == Tower::Vision.as_str())
        .cloned()
        .collect();
    let mtp_accounts: Vec<TowerTensor> = accounts
        .iter()
        .filter(|a| a.tower == Tower::Mtp.as_str())
        .cloned()
        .collect();

    let text_duplicate = if tie.treated_as_shared {
        text_accounts
            .iter()
            .filter(|a| a.component == "lm_head")
            .map(|a| a.elements)
            .sum::<u64>()
    } else {
        0
    };
    let text_checkpoint: u64 = text_accounts.iter().map(|a| a.elements).sum();
    let text_unique = text_checkpoint
        .checked_sub(text_duplicate)
        .ok_or_else(|| ConvertError::inconsistent("tie subtraction underflows"))?;
    let vision_checkpoint: u64 = vision_accounts.iter().map(|a| a.elements).sum();
    let mtp_checkpoint: u64 = mtp_accounts.iter().map(|a| a.elements).sum();
    if text_checkpoint + vision_checkpoint + mtp_checkpoint != checkpoint_elements {
        return Err(ConvertError::inconsistent(
            "tower element sums do not add up to the checkpoint total",
        ));
    }

    let path_decl = options.active_path;
    let mut towers: BTreeMap<String, TowerSummary> = BTreeMap::new();
    towers.insert(
        Tower::Text.as_str().to_owned(),
        tower_summary(
            Tower::Text,
            &text_accounts,
            text_unique,
            path_decl.towers().contains(&Tower::Text),
        ),
    );
    if config.vision_config.is_some() {
        towers.insert(
            Tower::Vision.as_str().to_owned(),
            tower_summary(Tower::Vision, &vision_accounts, vision_checkpoint, false),
        );
    }
    if facts.mtp_layers > 0 {
        towers.insert(
            Tower::Mtp.as_str().to_owned(),
            tower_summary(
                Tower::Mtp,
                &mtp_accounts,
                mtp_checkpoint,
                path_decl.towers().contains(&Tower::Mtp),
            ),
        );
    }

    // ---- the declared path, checked against the tensors ----------------
    let mut active_unique = 0_u64;
    let mut executed_towers = Vec::new();
    let mut excluded_towers = Vec::new();
    let mut excluded_elements = Vec::new();
    for tower in path_decl.towers() {
        let Some(summary) = towers.get(tower.as_str()) else {
            refusals.push(format!(
                "the declared path {} executes the {} tower, but the checkpoint contains no such \
                 weights; the declaration describes an execution the bytes cannot support",
                path_decl.as_str(),
                tower.as_str()
            ));
            continue;
        };
        if summary.tensors == 0 {
            refusals.push(format!(
                "the declared path {} executes the {} tower, but the checkpoint contains no {} \
                 weights; the declaration describes an execution the bytes cannot support",
                path_decl.as_str(),
                tower.as_str(),
                tower.as_str()
            ));
        }
        active_unique = active_unique
            .checked_add(summary.unique_elements)
            .ok_or_else(|| ConvertError::inconsistent("active count overflows u64"))?;
        executed_towers.push(tower.as_str().to_owned());
    }
    for tower in [Tower::Text, Tower::Vision, Tower::Mtp] {
        if path_decl.towers().contains(&tower) {
            continue;
        }
        if let Some(summary) = towers.get(tower.as_str()) {
            excluded_towers.push(tower.as_str().to_owned());
            excluded_elements.push(ExcludedTower {
                tower: tower.as_str().to_owned(),
                elements: summary.checkpoint_elements,
                reason: match tower {
                    Tower::Vision => {
                        "a text-only execution never reads the vision encoder".to_owned()
                    }
                    Tower::Mtp => {
                        "the MTP head executes only when multi-token drafting is enabled".to_owned()
                    }
                    Tower::Text => "the language model is always executed".to_owned(),
                },
            });
        }
    }

    // Namespace disjointness is computed, not assumed: a text-path tensor that
    // lives under an excluded namespace would make the declaration a lie.
    let text_keys: BTreeSet<&str> = text_accounts
        .iter()
        .map(|account| account.tensor.as_str())
        .collect();
    for tower in [Tower::Vision, Tower::Mtp] {
        if path_decl.towers().contains(&tower) {
            continue;
        }
        for key in &text_keys {
            if key.starts_with(tower.namespace_prefix()) {
                refusals.push(format!(
                    "the declared path {} excludes the {} tower, but the text path reads {key}, \
                     which lives in that tower's namespace",
                    path_decl.as_str(),
                    tower.as_str()
                ));
            }
        }
    }

    // Cross-tower reads: with mtp_use_dedicated_embeddings=false the MTP head
    // reads the language model's embedding and the tied output projection.
    let mut shared_reads = Vec::new();
    if !facts.mtp_use_dedicated_embeddings {
        let embedding = text_accounts
            .iter()
            .find(|account| account.component == "embedding")
            .ok_or_else(|| ConvertError::inconsistent("the text tower has no embedding tensor"))?;
        shared_reads.push(SharedRead {
            tensor: embedding.tensor.clone(),
            towers: vec![
                Tower::Text.as_str().to_owned(),
                Tower::Mtp.as_str().to_owned(),
            ],
            reason: "mtp_use_dedicated_embeddings is false, so the MTP head reads the language \
                     model's token embedding"
                .to_owned(),
        });
        if path_decl.towers().contains(&Tower::Mtp) {
            let has_mtp_weights = towers
                .get(Tower::Mtp.as_str())
                .is_some_and(|summary| summary.tensors > 0);
            if !has_mtp_weights {
                refusals.push(
                    "the declared path executes MTP, which by config shares the language model's \
                     embedding, but the checkpoint has no MTP weights to execute"
                        .to_owned(),
                );
            }
        }
    }

    let active_path_report = ActivePathReport {
        declared: path_decl.as_str().to_owned(),
        executed_towers,
        excluded_towers,
        unique_active_elements: active_unique,
        excluded_elements,
        shared_reads,
        verdict: if refusals.is_empty() {
            "consistent".to_owned()
        } else {
            "inconsistent".to_owned()
        },
    };

    // ---- checkpoint totals and the Hub cross-check ---------------------
    let index_total = index_total_bytes(path)?;
    let index_matches = index_total.map(|total| total == checkpoint_bytes);
    if index_matches == Some(false) {
        return Err(ConvertError::inconsistent(format!(
            "the Hub's model.safetensors.index.json declares {} total bytes, but the header \
             arithmetic gives {checkpoint_bytes}; refusing to report a count that disagrees with \
             the Hub",
            index_total.unwrap_or(0)
        )));
    }

    let unique_checkpoint = if tie.treated_as_shared {
        checkpoint_elements
            .checked_sub(text_duplicate)
            .ok_or_else(|| ConvertError::inconsistent("tie subtraction underflows"))?
    } else {
        checkpoint_elements
    };

    let mut expectations = Vec::new();
    for (name, expected_elements) in &options.expect_totals {
        let observed = match name.as_str() {
            "text" | "text-tower" => text_unique,
            "text-checkpoint" => text_checkpoint,
            "vision" => vision_checkpoint,
            "mtp" => mtp_checkpoint,
            "checkpoint" => checkpoint_elements,
            "checkpoint-unique" => unique_checkpoint,
            "active" => active_unique,
            other => {
                return Err(ConvertError::plan(format!(
                    "unknown expectation name {other:?}; accepted: text, text-checkpoint, vision, \
                     mtp, checkpoint, checkpoint-unique, active"
                )));
            }
        };
        let matches = observed == *expected_elements;
        if !matches {
            refusals.push(format!(
                "expected {name} = {expected_elements} elements, observed {observed}"
            ));
        }
        expectations.push(ExpectationCheck {
            name: name.clone(),
            expected: *expected_elements,
            observed,
            matches,
        });
    }

    let certification = certify(options, &path_decl, &towers, active_unique, &tie)?;
    if certification.verdict == "exceeds_ceiling" {
        refusals.push(format!(
            "the declared path reads {} unique active elements, above the {} ceiling",
            active_unique, certification.ceiling
        ));
    }

    let vision_facts = config.vision_config.as_ref().map(|vision| VisionFacts {
        depth: vision.depth,
        hidden: vision.hidden_size,
        intermediate: vision.intermediate_size,
        heads: vision.num_heads,
        out_hidden: vision.out_hidden_size,
        patch_size: vision.patch_size,
        spatial_merge_size: vision.spatial_merge_size,
        temporal_patch_size: vision.temporal_patch_size,
        in_channels: vision.in_channels,
        position_embeddings: vision.num_position_embeddings,
        hidden_act: vision.hidden_act.clone(),
    });

    let checkpoint_files = source.relative_paths().len().max(1) as u64;
    let digest = if options.compute_digest {
        Some(
            crate::convert::model_digest_for_path(path)?
                .digest()
                .to_hex(),
        )
    } else {
        None
    };

    Ok(Qwen35AccountingReport {
        schema: QWEN35_SCHEMA.to_owned(),
        ok: refusals.is_empty(),
        stamp: options.stamp.clone(),
        revision: options.revision.clone(),
        generated_by: format!("selvedge-convert {}", env!("CARGO_PKG_VERSION")),
        model: Qwen35Identity {
            path: path.display().to_string(),
            model_type: config.model_type.clone(),
            architectures: config.architectures.clone(),
            config_sha256,
            checkpoint_digest: digest,
            checkpoint_files,
            text: facts,
            vision: vision_facts,
        },
        tie,
        components: accounts,
        towers,
        checkpoint: CheckpointTotals {
            tensors: expected_keys.len() as u64,
            tensor_elements: checkpoint_elements,
            tensor_bytes: checkpoint_bytes,
            unique_elements: unique_checkpoint,
            index_total_bytes: index_total,
            index_total_matches: index_matches,
            by_dtype: by_dtype.into_values().collect(),
        },
        active_path: active_path_report,
        certification,
        expectations,
        refusals,
    })
}
