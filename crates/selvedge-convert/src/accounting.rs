//! Exact parameter and FLOP accounting for a real Qwen3 checkpoint.
//!
//! The accounting reads only safetensors headers for element counts, so the
//! counts are exact for the bytes on disk rather than estimated from a model
//! card. Two facts cannot be read from a header, so they are measured:
//!
//! - whether `lm_head.weight` is byte-identical to
//!   `model.embed_tokens.weight` (the tie is verified, not trusted);
//! - the exact component membership of every tensor (an unexpected tensor is a
//!   hard error, so no learned element is silently left out of the totals).
//!
//! Parameter identities are counted once per unique element. The embedding and
//! output projection are counted as whole matrices: a tied pair counts once,
//! and an untied pair counts twice because the model reads two distinct
//! learned tensors. FLOPs are arithmetic estimates from documented formulas;
//! no FLOP number is measured, and the report says so.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use selvedge_core::DType;
use serde::{Deserialize, Serialize};

use crate::error::{ConvertError, Result};
use crate::safetensors::CheckpointSource;
use crate::tensor_source::{TensorDescriptor, TensorSource};

/// The strict unique-active-parameter ceiling from the mandate.
pub const DEFAULT_ACTIVE_CEILING: u64 = 1_000_000_000;

/// Default group count used for per-group FLOP and budget arithmetic.
pub const DEFAULT_GROUP_COUNT: u32 = 16;

/// Row chunk used when comparing the tied matrices. Bounded to keep the
/// comparison inside a fixed allocation regardless of vocabulary size.
const TIE_CHECK_ROWS: u64 = 8_192;

/// Options for [`account_model`].
#[derive(Clone, Debug)]
pub struct AccountingOptions {
    /// Unique-active-parameter ceiling.
    pub ceiling: u64,
    /// Group count used for per-group arithmetic.
    pub group_count: u32,
    /// Context lengths reported for the attention core FLOP estimate.
    pub reference_contexts: Vec<u64>,
    /// Include the streaming model digest (hashes every checkpoint file).
    pub compute_digest: bool,
    /// Receipt stamp, recorded verbatim.
    pub stamp: Option<String>,
    /// Git revision, recorded verbatim.
    pub revision: Option<String>,
}

impl Default for AccountingOptions {
    fn default() -> Self {
        Self {
            ceiling: DEFAULT_ACTIVE_CEILING,
            group_count: DEFAULT_GROUP_COUNT,
            reference_contexts: vec![1, 2_048, 40_960],
            compute_digest: true,
            stamp: None,
            revision: None,
        }
    }
}

impl AccountingOptions {
    /// Validate the options.
    ///
    /// # Errors
    ///
    /// Rejects a zero ceiling, zero groups, zero contexts and an empty context
    /// list.
    pub fn validate(&self) -> Result<()> {
        if self.ceiling == 0 {
            return Err(ConvertError::plan("accounting ceiling must be non-zero"));
        }
        if self.group_count == 0 {
            return Err(ConvertError::plan("group count must be non-zero"));
        }
        if self.reference_contexts.is_empty() || self.reference_contexts.contains(&0) {
            return Err(ConvertError::plan(
                "reference context lengths must be non-empty and non-zero",
            ));
        }
        Ok(())
    }
}

/// One tensor's exact contribution.
#[derive(Clone, Debug, Serialize)]
pub struct TensorAccount {
    /// Checkpoint tensor key.
    pub tensor: String,
    /// Component label (`attention.q`, `mlp.gate`, `norm`, ...).
    pub component: String,
    /// Logical shape.
    pub shape: Vec<u64>,
    /// Exact logical element count.
    pub elements: u64,
    /// Encoded bytes in the checkpoint.
    pub bytes: u64,
    /// Stable dtype name.
    pub dtype: String,
    /// Layer index when the tensor belongs to a transformer block.
    pub layer: Option<u32>,
    /// Whether the tensor is executed for every token.
    pub always_active: bool,
    /// Whether the tensor is part of the removable FFN.
    pub ffn: bool,
}

/// Category totals. Every category is a sum over unique tensor keys.
#[derive(Clone, Debug, Default, Serialize)]
pub struct CategoryTotals {
    /// Token embedding matrix.
    pub embedding: u64,
    /// Attention projections plus q/k RMSNorm weights.
    pub attention: u64,
    /// Layer and final RMSNorm weights outside the attention block.
    pub norm: u64,
    /// SwiGLU gate/up/down tensors.
    pub ffn: u64,
    /// Output projection, counted separately even when tied.
    pub lm_head: u64,
}

/// Verified tie state.
#[derive(Clone, Debug, Serialize)]
pub struct TieReport {
    /// `tie_word_embeddings` from `config.json`.
    pub config_tie_word_embeddings: bool,
    /// Whether `lm_head.weight` exists in the checkpoint.
    pub lm_head_present: bool,
    /// Whether the two tensors are elementwise identical after exact widening.
    pub lm_head_identical_to_embedding: bool,
    /// Whether the accounting counts the matrix once.
    pub treated_as_shared: bool,
    /// Elements compared (zero when there is nothing to compare).
    pub compared_elements: u64,
}

/// Budget arithmetic against the ceiling.
#[derive(Clone, Debug, Serialize)]
pub struct BudgetReport {
    /// Ceiling used.
    pub ceiling: u64,
    /// Sum over every checkpoint tensor, counting a tied pair twice.
    pub checkpoint_tensor_elements: u64,
    /// Tie-aware unique learned elements.
    pub unique_learned_elements: u64,
    /// Unique elements always executed: embedding, attention, norms and the
    /// output projection as the model actually uses them.
    pub always_active_backbone_elements: u64,
    /// Unique elements that a group selector can omit.
    pub removable_ffn_elements: u64,
    /// `ceiling - unique_learned_elements`; negative means the dense model is
    /// over budget.
    pub headroom_elements: i64,
    /// `ceiling - always_active_backbone_elements`, saturating at zero.
    pub max_compliant_ffn_elements: u64,
    /// Feasible FFN share of all FFN elements, in `[0, 1]`.
    pub max_compliant_ffn_fraction: f64,
    /// Full FFN elements in one layer.
    pub per_layer_full_ffn_elements: u64,
    /// Whole intermediate channels per layer that fit when every layer runs.
    pub max_compliant_channels_per_layer: u64,
    /// Intermediate width per layer.
    pub intermediate_per_layer: u64,
    /// Group count used.
    pub group_count: u32,
    /// Nominal group width (`ceil(intermediate / group_count)`).
    pub group_width: u64,
    /// Whole groups per layer that fit when every layer runs.
    pub max_compliant_whole_groups_per_layer: u64,
    /// Full-FFN layers that fit if other layers are skipped entirely.
    pub max_compliant_full_ffn_layers: f64,
    /// Whether the ceiling constrains this model at all.
    pub ceiling_binding: bool,
    /// Whether the dense model is compliant.
    pub dense_execution_compliant: bool,
    /// Whether the always-on backbone alone is compliant.
    pub backbone_fits: bool,
    /// Machine-readable verdict label.
    pub verdict: String,
    /// Plain-language explanation of the verdict.
    pub explanation: String,
}

/// One context-length entry.
#[derive(Clone, Debug, Serialize)]
pub struct ContextFlops {
    /// Context length in tokens.
    pub context: u64,
    /// Attention core matmul FLOPs for one output token at this context.
    pub attention_core: u64,
    /// Softmax elementwise estimate at this context.
    pub attention_softmax_estimate: u64,
}

/// Per-token arithmetic estimates.
#[derive(Clone, Debug, Serialize)]
pub struct PerTokenFlops {
    /// Attention projection matmuls, one layer.
    pub attention_projections_per_layer: u64,
    /// Attention projection matmuls, all layers.
    pub attention_projections_all_layers: u64,
    /// Attention core matmul at the reference contexts.
    pub attention_core: Vec<ContextFlops>,
    /// RMSNorm elementwise estimate, one layer.
    pub norms_per_layer_estimate: u64,
    /// Output projection matmul (executed once whether or not weights are tied).
    pub lm_head_projection: u64,
    /// Backbone matmul plus norm estimates, excluding the FFN and the
    /// context-dependent attention core.
    pub always_on_backbone: u64,
    /// Full SwiGLU FFN matmul, one layer.
    pub ffn_matmul_per_layer: u64,
    /// Full SwiGLU FFN elementwise estimate, one layer.
    pub ffn_elementwise_per_layer_estimate: u64,
    /// Full SwiGLU FFN matmul, all layers.
    pub ffn_matmul_all_layers: u64,
    /// Full SwiGLU FFN including elementwise estimates, all layers.
    pub ffn_total_all_layers_estimate: u64,
    /// Dense per-token total including the attention core at the first
    /// reference context.
    pub dense_total_at_first_context_estimate: u64,
}

/// Per-group arithmetic.
#[derive(Clone, Debug, Serialize)]
pub struct GroupFlops {
    /// Group count.
    pub group_count: u32,
    /// Nominal group width.
    pub group_width: u64,
    /// Per-group element counts in bundle order (the final group may be
    /// smaller).
    pub group_widths: Vec<u64>,
    /// Parameters in one full-width group (gate, up and down).
    pub elements_per_full_group: u64,
    /// FFN matmul FLOPs for one full-width group.
    pub matmul_flops_per_full_group: u64,
    /// FFN elementwise estimate for one full-width group.
    pub elementwise_per_full_group_estimate: u64,
    /// Always-on backbone plus one full-width group, per token.
    pub always_on_plus_one_group: u64,
    /// Always-on backbone plus all FFN groups, per token.
    pub always_on_plus_all_groups: u64,
}

/// FLOP section.
#[derive(Clone, Debug, Serialize)]
pub struct FlopsReport {
    /// States that these are formula estimates, not measurements.
    pub measured: bool,
    /// The formulas, in words.
    pub formulas: Vec<String>,
    /// Per-token estimates.
    pub per_token: PerTokenFlops,
    /// Per-group estimates.
    pub groups: GroupFlops,
}

/// Model identity facts read from the checkpoint.
#[derive(Clone, Debug, Serialize)]
pub struct ModelIdentity {
    /// Checkpoint path as given.
    pub path: String,
    /// Model type from `config.json`.
    pub model_type: String,
    /// Architecture names from `config.json`.
    pub architectures: Vec<String>,
    /// SHA-256 of `config.json`.
    pub config_sha256: String,
    /// Streaming checkpoint digest (hex), when computed.
    pub checkpoint_digest: Option<String>,
    /// Checkpoint file count.
    pub checkpoint_files: usize,
    /// Hidden width.
    pub hidden: u64,
    /// Intermediate width.
    pub intermediate: u64,
    /// Transformer blocks.
    pub layers: u64,
    /// Vocabulary size.
    pub vocab: u64,
    /// Attention heads.
    pub heads: u64,
    /// Key/value heads.
    pub kv_heads: u64,
    /// Head width.
    pub head_dim: u64,
}

/// The full accounting report.
#[derive(Clone, Debug, Serialize)]
pub struct ModelAccountingReport {
    /// Report schema.
    pub schema: String,
    /// Whether accounting completed without an inconsistency.
    pub ok: bool,
    /// Receipt stamp, recorded verbatim.
    pub stamp: Option<String>,
    /// Git revision, recorded verbatim.
    pub revision: Option<String>,
    /// Producer identity.
    pub generated_by: String,
    /// Model identity.
    pub model: ModelIdentity,
    /// Tie state.
    pub tie: TieReport,
    /// Per-tensor accounts, sorted by tensor key.
    pub components: Vec<TensorAccount>,
    /// Category totals.
    pub totals: CategoryTotals,
    /// Budget arithmetic.
    pub budget: BudgetReport,
    /// FLOP estimates.
    pub flops: FlopsReport,
}

#[derive(Debug, Deserialize)]
struct QwenConfig {
    model_type: String,
    #[serde(default)]
    architectures: Vec<String>,
    hidden_size: u64,
    intermediate_size: u64,
    num_hidden_layers: u64,
    num_attention_heads: u64,
    #[serde(default)]
    num_key_value_heads: Option<u64>,
    #[serde(default)]
    head_dim: Option<u64>,
    vocab_size: u64,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
}

struct Expected {
    tensor: String,
    component: &'static str,
    shape: Vec<u64>,
    layer: Option<u32>,
    always_active: bool,
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

fn dtype_size(dtype: &DType) -> Result<u64> {
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
        let parent = path.parent().ok_or_else(|| {
            ConvertError::inconsistent(format!(
                "{} has no parent directory to find config.json",
                path.display()
            ))
        })?;
        Ok(parent.join("config.json"))
    }
}

fn load_config(path: &Path) -> Result<(QwenConfig, String)> {
    let config_path = config_path(path)?;
    let bytes = std::fs::read(&config_path)?;
    let config: QwenConfig = serde_json::from_slice(&bytes).map_err(|error| {
        ConvertError::inconsistent(format!(
            "{} is not a readable Qwen3 config: {error}",
            config_path.display()
        ))
    })?;
    let mut builder = selvedge_core::Sha256Builder::new();
    builder.update(&bytes);
    let digest = builder.finish().to_hex();
    Ok((config, digest))
}

fn expected_tensors(config: &QwenConfig) -> Result<(Vec<Expected>, u64)> {
    if config.model_type != "qwen3" {
        return Err(ConvertError::inconsistent(format!(
            "accounting is implemented for model_type \"qwen3\", config says {:?}",
            config.model_type
        )));
    }
    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let layers = config.num_hidden_layers;
    let heads = config.num_attention_heads;
    let kv_heads = config.num_key_value_heads.unwrap_or(heads);
    let head_dim = match config.head_dim {
        Some(value) => value,
        None => hidden
            .checked_div(heads)
            .ok_or_else(|| ConvertError::inconsistent("hidden size divided by zero heads"))?,
    };
    if hidden == 0
        || intermediate == 0
        || layers == 0
        || heads == 0
        || kv_heads == 0
        || head_dim == 0
    {
        return Err(ConvertError::inconsistent(
            "config has a zero hidden/intermediate/layer/head dimension",
        ));
    }
    if config.vocab_size == 0 {
        return Err(ConvertError::inconsistent(
            "config has a zero vocabulary size",
        ));
    }
    let q_width = heads
        .checked_mul(head_dim)
        .ok_or_else(|| ConvertError::inconsistent("q width overflows u64"))?;
    let kv_width = kv_heads
        .checked_mul(head_dim)
        .ok_or_else(|| ConvertError::inconsistent("kv width overflows u64"))?;

    let mut expected = Vec::new();
    expected.push(Expected {
        tensor: "model.embed_tokens.weight".to_owned(),
        component: "embedding",
        shape: vec![config.vocab_size, hidden],
        layer: None,
        always_active: true,
        ffn: false,
    });
    for layer in 0..layers {
        let layer_index = u32::try_from(layer)
            .map_err(|_| ConvertError::inconsistent("layer index does not fit in u32"))?;
        let prefix = format!("model.layers.{layer}");
        for (name, component, shape) in [
            (
                "self_attn.q_proj.weight",
                "attention.q",
                vec![q_width, hidden],
            ),
            (
                "self_attn.k_proj.weight",
                "attention.k",
                vec![kv_width, hidden],
            ),
            (
                "self_attn.v_proj.weight",
                "attention.v",
                vec![kv_width, hidden],
            ),
            (
                "self_attn.o_proj.weight",
                "attention.o",
                vec![hidden, q_width],
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
            ("input_layernorm.weight", "norm", vec![hidden]),
            ("post_attention_layernorm.weight", "norm", vec![hidden]),
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
        ] {
            let ffn = component.starts_with("mlp.");
            expected.push(Expected {
                tensor: format!("{prefix}.{name}"),
                component,
                shape,
                layer: Some(layer_index),
                always_active: !ffn,
                ffn,
            });
        }
    }
    expected.push(Expected {
        tensor: "model.norm.weight".to_owned(),
        component: "norm",
        shape: vec![hidden],
        layer: None,
        always_active: true,
        ffn: false,
    });
    Ok((expected, q_width))
}

fn count_tensor(source: &impl TensorSource, want: &Expected) -> Result<TensorAccount> {
    let descriptor = source.descriptor(&want.tensor)?;
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
        .checked_mul(dtype_size(&descriptor.dtype)?)
        .ok_or_else(|| {
            ConvertError::inconsistent(format!("{}: byte count overflows", want.tensor))
        })?;
    if descriptor.byte_len != expected_bytes {
        return Err(ConvertError::inconsistent(format!(
            "{} declares {} bytes, expected {expected_bytes} for {} elements of {}",
            want.tensor,
            descriptor.byte_len,
            elements,
            descriptor.dtype.stable_name()
        )));
    }
    Ok(TensorAccount {
        tensor: want.tensor.clone(),
        component: want.component.to_owned(),
        shape: descriptor.shape.clone(),
        elements,
        bytes: descriptor.byte_len,
        dtype: descriptor.dtype.stable_name(),
        layer: want.layer,
        always_active: want.always_active,
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
    let rows = left.shape.first().copied().unwrap_or(0);
    let cols = left.shape.get(1).copied().unwrap_or(0);
    if left.rank() != 2 || right.rank() != 2 {
        return Ok(false);
    }
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

fn flops_for(
    config: &QwenConfig,
    q_width: u64,
    options: &AccountingOptions,
) -> Result<FlopsReport> {
    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let layers = config.num_hidden_layers;
    let heads = config.num_attention_heads;
    let kv_heads = config.num_key_value_heads.unwrap_or(heads);
    let head_dim = config.head_dim.unwrap_or_else(|| hidden / heads.max(1));
    let kv_width = kv_heads * head_dim;
    let vocab = config.vocab_size;

    let matmul = |out: u64, in_features: u64| -> Result<u64> {
        out.checked_mul(in_features)
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| ConvertError::inconsistent("FLOP count overflows u64"))
    };

    let q = matmul(q_width, hidden)?;
    let k = matmul(kv_width, hidden)?;
    let v = k;
    let o = matmul(hidden, q_width)?;
    let attn_layer = q
        .checked_add(k)
        .and_then(|value| value.checked_add(v))
        .and_then(|value| value.checked_add(o))
        .ok_or_else(|| ConvertError::inconsistent("attention FLOPs overflow u64"))?;
    let attn_all = attn_layer
        .checked_mul(layers)
        .ok_or_else(|| ConvertError::inconsistent("attention FLOPs overflow u64"))?;

    // RMSNorm estimate: per element, square, add into the sum, multiply by the
    // inverse RMS and multiply by the learned weight. q_norm/k_norm are applied
    // once per head, so their element count is head_count * head_dim.
    let norm_layer = (hidden + hidden)
        .checked_add(heads * head_dim)
        .and_then(|value| value.checked_add(kv_heads * head_dim))
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| ConvertError::inconsistent("norm FLOPs overflow u64"))?;

    let lm_head = matmul(vocab, hidden)?;

    let always_on_backbone = attn_all
        .checked_add(
            norm_layer
                .checked_mul(layers)
                .ok_or_else(|| ConvertError::inconsistent("norm FLOPs overflow u64"))?,
        )
        .and_then(|value| value.checked_add(lm_head))
        .and_then(|value| value.checked_add(4 * hidden))
        .ok_or_else(|| ConvertError::inconsistent("backbone FLOPs overflow u64"))?;

    let ffn_matmul_layer = {
        let gate = matmul(intermediate, hidden)?;
        let up = matmul(intermediate, hidden)?;
        let down = matmul(hidden, intermediate)?;
        gate.checked_add(up)
            .and_then(|value| value.checked_add(down))
            .ok_or_else(|| ConvertError::inconsistent("FFN FLOPs overflow u64"))?
    };
    let ffn_elementwise_layer = intermediate
        .checked_mul(6)
        .ok_or_else(|| ConvertError::inconsistent("FFN FLOPs overflow u64"))?;
    let ffn_matmul_all = ffn_matmul_layer
        .checked_mul(layers)
        .ok_or_else(|| ConvertError::inconsistent("FFN FLOPs overflow u64"))?;
    let ffn_total_all = ffn_matmul_all
        .checked_add(
            ffn_elementwise_layer
                .checked_mul(layers)
                .ok_or_else(|| ConvertError::inconsistent("FFN FLOPs overflow u64"))?,
        )
        .ok_or_else(|| ConvertError::inconsistent("FFN FLOPs overflow u64"))?;

    let mut contexts = Vec::with_capacity(options.reference_contexts.len());
    for context in &options.reference_contexts {
        // QK^T plus AV: two matmuls of [heads, context, head_dim].
        let core = context
            .checked_mul(q_width)
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| ConvertError::inconsistent("attention core FLOPs overflow u64"))?;
        let softmax = context
            .checked_mul(heads)
            .and_then(|value| value.checked_mul(5))
            .ok_or_else(|| ConvertError::inconsistent("softmax FLOPs overflow u64"))?;
        contexts.push(ContextFlops {
            context: *context,
            attention_core: core,
            attention_softmax_estimate: softmax,
        });
    }
    let first_context = match contexts.first() {
        Some(entry) => entry
            .attention_core
            .checked_mul(layers)
            .and_then(|core| {
                entry
                    .attention_softmax_estimate
                    .checked_mul(layers)
                    .and_then(|softmax| core.checked_add(softmax))
            })
            .ok_or_else(|| ConvertError::inconsistent("attention FLOPs overflow u64"))?,
        None => 0,
    };
    let dense_total = always_on_backbone
        .checked_add(ffn_total_all)
        .and_then(|value| value.checked_add(first_context))
        .ok_or_else(|| ConvertError::inconsistent("dense FLOP total overflows u64"))?;

    let group_count = options.group_count;
    let group_width = intermediate.div_ceil(u64::from(group_count));
    let mut widths = Vec::new();
    let mut remaining = intermediate;
    while remaining > 0 {
        let width = remaining.min(group_width);
        widths.push(width);
        remaining -= width;
    }
    let elements_per_full_group = group_width
        .checked_mul(hidden)
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| ConvertError::inconsistent("group element count overflows u64"))?;
    let matmul_flops_full_group = {
        let gate = matmul(group_width, hidden)?;
        let up = matmul(group_width, hidden)?;
        let down = matmul(hidden, group_width)?;
        gate.checked_add(up)
            .and_then(|value| value.checked_add(down))
            .ok_or_else(|| ConvertError::inconsistent("group FLOPs overflow u64"))?
    };
    let elementwise_full_group = group_width
        .checked_mul(6)
        .ok_or_else(|| ConvertError::inconsistent("group FLOPs overflow u64"))?;
    let one_group_total = always_on_backbone
        .checked_add(matmul_flops_full_group)
        .and_then(|value| value.checked_add(elementwise_full_group))
        .ok_or_else(|| ConvertError::inconsistent("group FLOPs overflow u64"))?;
    let all_groups_total = always_on_backbone
        .checked_add(ffn_total_all)
        .ok_or_else(|| ConvertError::inconsistent("group FLOPs overflow u64"))?;

    let _ = ();

    Ok(FlopsReport {
        measured: false,
        formulas: vec![
            "matmul FLOPs = 2 * out_features * in_features for one token".to_owned(),
            "attention projections = q + k + v + o matmuls, using kv_heads for k and v".to_owned(),
            "attention core = 4 * context_length * (heads * head_dim) for QK^T and AV".to_owned(),
            "attention softmax estimate = 5 * context_length * heads (elementwise, not measured)"
                .to_owned(),
            "RMSNorm estimate = 4 * normalised_elements; q_norm and k_norm apply once per head"
                .to_owned(),
            "SwiGLU matmul = 2*(I*H) for gate + 2*(I*H) for up + 2*(H*I) for down = 6*I*H"
                .to_owned(),
            "SwiGLU elementwise estimate = 6 * I (SiLU, gate multiply, group accumulate)"
                .to_owned(),
            "lm_head = 2 * vocab * hidden, executed once per token whether or not weights are tied"
                .to_owned(),
            "embedding lookup is a gather and contributes no matmul FLOPs".to_owned(),
            "no FLOP here is measured; they are arithmetic estimates from these formulas"
                .to_owned(),
        ],
        per_token: PerTokenFlops {
            attention_projections_per_layer: attn_layer,
            attention_projections_all_layers: attn_all,
            attention_core: contexts,
            norms_per_layer_estimate: norm_layer,
            lm_head_projection: lm_head,
            always_on_backbone,
            ffn_matmul_per_layer: ffn_matmul_layer,
            ffn_elementwise_per_layer_estimate: ffn_elementwise_layer,
            ffn_matmul_all_layers: ffn_matmul_all,
            ffn_total_all_layers_estimate: ffn_total_all,
            dense_total_at_first_context_estimate: dense_total,
        },
        groups: GroupFlops {
            group_count,
            group_width,
            group_widths: widths,
            elements_per_full_group,
            matmul_flops_per_full_group: matmul_flops_full_group,
            elementwise_per_full_group_estimate: elementwise_full_group,
            always_on_plus_one_group: one_group_total,
            always_on_plus_all_groups: all_groups_total,
        },
    })
}

fn budget_for(
    config: &QwenConfig,
    totals: &CategoryTotals,
    unique: u64,
    backbone_unique: u64,
    checkpoint_elements: u64,
    options: &AccountingOptions,
) -> Result<BudgetReport> {
    let per_layer_ffn = totals
        .ffn
        .checked_div(config.num_hidden_layers.max(1))
        .unwrap_or(0);
    let backbone = backbone_unique;
    let max_ffn = options.ceiling.saturating_sub(backbone);
    let ffn_fraction = if totals.ffn == 0 {
        0.0
    } else {
        (max_ffn as f64 / totals.ffn as f64).min(1.0)
    };
    let per_channel = 3_u64
        .checked_mul(config.hidden_size)
        .ok_or_else(|| ConvertError::inconsistent("per-channel cost overflows u64"))?;
    let channels_per_layer = if per_channel == 0 {
        0
    } else {
        max_ffn / (per_channel * config.num_hidden_layers.max(1))
    };
    let group_width = config
        .intermediate_size
        .div_ceil(u64::from(options.group_count));
    let whole_groups = channels_per_layer.checked_div(group_width).unwrap_or(0);
    let full_ffn_layers = if per_layer_ffn == 0 {
        0.0
    } else {
        max_ffn as f64 / per_layer_ffn as f64
    };
    let dense_compliant = unique <= options.ceiling;
    let backbone_fits = backbone <= options.ceiling;
    let (verdict, explanation) = if dense_compliant {
        (
            "not_binding".to_owned(),
            format!(
                "the full dense checkpoint is {unique} unique learned parameters, at or below the \
                 {ceiling} ceiling; sparsity is a quality/cost choice here, not a compliance \
                 requirement",
                ceiling = options.ceiling
            ),
        )
    } else if backbone_fits {
        (
            "binding_backbone_fits".to_owned(),
            format!(
                "the dense checkpoint is {unique} unique parameters and exceeds the {ceiling} \
                 ceiling, but the always-on backbone is {backbone}; at most {max_ffn} FFN \
                 parameters (a fraction {ffn_fraction:.6} of the FFN) can be active per token",
                ceiling = options.ceiling
            ),
        )
    } else {
        (
            "impossible_backbone_over_ceiling".to_owned(),
            format!(
                "the always-on backbone alone is {backbone} unique parameters, above the \
                 {ceiling} ceiling; no group selection can make this model compliant without \
                 architectural compression or distillation",
                ceiling = options.ceiling
            ),
        )
    };
    Ok(BudgetReport {
        ceiling: options.ceiling,
        checkpoint_tensor_elements: checkpoint_elements,
        unique_learned_elements: unique,
        always_active_backbone_elements: backbone,
        removable_ffn_elements: totals.ffn,
        headroom_elements: i64::try_from(options.ceiling)
            .map_err(|_| ConvertError::inconsistent("ceiling does not fit in i64"))?
            - i64::try_from(unique)
                .map_err(|_| ConvertError::inconsistent("unique count does not fit in i64"))?,
        max_compliant_ffn_elements: max_ffn,
        max_compliant_ffn_fraction: ffn_fraction,
        per_layer_full_ffn_elements: per_layer_ffn,
        max_compliant_channels_per_layer: channels_per_layer,
        intermediate_per_layer: config.intermediate_size,
        group_count: options.group_count,
        group_width,
        max_compliant_whole_groups_per_layer: whole_groups,
        max_compliant_full_ffn_layers: full_ffn_layers,
        ceiling_binding: !dense_compliant,
        dense_execution_compliant: dense_compliant,
        backbone_fits,
        verdict,
        explanation,
    })
}

/// Account a real Qwen3 checkpoint.
///
/// # Errors
///
/// Rejects a missing or inconsistent `config.json`, an unknown model type,
/// missing or unexpected tensors, shape/dtype/byte mismatches, a tie claimed by
/// the config but not present in the bytes, and every reader failure.
pub fn account_model(path: &Path, options: &AccountingOptions) -> Result<ModelAccountingReport> {
    options.validate()?;
    let (config, config_sha256) = load_config(path)?;
    let (expected, q_width) = expected_tensors(&config)?;
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
        accounts.push(count_tensor(&source, want)?);
    }

    let config_tied = config.tie_word_embeddings.unwrap_or(false);
    let mut tie = TieReport {
        config_tie_word_embeddings: config_tied,
        lm_head_present,
        lm_head_identical_to_embedding: false,
        treated_as_shared: false,
        compared_elements: 0,
    };
    if lm_head_present {
        let embed = source.descriptor("model.embed_tokens.weight")?;
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
        let account = count_tensor(
            &source,
            &Expected {
                tensor: lm_head_key.clone(),
                component: "lm_head",
                shape: head.shape.clone(),
                layer: None,
                always_active: true,
                ffn: false,
            },
        )?;
        accounts.push(account);
        match (config_tied, identical) {
            (true, false) => {
                return Err(ConvertError::inconsistent(
                    "config declares tie_word_embeddings=true but lm_head.weight is not \
                     byte-identical to model.embed_tokens.weight",
                ));
            }
            (true, true) => tie.treated_as_shared = true,
            (false, _) => tie.treated_as_shared = false,
        }
    } else if config_tied {
        // Tied without an lm_head tensor: the embedding is the output projection.
        tie.treated_as_shared = true;
        tie.lm_head_identical_to_embedding = true;
    }

    accounts.sort_by(|a, b| a.tensor.cmp(&b.tensor));
    let mut totals = CategoryTotals::default();
    let mut checkpoint_elements = 0_u64;
    for account in &accounts {
        checkpoint_elements = checkpoint_elements
            .checked_add(account.elements)
            .ok_or_else(|| ConvertError::inconsistent("checkpoint element count overflows u64"))?;
        match account.component.as_str() {
            "embedding" => totals.embedding = totals.embedding.saturating_add(account.elements),
            "lm_head" => totals.lm_head = totals.lm_head.saturating_add(account.elements),
            "norm" => totals.norm = totals.norm.saturating_add(account.elements),
            "mlp.gate" | "mlp.up" | "mlp.down" => {
                totals.ffn = totals.ffn.saturating_add(account.elements)
            }
            component if component.starts_with("attention.") => {
                totals.attention = totals.attention.saturating_add(account.elements)
            }
            other => {
                return Err(ConvertError::inconsistent(format!(
                    "accounting has no category for component {other:?}"
                )));
            }
        }
    }
    let unique = if tie.treated_as_shared {
        checkpoint_elements
            .checked_sub(totals.lm_head)
            .ok_or_else(|| ConvertError::inconsistent("tie subtraction underflows"))?
    } else {
        checkpoint_elements
    };
    let backbone_unique = totals
        .embedding
        .checked_add(totals.attention)
        .and_then(|value| value.checked_add(totals.norm))
        .and_then(|value| {
            if tie.treated_as_shared {
                Some(value)
            } else {
                value.checked_add(totals.lm_head)
            }
        })
        .ok_or_else(|| ConvertError::inconsistent("backbone element count overflows u64"))?;
    if backbone_unique
        != if tie.treated_as_shared {
            totals
                .embedding
                .saturating_add(totals.attention)
                .saturating_add(totals.norm)
        } else {
            totals
                .embedding
                .saturating_add(totals.attention)
                .saturating_add(totals.norm)
                .saturating_add(totals.lm_head)
        }
    {
        return Err(ConvertError::inconsistent("backbone arithmetic drifted"));
    }

    let budget = budget_for(
        &config,
        &totals,
        unique,
        backbone_unique,
        checkpoint_elements,
        options,
    )?;

    let flops = flops_for(&config, q_width, options)?;
    let checkpoint_files = source.relative_paths().len().max(1);
    let digest = if options.compute_digest {
        Some(
            crate::convert::model_digest_for_path(path)?
                .digest()
                .to_hex(),
        )
    } else {
        None
    };

    Ok(ModelAccountingReport {
        schema: "selvedge/model-accounting/v1".to_owned(),
        ok: true,
        stamp: options.stamp.clone(),
        revision: options.revision.clone(),
        generated_by: format!("selvedge-convert {}", env!("CARGO_PKG_VERSION")),
        model: ModelIdentity {
            path: path.display().to_string(),
            model_type: config.model_type.clone(),
            architectures: config.architectures.clone(),
            config_sha256,
            checkpoint_digest: digest,
            checkpoint_files,
            hidden: config.hidden_size,
            intermediate: config.intermediate_size,
            layers: config.num_hidden_layers,
            vocab: config.vocab_size,
            heads: config.num_attention_heads,
            kv_heads: config
                .num_key_value_heads
                .unwrap_or(config.num_attention_heads),
            head_dim: config
                .head_dim
                .unwrap_or(config.hidden_size / config.num_attention_heads.max(1)),
        },
        tie,
        components: accounts,
        totals,
        budget,
        flops,
    })
}

/// Per-category element totals, for callers that do not need the full report.
#[must_use]
pub fn category_totals(report: &ModelAccountingReport) -> BTreeMap<&str, u64> {
    let mut map = BTreeMap::new();
    map.insert("embedding", report.totals.embedding);
    map.insert("attention", report.totals.attention);
    map.insert("norm", report.totals.norm);
    map.insert("ffn", report.totals.ffn);
    map.insert("lm_head", report.totals.lm_head);
    map
}
