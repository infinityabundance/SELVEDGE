//! Loading the always-executed part of Qwen3 from the original safetensors
//! checkpoint, through the S2 reader.
//!
//! Only the tensors that are **not** the SwiGLU MLP live here. The three MLP
//! matrices of every layer are read from the SELVEDGE bundle instead (see
//! [`crate::bundle`]), which is what makes the bundle the unit of execution.
//!
//! The loader validates every shape against [`Qwen3Config`] and refuses a
//! checkpoint that is missing a tensor, has an unexpected shape, carries an
//! unsupported dtype, or contains an `lm_head` next to `tie_word_embeddings`.
//! There is no fallback dtype and no partial load.

use std::path::Path;

use selvedge_convert::{CheckpointSource, TensorSource};
use selvedge_core::{ParamIdentity, ParamTensorId, ParamUse, StorageTier};

use crate::config::Qwen3Config;
use crate::error::{Result, RuntimeError};

/// Per-layer tensors that are read on every token.
#[derive(Clone, Debug)]
pub struct LayerWeights {
    /// Input layer norm gain, `[H]`.
    pub input_ln: Vec<f32>,
    /// Post-attention layer norm gain, `[H]`.
    pub post_ln: Vec<f32>,
    /// Query projection, `[n_heads * head_dim, H]` row-major.
    pub q: Vec<f32>,
    /// Key projection, `[n_kv * head_dim, H]` row-major.
    pub k: Vec<f32>,
    /// Value projection, `[n_kv * head_dim, H]` row-major.
    pub v: Vec<f32>,
    /// Output projection, `[H, n_heads * head_dim]` row-major.
    pub o: Vec<f32>,
    /// Per-head query norm gain, `[head_dim]`.
    pub q_norm: Vec<f32>,
    /// Per-head key norm gain, `[head_dim]`.
    pub k_norm: Vec<f32>,
}

/// The tied embedding matrix and the layer weights, in f32.
#[derive(Clone, Debug)]
pub struct Backbone {
    /// Architecture.
    pub config: Qwen3Config,
    /// Tied input/output embedding in logical row-major order, `[V, H]`.
    pub embed: Vec<f32>,
    /// One entry per decoder layer.
    pub layers: Vec<LayerWeights>,
    /// Final layer norm gain, `[H]`.
    pub final_norm: Vec<f32>,
    /// Per-token parameter uses for everything except the embedding row and the
    /// tied output projection, which are position dependent.
    pub uses: Vec<ParamUse>,
    /// Learned element count of the whole backbone, including the tied matrix.
    pub parameter_elements: u64,
    /// Whether the checkpoint carried an `lm_head.weight` that was verified
    /// bit-identical to the tied embedding.
    pub lm_head_verified_tied: bool,
}

/// Largest single tensor the loader will materialise, in bytes.
///
/// A bound rather than a heuristic: the pinned 0.6B embedding is 0.31 GiB of
/// bf16 and its f32 form is 0.62 GiB, so 1 GiB admits it and nothing plausible
/// beyond it.
pub const MAX_TENSOR_BYTES: u64 = 1024 * 1024 * 1024;

impl Backbone {
    /// Load the backbone from a checkpoint file or directory.
    ///
    /// # Errors
    ///
    /// Rejects a missing or malformed checkpoint, an unexpected tensor set or
    /// shape, an unsupported dtype, and an `lm_head` alongside tied embeddings.
    pub fn load(checkpoint: &Path, config: &Qwen3Config) -> Result<Self> {
        config.validate()?;
        let source = CheckpointSource::open(checkpoint)?;
        let hidden = usize::try_from(config.hidden_size)
            .map_err(|_| RuntimeError::config("hidden_size", "does not fit in usize"))?;
        let head_dim = usize::try_from(config.head_dim)
            .map_err(|_| RuntimeError::config("head_dim", "does not fit in usize"))?;
        let q_dim = config.q_dim()?;
        let kv_dim = config.kv_dim()?;
        let layers = usize::try_from(config.num_hidden_layers)
            .map_err(|_| RuntimeError::config("num_hidden_layers", "does not fit in usize"))?;

        let mut uses = Vec::new();
        let embed_key = "model.embed_tokens.weight".to_owned();
        let embed = read_matrix(&source, &embed_key, config.vocab_size, config.hidden_size)?;
        record_use(&mut uses, &embed_key, embed.len())?;

        // The pinned Qwen3 control ships `lm_head.weight` even though
        // `tie_word_embeddings` is true. HF ties the two at load, so the tied
        // matrix is authoritative — but this runtime will not assume the file
        // agrees. It reads the second matrix and requires it to be bit-identical
        // to the embedding; a difference is a hard error, never a silent choice.
        let keys = source.keys();
        let mut lm_head_verified_tied = false;
        if keys.iter().any(|key| key == "lm_head.weight") {
            let tied = read_matrix(
                &source,
                "lm_head.weight",
                config.vocab_size,
                config.hidden_size,
            )?;
            if tied.len() != embed.len()
                || tied
                    .iter()
                    .zip(embed.iter())
                    .any(|(left, right)| left.to_bits() != right.to_bits())
            {
                return Err(RuntimeError::model(
                    "checkpoint carries lm_head.weight that differs from model.embed_tokens.weight while tie_word_embeddings is true; refusing to guess which matrix is authoritative",
                ));
            }
            lm_head_verified_tied = true;
        }

        let mut per_layer = Vec::with_capacity(layers);
        for layer in 0..layers {
            let prefix = format!("model.layers.{layer}");
            let input_ln =
                read_vector(&source, &format!("{prefix}.input_layernorm.weight"), hidden)?;
            record_use(
                &mut uses,
                &format!("{prefix}.input_layernorm.weight"),
                hidden,
            )?;
            let post_ln = read_vector(
                &source,
                &format!("{prefix}.post_attention_layernorm.weight"),
                hidden,
            )?;
            record_use(
                &mut uses,
                &format!("{prefix}.post_attention_layernorm.weight"),
                hidden,
            )?;
            let q = read_matrix(
                &source,
                &format!("{prefix}.self_attn.q_proj.weight"),
                q_dim as u64,
                config.hidden_size,
            )?;
            record_use(
                &mut uses,
                &format!("{prefix}.self_attn.q_proj.weight"),
                q.len(),
            )?;
            let k = read_matrix(
                &source,
                &format!("{prefix}.self_attn.k_proj.weight"),
                kv_dim as u64,
                config.hidden_size,
            )?;
            record_use(
                &mut uses,
                &format!("{prefix}.self_attn.k_proj.weight"),
                k.len(),
            )?;
            let v = read_matrix(
                &source,
                &format!("{prefix}.self_attn.v_proj.weight"),
                kv_dim as u64,
                config.hidden_size,
            )?;
            record_use(
                &mut uses,
                &format!("{prefix}.self_attn.v_proj.weight"),
                v.len(),
            )?;
            let o = read_matrix(
                &source,
                &format!("{prefix}.self_attn.o_proj.weight"),
                config.hidden_size,
                q_dim as u64,
            )?;
            record_use(
                &mut uses,
                &format!("{prefix}.self_attn.o_proj.weight"),
                o.len(),
            )?;
            let q_norm = read_vector(
                &source,
                &format!("{prefix}.self_attn.q_norm.weight"),
                head_dim,
            )?;
            record_use(
                &mut uses,
                &format!("{prefix}.self_attn.q_norm.weight"),
                head_dim,
            )?;
            let k_norm = read_vector(
                &source,
                &format!("{prefix}.self_attn.k_norm.weight"),
                head_dim,
            )?;
            record_use(
                &mut uses,
                &format!("{prefix}.self_attn.k_norm.weight"),
                head_dim,
            )?;
            per_layer.push(LayerWeights {
                input_ln,
                post_ln,
                q,
                k,
                v,
                o,
                q_norm,
                k_norm,
            });
        }
        let final_norm = read_vector(&source, "model.norm.weight", hidden)?;
        record_use(&mut uses, "model.norm.weight", hidden)?;

        let mut parameter_elements = embed.len() as u64;
        for layer in &per_layer {
            for buffer in [
                &layer.input_ln,
                &layer.post_ln,
                &layer.q,
                &layer.k,
                &layer.v,
                &layer.o,
                &layer.q_norm,
                &layer.k_norm,
            ] {
                parameter_elements = parameter_elements.saturating_add(buffer.len() as u64);
            }
        }
        parameter_elements = parameter_elements.saturating_add(final_norm.len() as u64);

        Ok(Self {
            config: config.clone(),
            embed,
            layers: per_layer,
            final_norm,
            uses,
            parameter_elements,
            lm_head_verified_tied,
        })
    }

    /// Parameter uses for one token's embedding row and tied output projection.
    ///
    /// The row read and the full output projection read share one tensor key, so
    /// the union counts the tied matrix once while the reads counter keeps both.
    ///
    /// # Errors
    ///
    /// Rejects a token id outside the vocabulary and overflow.
    pub fn embedding_uses(&self, token: u32) -> Result<Vec<ParamUse>> {
        let hidden = usize::try_from(self.config.hidden_size)
            .map_err(|_| RuntimeError::config("hidden_size", "does not fit in usize"))?;
        let vocab = usize::try_from(self.config.vocab_size)
            .map_err(|_| RuntimeError::config("vocab_size", "does not fit in usize"))?;
        let token = usize::try_from(token)
            .map_err(|_| RuntimeError::model("token id does not fit in usize"))?;
        if token >= vocab {
            return Err(RuntimeError::model(format!(
                "token id {token} is outside the vocabulary of {vocab}"
            )));
        }
        let tensor = ParamTensorId::new("model.embed_tokens.weight")?;
        let start = (token * hidden) as u64;
        let end = start + hidden as u64;
        let row = ParamUse::new(
            ParamIdentity::new(tensor.clone(), start..end)?,
            hidden as u64,
            (hidden * 4) as u64,
            StorageTier::HostRam,
        )?;
        let elements = (vocab * hidden) as u64;
        let full = ParamUse::new(
            ParamIdentity::new(tensor, 0..elements)?,
            elements,
            elements * 4,
            StorageTier::HostRam,
        )?;
        Ok(vec![row, full])
    }

    /// Number of learned elements read per token by the backbone, including the
    /// tied output projection and the embedding row.
    ///
    /// The row is inside the tied matrix, so the union is the backbone count
    /// with the tied matrix counted once.
    #[must_use]
    pub fn per_token_union(&self) -> u64 {
        self.parameter_elements
    }
}

fn record_use(uses: &mut Vec<ParamUse>, key: &str, elements: usize) -> Result<()> {
    let elements = elements as u64;
    let tensor = ParamTensorId::new(key)?;
    uses.push(ParamUse::once(
        ParamIdentity::new(tensor, 0..elements)?,
        elements * 4,
        StorageTier::HostRam,
    )?);
    Ok(())
}

fn read_matrix(source: &CheckpointSource, key: &str, rows: u64, cols: u64) -> Result<Vec<f32>> {
    let descriptor = source.descriptor(key)?;
    if descriptor.rank() != 2 {
        return Err(RuntimeError::model(format!(
            "{key} has rank {}, expected 2",
            descriptor.rank()
        )));
    }
    if descriptor.shape != [rows, cols] {
        return Err(RuntimeError::model(format!(
            "{key} has shape {:?}, expected {:?}",
            descriptor.shape,
            [rows, cols]
        )));
    }
    let values = source.read_tensor_f32(key, MAX_TENSOR_BYTES)?;
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| RuntimeError::model(format!("{key} element count overflows")))?;
    if values.len() as u64 != expected {
        return Err(RuntimeError::model(format!(
            "{key} decoded {} values, expected {expected}",
            values.len()
        )));
    }
    Ok(values)
}

fn read_vector(source: &CheckpointSource, key: &str, len: usize) -> Result<Vec<f32>> {
    let descriptor = source.descriptor(key)?;
    if descriptor.rank() != 1 {
        return Err(RuntimeError::model(format!(
            "{key} has rank {}, expected 1",
            descriptor.rank()
        )));
    }
    if descriptor.shape != [len as u64] {
        return Err(RuntimeError::model(format!(
            "{key} has shape {:?}, expected [{len}]",
            descriptor.shape
        )));
    }
    let values = source.read_tensor_f32(key, MAX_TENSOR_BYTES)?;
    if values.len() != len {
        return Err(RuntimeError::model(format!(
            "{key} decoded {} values, expected {len}",
            values.len()
        )));
    }
    Ok(values)
}
