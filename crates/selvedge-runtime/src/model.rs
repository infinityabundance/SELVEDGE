//! The Qwen3 forward pass: tied embeddings, RMSNorm, RoPE, grouped-query
//! attention with per-head q/k norm, SwiGLU MLP through the bundle, final norm
//! and a tied output projection.
//!
//! The pass is **token-at-a-time**. A prompt is the same loop repeated over its
//! tokens with the KV cache growing, so the arithmetic applied to a given token
//! does not depend on how many tokens came before it in the same call, on the
//! batch, or on the streaming configuration. This is what makes the Court E
//! comparison meaningful: prompt composition changes the *set* of cached
//! key/value rows, never the reduction order of any dot product.
//!
//! # What the model owns and what it does not
//!
//! [`Model`] owns the always-executed weights, the rotary table and the scratch
//! buffers. It does **not** own the MLP: [`MlpExecutor`] is the seam where the
//! bundle, the store, the mode state and the budget gate live. A model cannot
//! silently execute a different MLP than the one it was given.

use crate::config::Qwen3Config;
use crate::error::{Result, RuntimeError};
use crate::kernels::{self, argmax};
use crate::weights::Backbone;

/// Per-token FLOP bookkeeping.
///
/// Matmul FLOPs use the documented `2 * M * N * K` formula through
/// [`selvedge_core::FlopEstimate::matmul`]; everything else is counted as a
/// per-element estimate and reported separately, so a reader can always see
/// which part is a formula over shapes and which part is an constant per
/// element.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TokenTrace {
    /// Dense matrix-multiply FLOPs.
    pub matmul_flops: u64,
    /// Norm, rope, softmax, activation and reduction FLOPs.
    pub other_flops: u64,
    /// Layers visited.
    pub layers: u64,
}

impl TokenTrace {
    /// Record a dense matrix-vector product of `rows x cols`.
    ///
    /// # Errors
    ///
    /// Rejects overflow.
    pub fn add_matvec(&mut self, rows: u64, cols: u64) -> Result<()> {
        let estimate = selvedge_core::FlopEstimate::matmul(1, rows, cols)?;
        self.matmul_flops = self
            .matmul_flops
            .checked_add(estimate.matmul_flops)
            .ok_or_else(|| RuntimeError::model("matmul FLOPs overflow"))?;
        Ok(())
    }

    /// Record a dense matrix-multiply of `batch x rows x cols`.
    ///
    /// # Errors
    ///
    /// Rejects overflow.
    pub fn add_matmul(&mut self, batch: u64, rows: u64, cols: u64) -> Result<()> {
        let estimate = selvedge_core::FlopEstimate::matmul(batch, rows, cols)?;
        self.matmul_flops = self
            .matmul_flops
            .checked_add(estimate.matmul_flops)
            .ok_or_else(|| RuntimeError::model("matmul FLOPs overflow"))?;
        Ok(())
    }

    /// Record an elementwise/reduction estimate.
    ///
    /// # Errors
    ///
    /// Rejects overflow.
    pub fn add_other(&mut self, flops: u64) -> Result<()> {
        self.other_flops = self
            .other_flops
            .checked_add(flops)
            .ok_or_else(|| RuntimeError::model("other FLOPs overflow"))?;
        Ok(())
    }

    /// The estimate in the core's representation.
    ///
    /// # Errors
    ///
    /// Rejects overflow.
    pub fn estimate(&self) -> Result<selvedge_core::FlopEstimate> {
        Ok(selvedge_core::FlopEstimate {
            matmul_flops: self.matmul_flops,
            other_flops: self.other_flops,
            controller_flops: 0,
        })
    }
}

/// The seam through which the MLP is executed.
pub trait MlpExecutor {
    /// Execute layer `layer`'s MLP contribution for the token whose pre-MLP
    /// state is `x`, writing the residual contribution (`H` values) into `out`.
    ///
    /// `out` must be fully overwritten, not accumulated into: the model applies
    /// the residual add itself.
    ///
    /// # Errors
    ///
    /// Any mode, store, budget or accounting failure.
    fn mlp(
        &mut self,
        layer: usize,
        x: &[f32],
        out: &mut [f32],
        trace: &mut TokenTrace,
    ) -> Result<()>;
}

/// The batched seam the model uses for batched prefill.
///
/// `xs` and `outs` are `batch x H` row-major. Row `b` is the pre-MLP state of the
/// batch's `b`-th token, and row `b` of `outs` must be exactly what
/// [`MlpExecutor::mlp`] would have written for that row's vector. That equality
/// is the contract the batched prefill is proven against.
pub trait BatchMlpExecutor {
    /// Execute layer `layer`'s MLP contribution for every row of the batch.
    ///
    /// # Errors
    ///
    /// Any mode, store, budget or accounting failure.
    fn mlp_batch(
        &mut self,
        layer: usize,
        xs: &[f32],
        outs: &mut [f32],
        batch: usize,
        trace: &mut TokenTrace,
    ) -> Result<()>;
}

/// Key/value cache for one sequence.
///
/// Capacity is sized from the request, not from `max_position_embeddings`: the
/// pinned model allows 40960 positions, whose cache for all 28 layers would be
/// about 9.4 GiB in f32. A run allocates exactly `prompt + max_new_tokens`
/// slots and refuses to decode past them.
#[derive(Clone, Debug)]
pub struct KvCache {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    len: usize,
    capacity: usize,
    kv_dim: usize,
}

impl KvCache {
    /// Allocate a cache for `capacity` positions.
    ///
    /// # Errors
    ///
    /// Rejects a zero capacity, a zero layer count and any allocation the
    /// allocator refuses (reported instead of aborting).
    pub fn new(config: &Qwen3Config, capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(RuntimeError::config(
                "kv_cache",
                "capacity must be non-zero",
            ));
        }
        let layers = usize::try_from(config.num_hidden_layers)
            .map_err(|_| RuntimeError::config("num_hidden_layers", "does not fit in usize"))?;
        let kv_dim = config.kv_dim()?;
        let elements = capacity
            .checked_mul(kv_dim)
            .ok_or_else(|| RuntimeError::memory("kv cache element count overflows"))?;
        let mut keys = Vec::with_capacity(layers);
        let mut values = Vec::with_capacity(layers);
        for _ in 0..layers {
            let mut k = Vec::new();
            let mut v = Vec::new();
            k.try_reserve_exact(elements).map_err(|_| {
                RuntimeError::memory(format!("cannot reserve {elements} f32 for k"))
            })?;
            v.try_reserve_exact(elements).map_err(|_| {
                RuntimeError::memory(format!("cannot reserve {elements} f32 for v"))
            })?;
            k.resize(elements, 0.0);
            v.resize(elements, 0.0);
            keys.push(k);
            values.push(v);
        }
        Ok(Self {
            keys,
            values,
            len: 0,
            capacity,
            kv_dim,
        })
    }

    /// Positions currently cached.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Positions the cache can hold.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Whether the cache is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Drop every cached position.
    pub fn reset(&mut self) {
        self.len = 0;
    }

    /// Bytes the cache occupies.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        let per_layer = (self.capacity * self.kv_dim * 4) as u64;
        per_layer * 2 * self.keys.len() as u64
    }

    /// A digest over the cache's populated `key`/`value` state.
    ///
    /// It covers `len` and every populated f32 in layer/position/element order,
    /// so two prefill strategies that leave the same state produce the same
    /// digest. Used as the KV-layout equivalence witness for the batched prefill.
    #[must_use]
    pub fn state_digest(&self) -> selvedge_core::Sha256Digest {
        let mut builder = selvedge_core::Sha256Builder::new();
        builder.update(&(self.len as u64).to_le_bytes());
        builder.update(&(self.kv_dim as u64).to_le_bytes());
        for layer in 0..self.keys.len() {
            let populated = self.len * self.kv_dim;
            for value in &self.keys[layer][..populated] {
                builder.update(&value.to_le_bytes());
            }
            for value in &self.values[layer][..populated] {
                builder.update(&value.to_le_bytes());
            }
        }
        builder.finish()
    }
}

#[derive(Debug)]
struct Scratch {
    x: Vec<f32>,
    residual: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    head_scores: Vec<f32>,
    head_out: Vec<f32>,
    mlp_out: Vec<f32>,
}

/// The model: weights, rotary table and scratch buffers.
#[derive(Debug)]
pub struct Model {
    /// Always-executed weights.
    pub backbone: Backbone,
    config: Qwen3Config,
    rope_inv_freq: Vec<f32>,
    hidden: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    group: usize,
    q_dim: usize,
    kv_dim: usize,
    eps: f32,
    scale: f32,
    threads: usize,
    scratch: Scratch,
}

impl Model {
    /// Build the model from a loaded backbone.
    ///
    /// # Errors
    ///
    /// Rejects an unsupported architecture, an inconsistent head split, and a
    /// zero thread count.
    pub fn new(backbone: Backbone, threads: usize) -> Result<Self> {
        let config = backbone.config.clone();
        config.validate()?;
        if threads == 0 {
            return Err(RuntimeError::config("threads", "must be at least one"));
        }
        let hidden = usize::try_from(config.hidden_size)
            .map_err(|_| RuntimeError::config("hidden_size", "does not fit in usize"))?;
        let heads = usize::try_from(config.num_attention_heads)
            .map_err(|_| RuntimeError::config("num_attention_heads", "does not fit in usize"))?;
        let kv_heads = usize::try_from(config.num_key_value_heads)
            .map_err(|_| RuntimeError::config("num_key_value_heads", "does not fit in usize"))?;
        let head_dim = usize::try_from(config.head_dim)
            .map_err(|_| RuntimeError::config("head_dim", "does not fit in usize"))?;
        let q_dim = config.q_dim()?;
        let kv_dim = config.kv_dim()?;
        let group = config.query_group()?;
        let rope_inv_freq = kernels::rope_inv_freq(head_dim, config.theta_f32()?);
        let eps = config.eps_f32()?;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        let vocab = usize::try_from(config.vocab_size)
            .map_err(|_| RuntimeError::config("vocab_size", "does not fit in usize"))?;
        let layer_count = usize::try_from(config.num_hidden_layers)
            .map_err(|_| RuntimeError::config("num_hidden_layers", "does not fit in usize"))?;
        if backbone.embed.len() != vocab * hidden {
            return Err(RuntimeError::model(
                "embedding matrix does not match vocab_size x hidden_size",
            ));
        }
        if backbone.layers.len() != layer_count {
            return Err(RuntimeError::model(
                "layer count disagrees with the configuration",
            ));
        }
        let scratch = Scratch {
            x: vec![0.0; hidden],
            residual: vec![0.0; hidden],
            q: vec![0.0; q_dim],
            k: vec![0.0; kv_dim],
            v: vec![0.0; kv_dim],
            attn: vec![0.0; q_dim],
            head_scores: Vec::new(),
            head_out: vec![0.0; head_dim],
            mlp_out: vec![0.0; hidden],
        };
        Ok(Self {
            backbone,
            config,
            rope_inv_freq,
            hidden,
            heads,
            kv_heads,
            head_dim,
            group,
            q_dim,
            kv_dim,
            eps,
            scale,
            threads,
            scratch,
        })
    }

    /// Architecture.
    #[must_use]
    pub fn config(&self) -> &Qwen3Config {
        &self.config
    }

    /// Hidden width.
    #[must_use]
    pub const fn hidden(&self) -> usize {
        self.hidden
    }

    /// Vocabulary size.
    ///
    /// # Errors
    ///
    /// Rejects a value that does not fit in `usize`.
    pub fn vocab(&self) -> Result<usize> {
        usize::try_from(self.config.vocab_size)
            .map_err(|_| RuntimeError::config("vocab_size", "does not fit in usize"))
    }

    /// Learned elements of the always-executed backbone.
    #[must_use]
    pub fn backbone_elements(&self) -> u64 {
        self.backbone.parameter_elements
    }

    /// Run one token through the model and produce logits.
    ///
    /// `position` must equal `cache.len()`; the cache is extended by one
    /// position per call.
    ///
    /// # Errors
    ///
    /// Rejects a position/cache mismatch, a token outside the vocabulary, any
    /// kernel shape failure, and any MLP execution failure.
    pub fn forward_token(
        &mut self,
        cache: &mut KvCache,
        mlp: &mut dyn MlpExecutor,
        token: u32,
        position: usize,
        trace: &mut TokenTrace,
        logits: &mut Vec<f32>,
    ) -> Result<()> {
        if position != cache.len() {
            return Err(RuntimeError::model(format!(
                "token position {position} does not match the cache length {}",
                cache.len()
            )));
        }
        if position >= cache.capacity() {
            return Err(RuntimeError::model(format!(
                "position {position} exceeds the cache capacity {}",
                cache.capacity()
            )));
        }
        let vocab = self.vocab()?;
        let token_index = usize::try_from(token)
            .map_err(|_| RuntimeError::model("token id does not fit in usize"))?;
        if token_index >= vocab {
            return Err(RuntimeError::model(format!(
                "token id {token_index} is outside the vocabulary of {vocab}"
            )));
        }

        // Embedding lookup: a copy, so the pass never aliases the tied matrix.
        let base = token_index * self.hidden;
        self.scratch
            .x
            .copy_from_slice(&self.backbone.embed[base..base + self.hidden]);

        let hidden = self.hidden;
        let head_dim = self.head_dim;
        let q_dim = self.q_dim;
        let kv_dim = self.kv_dim;
        let heads = self.heads;
        let kv_heads = self.kv_heads;
        let group = self.group;
        let scale = self.scale;
        let eps = self.eps;
        let threads = self.threads;
        let positions = position + 1;
        self.scratch.head_scores.clear();
        self.scratch
            .head_scores
            .try_reserve_exact(positions)
            .map_err(|_| RuntimeError::memory("cannot reserve attention scores"))?;
        self.scratch.head_scores.resize(positions, 0.0);

        for layer in 0..self.backbone.layers.len() {
            trace.layers = trace.layers.saturating_add(1);
            let weights = &self.backbone.layers[layer];

            // --- attention block -------------------------------------------------
            self.scratch.residual.copy_from_slice(&self.scratch.x);
            kernels::rms_norm(&mut self.scratch.x, &weights.input_ln, eps)?;
            trace.add_other((2 * hidden) as u64)?;

            kernels::matvec_into(
                &weights.q,
                q_dim,
                hidden,
                &self.scratch.x,
                &mut self.scratch.q,
                threads,
            )?;
            kernels::matvec_into(
                &weights.k,
                kv_dim,
                hidden,
                &self.scratch.x,
                &mut self.scratch.k,
                threads,
            )?;
            kernels::matvec_into(
                &weights.v,
                kv_dim,
                hidden,
                &self.scratch.x,
                &mut self.scratch.v,
                threads,
            )?;
            trace.add_matvec(q_dim as u64, hidden as u64)?;
            trace.add_matvec(kv_dim as u64, hidden as u64)?;
            trace.add_matvec(kv_dim as u64, hidden as u64)?;

            // Per-head q/k RMSNorm, then rotary embedding.
            for head in 0..heads {
                let start = head * head_dim;
                kernels::rms_norm(
                    &mut self.scratch.q[start..start + head_dim],
                    &weights.q_norm,
                    eps,
                )?;
                kernels::rope_in_place(
                    &mut self.scratch.q[start..start + head_dim],
                    &self.rope_inv_freq,
                    position,
                )?;
            }
            for head in 0..kv_heads {
                let start = head * head_dim;
                kernels::rms_norm(
                    &mut self.scratch.k[start..start + head_dim],
                    &weights.k_norm,
                    eps,
                )?;
                kernels::rope_in_place(
                    &mut self.scratch.k[start..start + head_dim],
                    &self.rope_inv_freq,
                    position,
                )?;
            }
            trace.add_other((8 * (q_dim + kv_dim)) as u64)?;

            // Append to the cache at `position`.
            let slot = position * kv_dim;
            cache.keys[layer][slot..slot + kv_dim].copy_from_slice(&self.scratch.k);
            cache.values[layer][slot..slot + kv_dim].copy_from_slice(&self.scratch.v);

            for head in 0..heads {
                let kv_head = head / group;
                let query = &self.scratch.q[head * head_dim..(head + 1) * head_dim];
                for (offset, score) in self.scratch.head_scores.iter_mut().enumerate() {
                    let key_start = offset * kv_dim + kv_head * head_dim;
                    let key = &cache.keys[layer][key_start..key_start + head_dim];
                    *score = kernels::dot(query, key)? * scale;
                }
                kernels::softmax(&mut self.scratch.head_scores)?;
                self.scratch.head_out.fill(0.0);
                for (offset, weight) in self.scratch.head_scores.iter().enumerate() {
                    let value_start = offset * kv_dim + kv_head * head_dim;
                    let value = &cache.values[layer][value_start..value_start + head_dim];
                    for (item, element) in self.scratch.head_out.iter_mut().zip(value.iter()) {
                        *item += weight * element;
                    }
                }
                let start = head * head_dim;
                self.scratch.attn[start..start + head_dim].copy_from_slice(&self.scratch.head_out);
            }
            let positions_u64 = positions as u64;
            trace.add_other(5 * heads as u64 * positions_u64)?;
            trace.add_other(2 * kv_dim as u64 * positions_u64)?;

            kernels::matvec_into(
                &weights.o,
                hidden,
                q_dim,
                &self.scratch.attn,
                &mut self.scratch.mlp_out,
                threads,
            )?;
            trace.add_matvec(hidden as u64, q_dim as u64)?;
            for (slot, value) in self.scratch.x.iter_mut().zip(self.scratch.mlp_out.iter()) {
                *slot = *value;
            }
            for (slot, value) in self.scratch.x.iter_mut().zip(self.scratch.residual.iter()) {
                *slot += *value;
            }

            // --- MLP block -------------------------------------------------------
            self.scratch.residual.copy_from_slice(&self.scratch.x);
            kernels::rms_norm(&mut self.scratch.x, &weights.post_ln, eps)?;
            trace.add_other((2 * hidden) as u64)?;
            mlp.mlp(layer, &self.scratch.x, &mut self.scratch.mlp_out, trace)?;
            for (slot, value) in self.scratch.x.iter_mut().zip(self.scratch.mlp_out.iter()) {
                *slot = *value;
            }
            for (slot, value) in self.scratch.x.iter_mut().zip(self.scratch.residual.iter()) {
                *slot += *value;
            }
        }

        cache.len = position + 1;

        // Final norm and tied output projection.
        kernels::rms_norm(&mut self.scratch.x, &self.backbone.final_norm, eps)?;
        trace.add_other((2 * hidden) as u64)?;
        logits.clear();
        logits
            .try_reserve_exact(vocab)
            .map_err(|_| RuntimeError::memory(format!("cannot reserve {vocab} logits")))?;
        logits.resize(vocab, 0.0);
        kernels::matvec_into(
            &self.backbone.embed,
            vocab,
            hidden,
            &self.scratch.x,
            logits,
            threads,
        )?;
        trace.add_matvec(vocab as u64, hidden as u64)?;
        Ok(())
    }

    /// Run a batch of consecutive tokens through the model, returning logits for
    /// every position in the batch.
    ///
    /// `start` must equal `cache.len()`, and the batch must fit the cache. The
    /// tokens are appended to the cache in order, exactly as the token-at-a-time
    /// path would append them, so a decode step that follows a batched prefill
    /// sees the same KV state it would have seen after a scalar prefill.
    ///
    /// # Bit-identity with the scalar path
    ///
    /// Every dense projection goes through [`kernels::matmul_into`], which
    /// computes each output element with the same [`kernels::dot_fixed`]
    /// reduction over `k` that [`kernels::matvec_into`] uses. Norms, RoPE,
    /// softmax and the attention value accumulation run per row with the row's
    /// own position, exactly as in [`Self::forward_token`]. The batched result is
    /// therefore bitwise identical to running the scalar path once per token;
    /// `tests/p14_parity.rs` asserts it and `docs/RUNTIME.md` states the
    /// reduction order it relies on.
    ///
    /// # Errors
    ///
    /// Rejects an empty or misplaced batch, a token outside the vocabulary, any
    /// kernel shape failure, and any MLP execution failure.
    pub fn forward_batch(
        &mut self,
        cache: &mut KvCache,
        mlp: &mut dyn BatchMlpExecutor,
        tokens: &[u32],
        start: usize,
        trace: &mut TokenTrace,
    ) -> Result<Vec<Vec<f32>>> {
        let batch = tokens.len();
        if batch == 0 {
            return Err(RuntimeError::model("an empty prefill batch"));
        }
        if start != cache.len() {
            return Err(RuntimeError::model(format!(
                "batch start {start} does not match the cache length {}",
                cache.len()
            )));
        }
        if start + batch > cache.capacity() {
            return Err(RuntimeError::model(format!(
                "a batch of {batch} at {start} exceeds the cache capacity {}",
                cache.capacity()
            )));
        }
        let vocab = self.vocab()?;
        let hidden = self.hidden;
        let head_dim = self.head_dim;
        let q_dim = self.q_dim;
        let kv_dim = self.kv_dim;
        let heads = self.heads;
        let kv_heads = self.kv_heads;
        let group = self.group;
        let scale = self.scale;
        let eps = self.eps;
        let threads = self.threads;

        let hidden_len = batch
            .checked_mul(hidden)
            .ok_or_else(|| RuntimeError::memory("batch hidden size overflows"))?;
        let q_len = batch
            .checked_mul(q_dim)
            .ok_or_else(|| RuntimeError::memory("batch q size overflows"))?;
        let kv_len = batch
            .checked_mul(kv_dim)
            .ok_or_else(|| RuntimeError::memory("batch kv size overflows"))?;
        let vocab_len = batch
            .checked_mul(vocab)
            .ok_or_else(|| RuntimeError::memory("batch logits size overflows"))?;
        let mut x = allocate(hidden_len, "batch hidden state")?;
        let mut residual = allocate(hidden_len, "batch residual")?;
        let mut q = allocate(q_len, "batch queries")?;
        let mut k = allocate(kv_len, "batch keys")?;
        let mut v = allocate(kv_len, "batch values")?;
        let mut attn = allocate(q_len, "batch attention")?;
        let mut mlp_out = allocate(hidden_len, "batch mlp output")?;
        let mut logits = allocate(vocab_len, "batch logits")?;
        let mut head_scores: Vec<f32> = Vec::new();

        for (index, token) in tokens.iter().enumerate() {
            let token_index = usize::try_from(*token)
                .map_err(|_| RuntimeError::model("token id does not fit in usize"))?;
            if token_index >= vocab {
                return Err(RuntimeError::model(format!(
                    "token id {token_index} is outside the vocabulary of {vocab}"
                )));
            }
            let base = token_index * hidden;
            x[index * hidden..(index + 1) * hidden]
                .copy_from_slice(&self.backbone.embed[base..base + hidden]);
        }

        for layer in 0..self.backbone.layers.len() {
            trace.layers = trace.layers.saturating_add(1);
            let weights = &self.backbone.layers[layer];
            let batch_u64 = batch as u64;

            // --- attention block -------------------------------------------------
            residual.copy_from_slice(&x);
            for index in 0..batch {
                kernels::rms_norm(
                    &mut x[index * hidden..(index + 1) * hidden],
                    &weights.input_ln,
                    eps,
                )?;
            }
            trace.add_other((2 * hidden * batch) as u64)?;

            kernels::matmul_into(&weights.q, q_dim, hidden, batch, &x, &mut q, threads)?;
            kernels::matmul_into(&weights.k, kv_dim, hidden, batch, &x, &mut k, threads)?;
            kernels::matmul_into(&weights.v, kv_dim, hidden, batch, &x, &mut v, threads)?;
            trace.add_matmul(batch_u64, q_dim as u64, hidden as u64)?;
            trace.add_matmul(batch_u64, kv_dim as u64, hidden as u64)?;
            trace.add_matmul(batch_u64, kv_dim as u64, hidden as u64)?;

            // Per-head q/k norm and RoPE with the token's own absolute position.
            for index in 0..batch {
                let position = start + index;
                for head in 0..heads {
                    let begin = index * q_dim + head * head_dim;
                    kernels::rms_norm(&mut q[begin..begin + head_dim], &weights.q_norm, eps)?;
                    kernels::rope_in_place(
                        &mut q[begin..begin + head_dim],
                        &self.rope_inv_freq,
                        position,
                    )?;
                }
                for head in 0..kv_heads {
                    let begin = index * kv_dim + head * head_dim;
                    kernels::rms_norm(&mut k[begin..begin + head_dim], &weights.k_norm, eps)?;
                    kernels::rope_in_place(
                        &mut k[begin..begin + head_dim],
                        &self.rope_inv_freq,
                        position,
                    )?;
                }
            }
            trace.add_other((8 * (q_dim + kv_dim) * batch) as u64)?;

            // Append to the cache, one absolute position per row, in order.
            for index in 0..batch {
                let position = start + index;
                let slot = position * kv_dim;
                cache.keys[layer][slot..slot + kv_dim]
                    .copy_from_slice(&k[index * kv_dim..(index + 1) * kv_dim]);
                cache.values[layer][slot..slot + kv_dim]
                    .copy_from_slice(&v[index * kv_dim..(index + 1) * kv_dim]);
            }

            // Attention, per row and per head, keys in ascending position order.
            for index in 0..batch {
                let position = start + index;
                let positions = position + 1;
                head_scores.clear();
                head_scores
                    .try_reserve_exact(positions)
                    .map_err(|_| RuntimeError::memory("cannot reserve attention scores"))?;
                head_scores.resize(positions, 0.0);
                for head in 0..heads {
                    let kv_head = head / group;
                    let query =
                        &q[index * q_dim + head * head_dim..index * q_dim + (head + 1) * head_dim];
                    for (offset, score) in head_scores.iter_mut().enumerate() {
                        let key_start = offset * kv_dim + kv_head * head_dim;
                        let key = &cache.keys[layer][key_start..key_start + head_dim];
                        *score = kernels::dot(query, key)? * scale;
                    }
                    kernels::softmax(&mut head_scores)?;
                    let begin = index * q_dim + head * head_dim;
                    for slot in attn[begin..begin + head_dim].iter_mut() {
                        *slot = 0.0;
                    }
                    for (offset, weight) in head_scores.iter().enumerate() {
                        let value_start = offset * kv_dim + kv_head * head_dim;
                        let value = &cache.values[layer][value_start..value_start + head_dim];
                        for (item, element) in
                            attn[begin..begin + head_dim].iter_mut().zip(value.iter())
                        {
                            *item += weight * element;
                        }
                    }
                }
                let positions_u64 = positions as u64;
                trace.add_other(5 * heads as u64 * positions_u64)?;
                trace.add_other(2 * kv_dim as u64 * positions_u64)?;
            }

            kernels::matmul_into(
                &weights.o,
                hidden,
                q_dim,
                batch,
                &attn,
                &mut mlp_out,
                threads,
            )?;
            trace.add_matmul(batch_u64, hidden as u64, q_dim as u64)?;
            for (slot, value) in x.iter_mut().zip(mlp_out.iter()) {
                *slot = *value;
            }
            for (slot, value) in x.iter_mut().zip(residual.iter()) {
                *slot += *value;
            }

            // --- MLP block -------------------------------------------------------
            residual.copy_from_slice(&x);
            for index in 0..batch {
                kernels::rms_norm(
                    &mut x[index * hidden..(index + 1) * hidden],
                    &weights.post_ln,
                    eps,
                )?;
            }
            trace.add_other((2 * hidden * batch) as u64)?;
            mlp.mlp_batch(layer, &x, &mut mlp_out, batch, trace)?;
            for (slot, value) in x.iter_mut().zip(mlp_out.iter()) {
                *slot = *value;
            }
            for (slot, value) in x.iter_mut().zip(residual.iter()) {
                *slot += *value;
            }
        }

        cache.len = start + batch;

        // Final norm per row, then the tied output projection for the whole batch:
        // the 600 MB embedding matrix is streamed once instead of once per token.
        for index in 0..batch {
            kernels::rms_norm(
                &mut x[index * hidden..(index + 1) * hidden],
                &self.backbone.final_norm,
                eps,
            )?;
        }
        trace.add_other((2 * hidden * batch) as u64)?;
        kernels::matmul_into(
            &self.backbone.embed,
            vocab,
            hidden,
            batch,
            &x,
            &mut logits,
            threads,
        )?;
        trace.add_matmul(batch as u64, vocab as u64, hidden as u64)?;
        let mut out = Vec::new();
        out.try_reserve_exact(batch)
            .map_err(|_| RuntimeError::memory("cannot reserve batch logits rows"))?;
        for index in 0..batch {
            out.push(logits[index * vocab..(index + 1) * vocab].to_vec());
        }
        Ok(out)
    }
}

/// A zeroed, exactly sized scratch buffer, refusing instead of aborting.
fn allocate(len: usize, what: &'static str) -> Result<Vec<f32>> {
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(len)
        .map_err(|_| RuntimeError::memory(format!("cannot reserve {len} f32 for the {what}")))?;
    buffer.resize(len, 0.0);
    Ok(buffer)
}

/// Greedy choice over logits, with the documented tie-break.
///
/// # Errors
///
/// Rejects an empty or non-finite logit vector.
pub fn greedy_token(logits: &[f32]) -> Result<u32> {
    let index = argmax(logits)?;
    u32::try_from(index).map_err(|_| RuntimeError::model("token index does not fit u32"))
}
