//! Typed configuration: the pinned Qwen3 architecture, the run request and the
//! streaming knobs.
//!
//! The architecture file is the checkpoint's own `config.json`. It is parsed
//! with serde and then *validated against what this runtime implements*: the
//! run refuses to start on a configuration whose semantics it does not execute
//! (a sliding window, partial rotary, untied embeddings, an attention bias, or a
//! quantized attention projection). There is no silent architecture fallback.

use std::path::{Path, PathBuf};

use selvedge_core::{Budget, ComplianceMode};
use selvedge_store::{PolicyKind, ReadOrder, StoreConfig, VerificationMode};
use serde::{Deserialize, Serialize};

use crate::error::{Result, RuntimeError};

/// Qwen3 architecture, as declared by the checkpoint's `config.json`.
///
/// Every field the forward pass depends on is required. Unknown fields are
/// ignored so that a future checkpoint revision with extra metadata still
/// parses, but every field present is validated by [`Qwen3Config::validate`].
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Qwen3Config {
    /// `model_type`, which must be `qwen3`.
    pub model_type: String,
    /// Residual width `H`.
    pub hidden_size: u64,
    /// SwiGLU intermediate width `I`.
    pub intermediate_size: u64,
    /// Number of decoder layers.
    pub num_hidden_layers: u64,
    /// Query heads.
    pub num_attention_heads: u64,
    /// Key/value heads (grouped-query attention).
    pub num_key_value_heads: u64,
    /// Per-head width.
    pub head_dim: u64,
    /// Vocabulary size.
    pub vocab_size: u64,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f64,
    /// RoPE base.
    pub rope_theta: f64,
    /// Tied input/output embeddings.
    pub tie_word_embeddings: bool,
    /// Whether the attention projections carry a bias.
    pub attention_bias: bool,
    /// Activation name; only `silu` is implemented.
    pub hidden_act: String,
    /// Maximum position the model was trained for.
    #[serde(default = "default_context")]
    pub max_position_embeddings: u64,
    /// Sliding window width, if any.
    #[serde(default)]
    pub sliding_window: Option<u64>,
    /// Whether a sliding window is enabled.
    #[serde(default)]
    pub use_sliding_window: Option<bool>,
    /// Declared checkpoint dtype (recorded, never used to change arithmetic).
    #[serde(default)]
    pub torch_dtype: Option<String>,
}

const fn default_context() -> u64 {
    40960
}

impl Qwen3Config {
    /// Read and validate `<dir>/config.json`.
    ///
    /// # Errors
    ///
    /// Rejects a missing or malformed file and every unsupported architecture.
    pub fn from_model_dir(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path).map_err(|err| {
            RuntimeError::config(
                "config.json",
                format!("cannot read {}: {err}", path.display()),
            )
        })?;
        let config: Self = serde_json::from_str(&text)?;
        config.validate()?;
        Ok(config)
    }

    /// Validate every field the forward pass relies on.
    ///
    /// # Errors
    ///
    /// Rejects an unknown architecture, a sliding window, an attention bias,
    /// untied embeddings, a non-SwiGLU activation, a head split that does not
    /// divide, an odd head dimension, and any zero width.
    pub fn validate(&self) -> Result<()> {
        if self.model_type != "qwen3" {
            return Err(RuntimeError::config(
                "model_type",
                format!(
                    "{} is not implemented; this runtime executes qwen3 only",
                    self.model_type
                ),
            ));
        }
        if self.hidden_act != "silu" {
            return Err(RuntimeError::config(
                "hidden_act",
                format!("{} is not implemented", self.hidden_act),
            ));
        }
        if !self.tie_word_embeddings {
            return Err(RuntimeError::config(
                "tie_word_embeddings",
                "only tied embeddings are implemented; an untied lm_head is a different parameter set",
            ));
        }
        if self.attention_bias {
            return Err(RuntimeError::config(
                "attention_bias",
                "biased attention projections are not implemented",
            ));
        }
        if self.sliding_window.is_some() || self.use_sliding_window == Some(true) {
            return Err(RuntimeError::config(
                "sliding_window",
                "sliding-window attention is not implemented",
            ));
        }
        for (field, value) in [
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("head_dim", self.head_dim),
            ("vocab_size", self.vocab_size),
            ("max_position_embeddings", self.max_position_embeddings),
        ] {
            if value == 0 {
                return Err(RuntimeError::config(field, "must be non-zero"));
            }
            if usize::try_from(value).is_err() {
                return Err(RuntimeError::config(field, "does not fit in usize"));
            }
        }
        if !self.head_dim.is_multiple_of(2) {
            return Err(RuntimeError::config(
                "head_dim",
                "rotary embeddings require an even head dimension",
            ));
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(RuntimeError::config(
                "num_key_value_heads",
                "must divide the query head count",
            ));
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err(RuntimeError::config(
                "rms_norm_eps",
                "must be finite and positive",
            ));
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 1.0 {
            return Err(RuntimeError::config(
                "rope_theta",
                "must be finite and greater than one",
            ));
        }
        Ok(())
    }

    /// Query width `num_attention_heads * head_dim`.
    ///
    /// # Errors
    ///
    /// Rejects overflow.
    pub fn q_dim(&self) -> Result<usize> {
        self.checked_product(self.num_attention_heads, self.head_dim, "q_dim")
    }

    /// Key/value width `num_key_value_heads * head_dim`.
    ///
    /// # Errors
    ///
    /// Rejects overflow.
    pub fn kv_dim(&self) -> Result<usize> {
        self.checked_product(self.num_key_value_heads, self.head_dim, "kv_dim")
    }

    /// Query heads sharing one key/value head.
    ///
    /// # Errors
    ///
    /// Rejects a non-dividing split.
    pub fn query_group(&self) -> Result<usize> {
        let group = self.num_attention_heads / self.num_key_value_heads;
        if group == 0 {
            return Err(RuntimeError::config(
                "num_key_value_heads",
                "more key/value heads than query heads",
            ));
        }
        usize::try_from(group)
            .map_err(|_| RuntimeError::config("num_attention_heads", "does not fit in usize"))
    }

    /// `rms_norm_eps` as f32.
    ///
    /// # Errors
    ///
    /// Rejects a value that is not representable.
    pub fn eps_f32(&self) -> Result<f32> {
        let eps = self.rms_norm_eps as f32;
        if !eps.is_finite() || eps <= 0.0 {
            return Err(RuntimeError::config("rms_norm_eps", "is not a finite f32"));
        }
        Ok(eps)
    }

    /// `rope_theta` as f32.
    ///
    /// # Errors
    ///
    /// Rejects a value that is not representable.
    pub fn theta_f32(&self) -> Result<f32> {
        let theta = self.rope_theta as f32;
        if !theta.is_finite() || theta <= 1.0 {
            return Err(RuntimeError::config("rope_theta", "is not a finite f32"));
        }
        Ok(theta)
    }

    fn checked_product(&self, left: u64, right: u64, name: &'static str) -> Result<usize> {
        let value = left
            .checked_mul(right)
            .ok_or_else(|| RuntimeError::config("head_dim", format!("{name} overflows")))?;
        usize::try_from(value)
            .map_err(|_| RuntimeError::config("head_dim", format!("{name} does not fit in usize")))
    }
}

/// Where the model, its checkpoint and its tokenizer live.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelDescriptor {
    /// Directory holding `config.json`.
    pub dir: PathBuf,
    /// Safetensors file or directory. Defaults to `<dir>`.
    pub checkpoint: Option<PathBuf>,
    /// `tokenizer.json`. Defaults to `<dir>/tokenizer.json`.
    pub tokenizer: Option<PathBuf>,
}

impl ModelDescriptor {
    /// Checkpoint path, defaulting to the model directory.
    ///
    /// # Errors
    ///
    /// Rejects a missing path.
    pub fn checkpoint_path(&self) -> Result<PathBuf> {
        let path = self.checkpoint.clone().unwrap_or_else(|| self.dir.clone());
        if !path.exists() {
            return Err(RuntimeError::config(
                "checkpoint",
                format!("{} does not exist", path.display()),
            ));
        }
        Ok(path)
    }

    /// Tokenizer path, defaulting to `<dir>/tokenizer.json`.
    ///
    /// # Errors
    ///
    /// Rejects a missing file.
    pub fn tokenizer_path(&self) -> Result<PathBuf> {
        let path = self
            .tokenizer
            .clone()
            .unwrap_or_else(|| self.dir.join("tokenizer.json"));
        if !path.exists() {
            return Err(RuntimeError::config(
                "tokenizer",
                format!("{} does not exist", path.display()),
            ));
        }
        Ok(path)
    }
}

/// Streaming knobs.
///
/// These change when and how bytes arrive and never which shards execute.
/// Court E varies exactly these fields and requires bitwise-identical logits.
#[derive(Clone, Debug)]
pub struct StreamConfig {
    /// Cache policy.
    pub policy: PolicyKind,
    /// Cache capacity in bytes (must be zero for [`PolicyKind::None`]).
    pub cache_bytes: u64,
    /// Whether to prefetch the next token's shard set before decoding it.
    pub prefetch: bool,
    /// Order in which the fetched set is visited.
    pub read_order: ReadOrder,
    /// Extra delay injected after every backing-tier read, in nanoseconds.
    ///
    /// This models a slower device without touching the arithmetic. It is
    /// recorded in every receipt so a latency number always names its delay.
    pub injected_read_delay_ns: u64,
    /// Largest shard the runtime will accept.
    pub max_shard_bytes: u64,
    /// Verification mode for shard reads.
    pub verification: VerificationMode,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            policy: PolicyKind::None,
            cache_bytes: 0,
            prefetch: false,
            read_order: ReadOrder::Plan,
            injected_read_delay_ns: 0,
            max_shard_bytes: 64 * 1024 * 1024,
            verification: VerificationMode::Always,
        }
    }
}

impl StreamConfig {
    /// Stable policy name for receipts.
    #[must_use]
    pub fn policy_name(&self) -> &'static str {
        self.policy.name()
    }

    /// Read-order name for receipts.
    #[must_use]
    pub fn order_name(&self) -> &'static str {
        match self.read_order {
            ReadOrder::Plan => "plan",
            ReadOrder::Physical => "physical",
            ReadOrder::Reverse => "reverse",
        }
    }

    /// Build the store configuration these knobs imply.
    ///
    /// # Errors
    ///
    /// Propagates [`StoreConfig::validate`].
    pub fn store_config(&self) -> Result<StoreConfig> {
        let config = StoreConfig {
            cache_bytes: self.cache_bytes,
            policy: self.policy,
            max_shard_bytes: self.max_shard_bytes,
            verification: self.verification.clone(),
            max_run_bytes: self.max_run_bytes()?,
            ..StoreConfig::default()
        };
        config.validate()?;
        Ok(config)
    }

    fn max_run_bytes(&self) -> Result<u64> {
        // A coalesced run is never larger than the largest extent it may cover,
        // but the store requires the run cap to be at least the shard cap.
        if self.max_shard_bytes == 0 {
            return Err(RuntimeError::config("max_shard_bytes", "must be non-zero"));
        }
        Ok(self.max_shard_bytes)
    }
}

/// The selector used by approximate mode.
///
/// The selector is a **control** selector: it reads the current hidden state and
/// a fixed seed-derived projection, and it reads no learned parameters, so the
/// selector's own cost is a documented FLOP count rather than a hidden weight
/// read. Its purpose in M4 part 1 is to exercise the approximate path, the
/// budget gate and the two verdicts on real data; it makes no quality claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectorSpec {
    /// Groups kept per layer. Must be in `1..=groups_per_layer`.
    pub keep_per_layer: usize,
    /// Seed for the projection directions.
    pub seed: u64,
}

impl SelectorSpec {
    /// Stable selector name recorded in receipts.
    #[must_use]
    pub fn name(&self) -> String {
        format!("projection-control/seed={}", self.seed)
    }
}

/// Which execution mode a run uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Every group of every layer, proven against the bundle's declaration.
    Exact,
    /// A selector-chosen subset, labelled approximate.
    Approximate(SelectorSpec),
}

impl ExecutionMode {
    /// Whether this mode may produce a semantic-exactness proof.
    #[must_use]
    pub const fn is_exact(self) -> bool {
        matches!(self, Self::Exact)
    }

    /// Stable label for receipts.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Approximate(_) => "approximate",
        }
    }
}

/// Prompt input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PromptSource {
    /// Raw text, encoded with no chat template and no added special tokens.
    Text(String),
    /// Explicit token ids; no tokenizer is consulted.
    Ids(Vec<u32>),
    /// Text wrapped in the documented Qwen3 chat template.
    Chat {
        /// User turn text.
        user: String,
    },
}

impl PromptSource {
    /// Stable label for receipts.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Text(_) => "text",
            Self::Ids(_) => "ids",
            Self::Chat { .. } => "chat",
        }
    }
}

/// Comparison tolerance against a reference logits dump.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tolerance {
    /// Absolute tolerance.
    pub abs: f64,
    /// Relative tolerance.
    pub rel: f64,
}

impl Default for Tolerance {
    fn default() -> Self {
        Self {
            abs: 1.0e-3,
            rel: 1.0e-3,
        }
    }
}

impl Tolerance {
    /// Whether an element passes.
    #[must_use]
    pub fn accepts(&self, reference: f64, observed: f64) -> bool {
        let difference = (reference - observed).abs();
        difference <= self.abs + self.rel * reference.abs()
    }
}

/// A complete run request.
#[derive(Clone, Debug)]
pub struct RunConfig {
    /// Model location.
    pub model: ModelDescriptor,
    /// Bundle to execute.
    pub bundle: PathBuf,
    /// Execution mode.
    pub mode: ExecutionMode,
    /// Streaming knobs.
    pub stream: StreamConfig,
    /// Active-parameter budget.
    pub budget: Budget,
    /// Whether an over-budget approximate path may be reduced to fit.
    pub fallback: bool,
    /// Worker threads used by the dense kernels.
    pub threads: usize,
    /// JSONL accounting destination.
    pub account: Option<PathBuf>,
    /// Prompt.
    pub prompt: PromptSource,
    /// Maximum number of generated tokens.
    pub max_new_tokens: usize,
    /// Stop-token set for greedy decoding.
    pub stop_tokens: Vec<u32>,
    /// Optional reference logits dump to compare against.
    pub compare_logits: Option<PathBuf>,
    /// Comparison tolerance.
    pub tolerance: Tolerance,
    /// Optional raw f32 logits dump for this run.
    pub dump_logits: Option<PathBuf>,
}

impl RunConfig {
    /// A run request with the documented defaults: compliance budget of one
    /// billion active unique parameters, greedy decoding, no cache.
    ///
    /// # Errors
    ///
    /// Rejects a zero budget.
    pub fn new(
        model: ModelDescriptor,
        bundle: PathBuf,
        mode: ExecutionMode,
        prompt: PromptSource,
    ) -> Result<Self> {
        Ok(Self {
            model,
            bundle,
            mode,
            stream: StreamConfig::default(),
            budget: Budget::new(1_000_000_000, ComplianceMode::Compliance)?,
            fallback: false,
            threads: 1,
            account: None,
            prompt,
            max_new_tokens: 8,
            stop_tokens: Vec::new(),
            compare_logits: None,
            tolerance: Tolerance::default(),
            dump_logits: None,
        })
    }

    /// Validate the request before any file is opened.
    ///
    /// # Errors
    ///
    /// Rejects a zero thread count, an untuned approximate selector, a missing
    /// stop set for chat mode, a zero token budget, and an over-budget exact
    /// path that has no fallback.
    pub fn validate(&self) -> Result<()> {
        if self.threads == 0 {
            return Err(RuntimeError::config("threads", "must be at least one"));
        }
        if self.tolerance.abs < 0.0 || self.tolerance.rel < 0.0 {
            return Err(RuntimeError::config("tolerance", "must be non-negative"));
        }
        if let ExecutionMode::Approximate(spec) = self.mode
            && spec.keep_per_layer == 0
        {
            return Err(RuntimeError::config(
                "selector.keep_per_layer",
                "must be at least one group",
            ));
        }
        if self.stream.prefetch && self.stream.policy == PolicyKind::None {
            return Err(RuntimeError::config(
                "stream.prefetch",
                "prefetch needs a cache policy; with 'none' it would read every shard twice",
            ));
        }
        self.stream.store_config()?;
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn config() -> Qwen3Config {
        Qwen3Config {
            model_type: "qwen3".to_owned(),
            hidden_size: 1024,
            intermediate_size: 3072,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            vocab_size: 151_936,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            tie_word_embeddings: true,
            attention_bias: false,
            hidden_act: "silu".to_owned(),
            max_position_embeddings: 40960,
            sliding_window: None,
            use_sliding_window: Some(false),
            torch_dtype: Some("bfloat16".to_owned()),
        }
    }

    #[test]
    fn pinned_architecture_validates_and_splits_heads() {
        let config = config();
        config.validate().unwrap();
        assert_eq!(config.q_dim().unwrap(), 2048);
        assert_eq!(config.kv_dim().unwrap(), 1024);
        assert_eq!(config.query_group().unwrap(), 2);
    }

    #[test]
    fn unsupported_architecture_is_rejected_not_approximated() {
        let mut untied = config();
        untied.tie_word_embeddings = false;
        assert!(untied.validate().is_err());
        let mut sliding = config();
        sliding.sliding_window = Some(4096);
        assert!(sliding.validate().is_err());
        let mut biased = config();
        biased.attention_bias = true;
        assert!(biased.validate().is_err());
        let mut split = config();
        split.num_key_value_heads = 5;
        assert!(split.validate().is_err());
    }
}
