//! Greedy tokenization through the pinned `tokenizer.json`.
//!
//! The tokenizer is the Hugging Face `tokenizers` crate at a pinned version with
//! the vendored Oniguruma regex engine, which is the same engine the reference
//! Python tokenizer uses. Decoding is greedy only: this runtime has no sampling
//! path, so there is no RNG and no temperature to document.
//!
//! The chat form is the Qwen3 template, written out literally rather than
//! interpreted:
//!
//! ```text
//! <|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n
//! ```
//!
//! Its two special tokens are added tokens of the pinned vocabulary, so they are
//! tokenized from the text rather than injected by a post-processor (the pinned
//! `tokenizer.json` has none).

use std::collections::BTreeSet;
use std::path::Path;

use crate::error::{Result, RuntimeError};

/// Chat template pieces, verbatim.
pub const CHAT_PREFIX: &str = "<|im_start|>user\n";
/// Closing piece before the assistant turn.
pub const CHAT_TURN_END: &str = "<|im_end|>\n<|im_start|>assistant\n";

/// A tokenizer restricted to what greedy decoding needs.
pub struct GreedyTokenizer {
    inner: tokenizers::Tokenizer,
    stop: BTreeSet<u32>,
}

impl std::fmt::Debug for GreedyTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GreedyTokenizer")
            .field("stop", &self.stop)
            .finish_non_exhaustive()
    }
}

impl GreedyTokenizer {
    /// Load a `tokenizer.json`.
    ///
    /// # Errors
    ///
    /// Rejects a missing, malformed or unsupported tokenizer file.
    pub fn from_file(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Err(RuntimeError::Tokenizer(format!(
                "{} does not exist",
                path.display()
            )));
        }
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|err| RuntimeError::Tokenizer(format!("{}: {err}", path.display())))?;
        Ok(Self {
            inner,
            stop: BTreeSet::new(),
        })
    }

    /// Set the stop-token set.
    #[must_use]
    pub fn with_stop_tokens(mut self, tokens: impl IntoIterator<Item = u32>) -> Self {
        self.stop = tokens.into_iter().collect();
        self
    }

    /// Encode text with no added special tokens.
    ///
    /// # Errors
    ///
    /// Rejects an encoding failure and any token id that does not fit `u32`.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|err| RuntimeError::Tokenizer(format!("encode failed: {err}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Encode a user turn with the documented Qwen3 chat template.
    ///
    /// # Errors
    ///
    /// Propagates [`GreedyTokenizer::encode`].
    pub fn encode_chat(&self, user: &str) -> Result<Vec<u32>> {
        let text = format!("{CHAT_PREFIX}{user}{CHAT_TURN_END}");
        self.encode(&text)
    }

    /// Decode token ids, skipping special tokens.
    ///
    /// # Errors
    ///
    /// Rejects a decoding failure and an id that does not fit `u32`.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let ids: Vec<u32> = ids.to_vec();
        self.inner
            .decode(&ids, true)
            .map_err(|err| RuntimeError::Tokenizer(format!("decode failed: {err}")))
    }

    /// Whether `id` stops generation.
    #[must_use]
    pub fn is_stop(&self, id: u32) -> bool {
        self.stop.contains(&id)
    }

    /// The configured stop set.
    #[must_use]
    pub fn stop_tokens(&self) -> Vec<u32> {
        self.stop.iter().copied().collect()
    }

    /// Vocabulary size implied by the tokenizer.
    ///
    /// # Errors
    ///
    /// Rejects a vocabulary larger than `u32`.
    pub fn vocabulary_size(&self) -> Result<u32> {
        u32::try_from(self.inner.get_vocab_size(true))
            .map_err(|_| RuntimeError::Tokenizer("vocabulary does not fit u32".to_owned()))
    }
}
