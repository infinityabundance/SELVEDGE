//! Shared helpers for the runtime integration courts.
//!
//! The tiny fixture is converted into a real bundle on disk through the S2
//! converter, so every court exercises the format, the validator, the store and
//! the runtime together instead of a mock.

// Test support: a panic here is the test failing, which is the intended report.
#![allow(dead_code, unreachable_pub, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};

use selvedge_convert::{
    CheckpointSource, ConversionOptions, GroupSpec, TensorSource, convert, discover_ffn_maps,
    model_digest_for_path, plan_conversion,
};
use selvedge_core::{Budget, ComplianceMode};
use selvedge_runtime::{
    ExecutionMode, ModelDescriptor, PromptSource, Qwen3Config, RunConfig, SelectorSpec,
    StreamConfig,
};
use selvedge_store::PolicyKind;

/// The committed tiny fixture.
pub fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tiny-qwen3")
}

/// Convert the fixture into a bundle inside `out`, returning the bundle path.
pub fn build_tiny_bundle(out: &Path) -> PathBuf {
    let fixture = fixture_dir();
    let checkpoint = fixture.join("model.safetensors");
    let mut source = CheckpointSource::open(&checkpoint).expect("open checkpoint");
    let keys = source.tensor_keys();
    let discovered = discover_ffn_maps(&keys).expect("discover FFN");
    let digest = model_digest_for_path(&checkpoint).expect("digest");
    let options = ConversionOptions {
        complete: true,
        ..ConversionOptions::default()
    };
    let plan = plan_conversion(
        &source,
        &discovered,
        &GroupSpec::ByCount(4),
        digest,
        &options,
    )
    .expect("plan conversion");
    let (published, _report) = convert(&mut source, &plan, out, &options).expect("convert");
    published.path
}

/// A run configuration for the tiny fixture.
pub fn tiny_config(bundle: &Path, prompt: Vec<u32>, max_new_tokens: usize) -> RunConfig {
    let fixture = fixture_dir();
    let model = ModelDescriptor {
        dir: fixture.clone(),
        checkpoint: Some(fixture.join("model.safetensors")),
        tokenizer: None,
    };
    let mut config = RunConfig::new(
        model,
        bundle.to_path_buf(),
        ExecutionMode::Exact,
        PromptSource::Ids(prompt),
    )
    .expect("config");
    config.stream = StreamConfig {
        policy: PolicyKind::Lru,
        cache_bytes: 1 << 20,
        ..StreamConfig::default()
    };
    config.max_new_tokens = max_new_tokens;
    config.threads = 2;
    config
}

/// A bounded compliance budget for the tiny model.
#[must_use]
pub fn tiny_budget() -> Budget {
    Budget::new(64 * 1024 * 1024, ComplianceMode::Compliance).expect("budget")
}

/// The fixture's architecture.
pub fn tiny_architecture() -> Qwen3Config {
    Qwen3Config::from_model_dir(&fixture_dir()).expect("config.json")
}

/// A selector spec that keeps one group per layer.
#[must_use]
pub fn one_group_selector() -> SelectorSpec {
    SelectorSpec {
        keep_per_layer: 1,
        seed: 0x5E1_DE6E,
    }
}

/// A temporary directory that removes itself.
pub struct TempDir(PathBuf);

impl TempDir {
    /// Create a fresh directory under the system temporary directory.
    #[must_use]
    pub fn new(tag: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "selvedge-runtime-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("create temp dir");
        Self(path)
    }

    /// The directory.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
