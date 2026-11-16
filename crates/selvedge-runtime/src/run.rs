//! The runner: one device, one sequence, greedy decoding, with the budget gate
//! in front of execution and a JSONL audit trail behind it.
//!
//! # Order of operations for one token
//!
//! 1. The embedding row and the always-executed backbone are admitted as one
//!    selection path (`admit_selection_path`), **before** any byte is read or any
//!    kernel runs. A path that exceeds the ceiling has no value that execution
//!    could accept.
//! 2. For each layer the mode chooses the groups (all of them in exact mode), and
//!    the *cumulative* path — everything already admitted plus this layer's
//!    fragments — is re-admitted before this layer's shards are read. An
//!    over-budget path either fails the run or, in approximate mode with the
//!    documented fallback enabled, is reduced by dropping the lowest-scoring
//!    groups and re-admitted; a fallback that is still over budget is refused.
//! 3. Groups execute in ascending group order, one at a time, through the store.
//! 4. The token's shard set is classified: exact mode proves it equal to the
//!    manifest's declaration, approximate mode records the gap.
//! 5. The ledger snapshot becomes one JSONL row.
//!
//! Nothing in steps 1-5 depends on the wall clock, on the cache policy, or on how
//! many threads the kernels use.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::Instant;

use selvedge_core::{
    BudgetError, BudgetVerdict, ComplianceMode, DecisionPlan, FlopEstimate, ParamIdentity,
    ParamTensorId, ParamUse, ParameterLedger, SelectionPath, SemanticExactness, Sha256Digest,
    StorageTier, TokenAccounting,
};
use selvedge_store::{ExecutionScope, ReadOrder, ShardStore, StoreStats};

use crate::account::{
    ACCOUNTING_SCHEMA, AccountingRow, AccountingTotals, AccountingWriter, Latency, Phase,
    RUN_SCHEMA, RunSummary, StreamSummary, require_tokens,
};
use crate::bundle::{BundlePlan, GroupPlan, GroupReader, GroupTensors, ShardFragment};
use crate::config::{ExecutionMode, ModelDescriptor, PromptSource, Qwen3Config, RunConfig};
use crate::error::{Result, RuntimeError};
use crate::kernels;
use crate::model::{KvCache, MlpExecutor, Model, TokenTrace, greedy_token};
use crate::modes::{GroupChoice, ModeState, READ_AUDIT};
use crate::tokenizer::GreedyTokenizer;
use crate::weights::Backbone;

/// Cache-resident bytes represented as one identity in the ledger.
///
/// The ledger models residency per [`ParamIdentity`], so the always-resident
/// backbone is recorded tensor by tensor and the bundle's residency is recorded
/// as one identity whose element count is the *measured* resident bytes divided
/// by four. Every bundle shard is unquantized f32 by construction, so that
/// division is exact. The identity is deliberately not tied to a checkpoint
/// tensor: the cache does not report which shards survived its own last
/// eviction, and naming one would be a fabricated attribution.
const CACHE_IDENTITY_KEY: &str = "selvedge.runtime.cache.resident";

/// One token's execution result.
#[derive(Clone, Debug)]
pub struct TokenStep {
    /// Logits for the next token.
    pub logits: Vec<f32>,
    /// The mandate's accounting record.
    pub accounting: TokenAccounting,
    /// FLOP estimate for this token.
    pub trace: TokenTrace,
    /// Wall time spent in the forward pass.
    pub elapsed_ns: u64,
    /// Digest of the f32 logits.
    pub digest: Sha256Digest,
    /// Group shards executed.
    pub group_shards: u64,
    /// Exactness classification for this token.
    pub exactness: SemanticExactness,
}

/// Comparison against a reference logits dump.
#[derive(Clone, Debug, serde::Serialize)]
pub struct LogitsComparison {
    /// Reference file.
    pub reference: String,
    /// Positions compared.
    pub positions: u64,
    /// Vocabulary compared per position.
    pub vocab: u64,
    /// Largest absolute difference.
    pub max_abs: f64,
    /// Largest relative difference on the reference value.
    pub max_rel: f64,
    /// Positions whose argmax agrees.
    pub top1_matches: u64,
    /// Absolute tolerance applied.
    pub abs: f64,
    /// Relative tolerance applied.
    pub rel: f64,
    /// Whether every element passed.
    pub passed: bool,
}

/// What a complete run produced.
#[derive(Clone, Debug)]
pub struct RunOutcome {
    /// The serializable summary.
    pub summary: RunSummary,
    /// Prefill logits, one vector per prompt token, when retention was asked for.
    pub prefill_logits: Vec<Vec<f32>>,
}

impl RunOutcome {
    /// Whether every gate and comparison passed.
    #[must_use]
    pub fn ok(&self) -> bool {
        self.summary.ok
    }
}

/// Everything one forward pass hands to the accounting step.
struct TokenOutcomeInput {
    token: u32,
    position: usize,
    phase: Phase,
    elapsed_ns: u64,
    trace: TokenTrace,
    logits: Vec<f32>,
    controller_flops: u64,
}

/// Counters the MLP executor owns for the duration of one forward pass.
#[derive(Clone, Copy, Debug, Default)]
struct ExecutorState {
    controller_flops: u64,
    admitted_flops: u64,
    fallback_applied: bool,
}

/// The runner.
pub struct Runner {
    config: RunConfig,
    architecture: Qwen3Config,
    plan: BundlePlan,
    store: ShardStore,
    model: Model,
    mode: ModeState,
    reader: GroupReader,
    cache: KvCache,
    writer: AccountingWriter,
    tokenizer: Option<GreedyTokenizer>,
    total_accessible: u64,
    backbone_uses: Vec<ParamUse>,
    backbone_resident: Vec<ParamIdentity>,
    offsets: BTreeMap<selvedge_core::ShardId, u64>,
    token_uses: Vec<ParamUse>,
    token_ledger: ParameterLedger,
    mlp_gate: Vec<f32>,
    mlp_up: Vec<f32>,
    fallback_applied: bool,
    totals: AccountingTotals,
    prefill_ns: Vec<u64>,
    decode_ns: Vec<u64>,
    logits_digests: Vec<String>,
    exactness: Option<SemanticExactness>,
    verdict: Option<selvedge_core::ArmVerdict>,
}

impl std::fmt::Debug for Runner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Runner")
            .field("mode", &self.mode.label())
            .field("bundle_id", &self.plan.bundle_id)
            .field("layers", &self.plan.layers.len())
            .finish_non_exhaustive()
    }
}

/// Digest a logits vector's exact f32 bytes.
///
/// The digest covers the little-endian f32 bit patterns, so a different
/// reduction order shows up as a different digest. Court E compares these
/// digests for bitwise identity.
#[must_use]
pub fn logits_digest(logits: &[f32]) -> Sha256Digest {
    let mut builder = selvedge_core::Sha256Builder::new();
    for value in logits {
        builder.update(&value.to_le_bytes());
    }
    builder.finish()
}

/// Read the documented stop set from a `generation_config.json`, if present.
///
/// The pinned Qwen3 control declares `eos_token_id: [151645, 151643]`. A missing
/// file yields an empty set, which the caller must then supply explicitly;
/// generation never invents a stop token.
#[must_use]
pub fn default_stop_tokens(model_dir: &Path) -> Vec<u32> {
    let path = model_dir.join("generation_config.json");
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
        return Vec::new();
    };
    let Some(eos) = value.get("eos_token_id") else {
        return Vec::new();
    };
    let mut out = Vec::new();
    if let Some(single) = eos.as_u64().and_then(|value| u32::try_from(value).ok()) {
        out.push(single);
    }
    if let Some(list) = eos.as_array() {
        for item in list {
            if let Some(id) = item.as_u64().and_then(|value| u32::try_from(value).ok()) {
                out.push(id);
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Describe a model directory without executing it.
///
/// # Errors
///
/// Rejects a missing or unsupported `config.json`.
pub fn describe_model(model: &ModelDescriptor) -> Result<Qwen3Config> {
    Qwen3Config::from_model_dir(&model.dir)
}

impl Runner {
    /// Open everything a run needs and validate that the pieces agree.
    ///
    /// # Errors
    ///
    /// Rejects an invalid configuration, a model that does not load, a bundle
    /// that is not a complete partition, a bundle and checkpoint that disagree on
    /// shape, a selector that cannot be honoured, and a prompt that needs a
    /// tokenizer which is not present.
    pub fn open(config: RunConfig) -> Result<Self> {
        config.validate()?;
        let architecture = Qwen3Config::from_model_dir(&config.model.dir)?;
        let tokenizer = load_tokenizer(&config)?;
        let prompt_ids = resolve_prompt(&config, tokenizer.as_ref())?;
        if prompt_ids.is_empty() {
            return Err(RuntimeError::config("prompt", "encodes to no tokens"));
        }

        let checkpoint = config.model.checkpoint_path()?;
        let backbone = Backbone::load(&checkpoint, &architecture)?;
        let store = ShardStore::open(&config.bundle, config.stream.store_config()?)?;
        let plan =
            BundlePlan::from_manifest(store.manifest(), store.manifest_sha256(), &architecture)?;
        if plan.hidden != architecture.hidden_size {
            return Err(RuntimeError::bundle(format!(
                "bundle hidden width {} disagrees with the checkpoint's {}",
                plan.hidden, architecture.hidden_size
            )));
        }
        if plan.intermediate != architecture.intermediate_size {
            return Err(RuntimeError::bundle(format!(
                "bundle partitions {} intermediate channels, the checkpoint declares {}",
                plan.intermediate, architecture.intermediate_size
            )));
        }

        let mode = ModeState::new(&plan, config.mode)?;
        let scope = plan.native_scope()?;
        let reader = GroupReader::new(scope, &config.stream)?;
        let model = Model::new(backbone, config.threads)?;

        let capacity = prompt_ids
            .len()
            .checked_add(config.max_new_tokens)
            .ok_or_else(|| RuntimeError::memory("sequence length overflows"))?;
        let cache = KvCache::new(&architecture, capacity)?;

        let total_accessible = model
            .backbone_elements()
            .checked_add(plan.total_elements())
            .ok_or_else(|| RuntimeError::model("total accessible element count overflows"))?;
        let token_ledger = ParameterLedger::begin(config.budget, total_accessible)?;

        let writer = match &config.account {
            Some(path) => AccountingWriter::create(path)?,
            None => AccountingWriter::disabled(),
        };

        let backbone_resident: Vec<ParamIdentity> = model
            .backbone
            .uses
            .iter()
            .map(|use_value| use_value.identity.clone())
            .collect();
        let backbone_uses = model.backbone.uses.clone();
        let offsets: BTreeMap<selvedge_core::ShardId, u64> = store
            .manifest()
            .shards
            .iter()
            .map(|record| (record.shard_id, record.offset))
            .collect();

        Ok(Self {
            config,
            architecture,
            plan,
            store,
            model,
            mode,
            reader,
            cache,
            writer,
            tokenizer,
            total_accessible,
            backbone_uses,
            backbone_resident,
            offsets,
            token_uses: Vec::new(),
            token_ledger,
            mlp_gate: Vec::new(),
            mlp_up: Vec::new(),
            fallback_applied: false,
            totals: AccountingTotals::default(),
            prefill_ns: Vec::new(),
            decode_ns: Vec::new(),
            logits_digests: Vec::new(),
            exactness: None,
            verdict: None,
        })
    }

    /// The architecture the run executes.
    #[must_use]
    pub fn architecture(&self) -> &Qwen3Config {
        &self.architecture
    }

    /// The derived execution plan.
    #[must_use]
    pub fn plan(&self) -> &BundlePlan {
        &self.plan
    }

    /// The model.
    #[must_use]
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// The store, for read-only inspection in tests and receipts.
    #[must_use]
    pub fn store(&self) -> &ShardStore {
        &self.store
    }

    /// The mode state.
    #[must_use]
    pub fn mode(&self) -> &ModeState {
        &self.mode
    }

    /// The key/value cache.
    #[must_use]
    pub fn cache(&self) -> &KvCache {
        &self.cache
    }

    /// The run configuration.
    #[must_use]
    pub fn config(&self) -> &RunConfig {
        &self.config
    }

    /// The tokenizer, when one is loaded.
    #[must_use]
    pub fn tokenizer(&self) -> Option<&GreedyTokenizer> {
        self.tokenizer.as_ref()
    }

    /// Prompt ids this run will use.
    ///
    /// # Errors
    ///
    /// Rejects a prompt that needs a tokenizer which is not loaded.
    pub fn prompt_ids(&self) -> Result<Vec<u32>> {
        resolve_prompt(&self.config, self.tokenizer.as_ref())
    }

    /// Adjusted accounting totals so far.
    #[must_use]
    pub fn totals(&self) -> &AccountingTotals {
        &self.totals
    }

    /// Run one forward pass for `token` at `position`.
    ///
    /// # Errors
    ///
    /// Rejects a position/cache mismatch, an over-budget path, a failed witness,
    /// and any store or kernel failure.
    pub fn forward(&mut self, token: u32, position: usize, phase: Phase) -> Result<TokenStep> {
        let started = Instant::now();
        self.token_uses.clear();
        self.token_ledger = ParameterLedger::begin(self.config.budget, self.total_accessible)?;
        self.reader.reset_token();

        // Step 1: the always-executed path, admitted before anything runs.
        let mut path_uses = self.model.backbone.embedding_uses(token)?;
        path_uses.extend(self.backbone_uses.iter().cloned());
        let planned = self.planned_flops()?;
        self.admit(&path_uses, 0, planned)?;
        for use_value in &path_uses {
            self.token_uses.push(use_value.clone());
            self.token_ledger.record_use(use_value.clone())?;
        }

        let mut trace = TokenTrace::default();
        let mut logits = Vec::new();
        let mut state = ExecutorState {
            controller_flops: 0,
            admitted_flops: planned,
            fallback_applied: self.fallback_applied,
        };
        if self.mode.is_exact() {
            // Exact mode's scope is the whole native set, fixed for the run.
            self.reader.replace_scope(self.plan.native_scope()?);
        }
        {
            let mut mlp = LayerMlp {
                plan: &self.plan,
                store: &self.store,
                mode: &mut self.mode,
                reader: &mut self.reader,
                ledger: &mut self.token_ledger,
                token_uses: &mut self.token_uses,
                gate_buf: &mut self.mlp_gate,
                up_buf: &mut self.mlp_up,
                offsets: &self.offsets,
                threads: self.config.threads,
                fallback: self.config.fallback,
                prefetch: self.config.stream.prefetch,
                read_order: self.config.stream.read_order,
                state: &mut state,
            };
            self.model.forward_token(
                &mut self.cache,
                &mut mlp,
                token,
                position,
                &mut trace,
                &mut logits,
            )?;
        }
        self.fallback_applied = state.fallback_applied;

        let elapsed_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.finish_token(TokenOutcomeInput {
            token,
            position,
            phase,
            elapsed_ns,
            trace,
            logits,
            controller_flops: state.controller_flops,
        })
    }

    fn finish_token(&mut self, input: TokenOutcomeInput) -> Result<TokenStep> {
        let TokenOutcomeInput {
            token,
            position,
            phase,
            elapsed_ns,
            trace,
            logits,
            controller_flops,
        } = input;
        let executed: BTreeSet<selvedge_core::ShardId> = self.reader.executed().clone();
        let exactness = self.mode.classify_token(&self.plan, &executed)?;
        if self.mode.is_exact() && !exactness.is_proven() {
            return Err(RuntimeError::exactness(
                "exact mode produced a token whose executed set is not the declared native set",
            ));
        }

        let mut estimate = trace.estimate()?;
        estimate.controller_flops = controller_flops;
        self.token_ledger.record_flops(&estimate)?;
        for identity in &self.backbone_resident {
            self.token_ledger.record_resident(identity.clone());
        }
        let resident_bytes = self.store.cache_stats().used_bytes;
        if resident_bytes > 0 {
            let elements = resident_bytes / 4;
            let tensor = ParamTensorId::new(CACHE_IDENTITY_KEY)?;
            self.token_ledger
                .record_resident(ParamIdentity::new(tensor, 0..elements)?);
        }
        let accounting = self.token_ledger.snapshot()?;
        if !accounting.within_budget {
            return Err(RuntimeError::budget(format!(
                "token at position {position} executed {} active unique elements above the ceiling {}; refusing to emit it",
                accounting.active_unique_params, accounting.budget_limit
            )));
        }

        // The same path, as a plan, checked again by the ledger's own gate. The
        // verdict is recorded in the row; execution already happened under the
        // pre-execution admissions, so this is a consistency record, not
        // permission.
        let mut post_plan = DecisionPlan::new(selvedge_core::SelectionDecision::new(
            executed.iter().copied().collect(),
            selvedge_core::SelectionProvenance::approximate("runtime:post-hoc", Vec::new())?,
        )?);
        for use_value in &self.token_uses {
            post_plan.push_use(use_value.clone())?;
        }
        let budget_verdict = self
            .token_ledger
            .check_decision(&post_plan)
            .map_err(|error| RuntimeError::budget(error.to_string()))?;
        let plan_unique = match &budget_verdict {
            BudgetVerdict::Within {
                active_unique_params,
                ..
            }
            | BudgetVerdict::NoncompliantTeacher {
                active_unique_params,
                ..
            } => *active_unique_params,
        };
        if plan_unique != accounting.active_unique_params {
            return Err(RuntimeError::budget(format!(
                "plan union {plan_unique} disagrees with the ledger union {}",
                accounting.active_unique_params
            )));
        }

        let digest = logits_digest(&logits);
        let record = format!("{}:{}", phase.label(), position);
        if let Some(verdict) = self.mode.verdict(&record, &accounting, controller_flops)? {
            self.verdict = Some(verdict);
        }
        self.exactness = Some(exactness.clone());

        let row = AccountingRow {
            schema: ACCOUNTING_SCHEMA,
            phase,
            position: position as u64,
            token_id: token,
            elapsed_ns,
            mode: self.mode.label(),
            bundle_id: self.plan.bundle_id.clone(),
            layers_executed: trace.layers,
            group_shards_executed: executed.len() as u64,
            read_calls: self.reader.read_calls(),
            cache_hits: self.reader.cache_hits(),
            bytes_device: self.reader.bytes_device(),
            groups_trimmed: self.mode.trimmed_groups(),
            budget: budget_verdict,
            accounting: accounting.clone(),
        };
        self.totals.observe(&row);
        self.writer.write(&row)?;
        match phase {
            Phase::Prefill => self.prefill_ns.push(elapsed_ns),
            Phase::Decode => self.decode_ns.push(elapsed_ns),
        }
        self.logits_digests.push(digest.to_hex());

        Ok(TokenStep {
            logits,
            accounting,
            trace,
            elapsed_ns,
            digest,
            group_shards: executed.len() as u64,
            exactness,
        })
    }

    /// Planned FLOPs for one token, from the plan and the mode.
    ///
    /// This is the estimate that travels with the pre-execution admission. The
    /// authoritative per-token FLOPs are the ones [`TokenTrace`] records from the
    /// slices that actually ran.
    fn planned_flops(&self) -> Result<u64> {
        let hidden = self.plan.hidden;
        let q_dim = self.architecture.q_dim()? as u64;
        let kv_dim = self.architecture.kv_dim()? as u64;
        let vocab = self.architecture.vocab_size;
        let layers = self.plan.layers.len() as u64;
        let per_layer = 2 * q_dim * hidden + 4 * kv_dim * hidden + 2 * hidden * q_dim;
        let groups_total = u64::try_from(self.plan.layers[0].groups.len())
            .map_err(|_| RuntimeError::model("group count does not fit u64"))?;
        let selected_channels = match self.config.mode {
            ExecutionMode::Exact => self.plan.intermediate,
            ExecutionMode::Approximate(spec) => {
                let keep = u64::try_from(spec.keep_per_layer)
                    .map_err(|_| RuntimeError::model("selector size does not fit u64"))?;
                (keep * (self.plan.intermediate / groups_total.max(1))).min(self.plan.intermediate)
            }
        };
        let group_flops = 6 * selected_channels * hidden;
        let total = layers
            .checked_mul(per_layer + group_flops)
            .and_then(|value| value.checked_add(2 * vocab * hidden))
            .ok_or_else(|| RuntimeError::model("planned FLOPs overflow"))?;
        Ok(total)
    }

    /// Admit a selection path, refusing an over-budget one.
    fn admit(
        &self,
        execution_uses: &[ParamUse],
        selector_flops: u64,
        execution_flops: u64,
    ) -> Result<()> {
        self.try_admit(execution_uses, selector_flops, execution_flops)
            .map_err(|error| {
                RuntimeError::budget(format!(
                    "budget gate refused the path before execution: {error}"
                ))
            })
    }

    /// Admit a path, or hand back the gate's error so a caller can fall back.
    fn try_admit(
        &self,
        execution_uses: &[ParamUse],
        selector_flops: u64,
        execution_flops: u64,
    ) -> core::result::Result<(), BudgetError> {
        let path = SelectionPath::new(
            Vec::new(),
            execution_uses.to_vec(),
            FlopEstimate {
                matmul_flops: 0,
                other_flops: 0,
                controller_flops: selector_flops,
            },
            FlopEstimate {
                matmul_flops: execution_flops,
                other_flops: 0,
                controller_flops: 0,
            },
        )
        .map_err(|error| BudgetError::InvalidPlan {
            reason: error.to_string(),
        })?;
        self.token_ledger
            .admit_selection_path(path)
            .map(|_admitted| ())
    }

    /// Run the whole request: prefill, then greedy decoding.
    ///
    /// # Errors
    ///
    /// Propagates any validation, store, budget, witness or comparison failure.
    pub fn run(&mut self) -> Result<RunOutcome> {
        let prompt = self.prompt_ids()?;
        let retain = self.config.compare_logits.is_some() || self.config.dump_logits.is_some();
        let mut prefill_logits = Vec::new();
        let mut last = Vec::new();
        for (position, token) in prompt.iter().enumerate() {
            let step = self.forward(*token, position, Phase::Prefill)?;
            last = step.logits;
            if retain {
                prefill_logits.push(last.clone());
            }
        }

        let mut generated = Vec::new();
        let mut next = greedy_token(&last)?;
        for index in 0..self.config.max_new_tokens {
            if self.config.stop_tokens.contains(&next) {
                break;
            }
            generated.push(next);
            let position = prompt.len() + index;
            let step = self.forward(next, position, Phase::Decode)?;
            last = step.logits;
            next = greedy_token(&last)?;
        }

        let comparison = match &self.config.compare_logits {
            Some(path) => {
                let comparison = self.compare_logits(path, &prefill_logits)?;
                if !comparison.passed {
                    let summary =
                        self.finish_summary(&prompt, &generated, Some(comparison.clone()))?;
                    let _ = summary;
                    return Err(RuntimeError::model(format!(
                        "logits disagree with the reference beyond tolerance: max_abs={} max_rel={} top1={}/{}",
                        comparison.max_abs,
                        comparison.max_rel,
                        comparison.top1_matches,
                        comparison.positions
                    )));
                }
                Some(comparison)
            }
            None => None,
        };
        if let Some(path) = &self.config.dump_logits {
            write_logits_dump(path, &prefill_logits)?;
        }
        let summary = self.finish_summary(&prompt, &generated, comparison)?;
        Ok(RunOutcome {
            summary,
            prefill_logits,
        })
    }

    fn finish_summary(
        &mut self,
        prompt: &[u32],
        generated: &[u32],
        comparison: Option<LogitsComparison>,
    ) -> Result<RunSummary> {
        self.writer.flush()?;
        require_tokens(&self.totals)?;
        let mut decode_samples = self.decode_ns.clone();
        let mut prefill_samples = self.prefill_ns.clone();
        let prefill = Latency::from_samples(&mut prefill_samples);
        let decode = Latency::from_samples(&mut decode_samples);
        let decode_tokens_per_second = if decode.mean_ns == 0 {
            0.0
        } else {
            1.0e9 / decode.mean_ns as f64
        };
        let text = match &self.tokenizer {
            Some(tokenizer) if !generated.is_empty() => tokenizer.decode(generated)?,
            _ => String::new(),
        };
        let mut sequence_hasher = selvedge_core::Sha256Builder::new();
        for digest in &self.logits_digests {
            sequence_hasher.update(digest.as_bytes());
        }
        let sequence_digest = sequence_hasher.finish().to_hex();
        let stats: StoreStats = self.store.stats();
        let cache_stats = self.store.cache_stats();
        let exactness = match &self.exactness {
            Some(value) => serde_json::to_value(value)?,
            None => serde_json::json!({"status": "not_classified"}),
        };
        let verdict = match &self.verdict {
            Some(value) => Some(serde_json::to_value(value)?),
            None => None,
        };
        let environment = serde_json::json!({
            "threads": self.config.threads,
            "kernel": "in-repo f32 dot, eight interleaved accumulators, rows parallelised",
            "reduction_order": "fixed; independent of thread count and cache state",
            "read_audit": READ_AUDIT,
            "store": {
                "reads": stats.reads,
                "hits": stats.hits,
                "misses": stats.misses,
                "device_bytes": stats.device_bytes,
                "bytes_used": stats.bytes_used,
                "bytes_requested": stats.bytes_requested,
                "verify_ns": stats.verify_ns,
                "transfer_ns": stats.transfer_ns,
                "stall_ns": stats.stall_ns,
                "cache": cache_stats,
            },
            "kv_cache_bytes": self.cache.bytes(),
            "lm_head_verified_tied": self.model.backbone.lm_head_verified_tied,
        });
        let ok = comparison
            .as_ref()
            .map(|value| value.passed)
            .unwrap_or(true);
        Ok(RunSummary {
            schema: RUN_SCHEMA,
            ok,
            revision: std::env::var("SELVEDGE_REVISION").ok(),
            stamp: std::env::var("SELVEDGE_STAMP").ok(),
            mode: self.mode.label().to_owned(),
            selector: self.mode.selector_name(),
            bundle_id: self.plan.bundle_id.clone(),
            manifest_sha256: self.plan.manifest_sha256.to_hex(),
            model_dir: self.config.model.dir.display().to_string(),
            prompt_kind: self.config.prompt.label().to_owned(),
            prompt_tokens: prompt.to_vec(),
            generated_tokens: generated.to_vec(),
            generated_text: text,
            logits_digests: self.logits_digests.clone(),
            sequence_digest,
            layers: self.plan.layers.len(),
            groups_per_layer: self
                .plan
                .layers
                .first()
                .map(|layer| layer.groups.len())
                .unwrap_or(0),
            shards: self.plan.shard_count,
            payload_bytes: self.plan.payload_bytes,
            backbone_elements: self.model.backbone_elements(),
            total_accessible_elements: self.total_accessible,
            prefill,
            decode,
            decode_tokens_per_second,
            totals: self.totals.clone(),
            stream: StreamSummary {
                policy: self.config.stream.policy_name().to_owned(),
                cache_bytes: self.config.stream.cache_bytes,
                prefetch: self.config.stream.prefetch,
                read_order: self.config.stream.order_name().to_owned(),
                injected_read_delay_ns: self.config.stream.injected_read_delay_ns,
                verification: self.config.stream.verification.name().to_owned(),
            },
            budget_limit: self.config.budget.active_unique_limit,
            compliance: match self.config.budget.mode {
                ComplianceMode::Compliance => "compliance".to_owned(),
                ComplianceMode::ResearchNoncompliantTeacher => {
                    "research_noncompliant_teacher".to_owned()
                }
            },
            fallback_applied: self.fallback_applied,
            exactness,
            verdict,
            comparison,
            accounting_path: self
                .config
                .account
                .as_ref()
                .map(|path| path.display().to_string()),
            environment,
        })
    }

    fn compare_logits(&self, path: &Path, prefill_logits: &[Vec<f32>]) -> Result<LogitsComparison> {
        let vocab = self.model.vocab()?;
        let bytes = std::fs::read(path)?;
        let position_bytes = vocab
            .checked_mul(4)
            .ok_or_else(|| RuntimeError::model("vocabulary byte length overflows"))?;
        if bytes.len() % position_bytes != 0 {
            return Err(RuntimeError::model(format!(
                "{} holds {} bytes, which is not a whole number of {vocab}-token f32 positions",
                path.display(),
                bytes.len()
            )));
        }
        let positions = bytes.len() / position_bytes;
        if positions != prefill_logits.len() {
            return Err(RuntimeError::model(format!(
                "reference holds {positions} positions, this run has {}",
                prefill_logits.len()
            )));
        }
        let tolerance = self.config.tolerance;
        let mut max_abs = 0.0_f64;
        let mut max_rel = 0.0_f64;
        let mut top1_matches = 0_u64;
        let mut passed = true;
        for (position, observed) in prefill_logits.iter().enumerate() {
            let base = position * position_bytes;
            for (index, value) in observed.iter().enumerate() {
                let start = base + index * 4;
                let reference = f32::from_le_bytes([
                    bytes[start],
                    bytes[start + 1],
                    bytes[start + 2],
                    bytes[start + 3],
                ]);
                let difference = (f64::from(reference) - f64::from(*value)).abs();
                let relative = if reference == 0.0 {
                    difference
                } else {
                    difference / f64::from(reference).abs()
                };
                max_abs = max_abs.max(difference);
                max_rel = max_rel.max(relative);
                if !tolerance.accepts(f64::from(reference), f64::from(*value)) {
                    passed = false;
                }
            }
            let reference_slice = &bytes[base..base + position_bytes];
            if argmax_of_bytes(reference_slice) == kernels::argmax(observed)? {
                top1_matches += 1;
            }
        }
        Ok(LogitsComparison {
            reference: path.display().to_string(),
            positions: positions as u64,
            vocab: vocab as u64,
            max_abs,
            max_rel,
            top1_matches,
            abs: tolerance.abs,
            rel: tolerance.rel,
            passed,
        })
    }
}

fn argmax_of_bytes(bytes: &[u8]) -> usize {
    let mut best = (0_usize, f32::NEG_INFINITY);
    for (index, chunk) in bytes.as_chunks::<4>().0.iter().enumerate() {
        let value = f32::from_le_bytes(*chunk);
        if value > best.1 {
            best = (index, value);
        }
    }
    best.0
}

fn write_logits_dump(path: &Path, positions: &[Vec<f32>]) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let mut bytes = Vec::new();
    for position in positions {
        for value in position {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

/// The MLP executor the model drives, holding disjoint borrows of the runner.
struct LayerMlp<'a> {
    plan: &'a BundlePlan,
    store: &'a ShardStore,
    mode: &'a mut ModeState,
    reader: &'a mut GroupReader,
    ledger: &'a mut ParameterLedger,
    token_uses: &'a mut Vec<ParamUse>,
    gate_buf: &'a mut Vec<f32>,
    up_buf: &'a mut Vec<f32>,
    offsets: &'a BTreeMap<selvedge_core::ShardId, u64>,
    threads: usize,
    fallback: bool,
    prefetch: bool,
    read_order: ReadOrder,
    state: &'a mut ExecutorState,
}

impl LayerMlp<'_> {
    fn try_admit(
        &self,
        uses: &[ParamUse],
        selector_flops: u64,
        execution_flops: u64,
    ) -> core::result::Result<(), BudgetError> {
        let path = SelectionPath::new(
            Vec::new(),
            uses.to_vec(),
            FlopEstimate {
                matmul_flops: 0,
                other_flops: 0,
                controller_flops: selector_flops,
            },
            FlopEstimate {
                matmul_flops: execution_flops,
                other_flops: 0,
                controller_flops: 0,
            },
        )
        .map_err(|error| BudgetError::InvalidPlan {
            reason: error.to_string(),
        })?;
        self.ledger.admit_selection_path(path).map(|_admitted| ())
    }
}

impl MlpExecutor for LayerMlp<'_> {
    fn mlp(
        &mut self,
        layer: usize,
        x: &[f32],
        out: &mut [f32],
        trace: &mut TokenTrace,
    ) -> Result<()> {
        let hidden = self.plan.hidden;
        let groups: &[GroupPlan] = self.plan.groups_for(layer)?;
        let choices = self.mode.decide(layer, groups, x, hidden)?;
        let selector_flops = self.mode.selector_flops(groups.len(), hidden);
        self.state.controller_flops = self.state.controller_flops.saturating_add(selector_flops);

        let mut selected: Vec<&GroupPlan> = Vec::with_capacity(choices.len());
        for choice in &choices {
            let group = groups.get(choice.index).ok_or_else(|| {
                RuntimeError::model(format!(
                    "selector chose group {} of layer {layer}, which has {} groups",
                    choice.index,
                    groups.len()
                ))
            })?;
            selected.push(group);
        }

        let mut planned_flops = self.state.admitted_flops;
        // Reserve room for one group in every layer that has not run yet, so an
        // early layer cannot spend the budget the rest of the network needs. The
        // reserved group is the layer's smallest, chosen deterministically.
        let mut reserve_uses: Vec<ParamUse> = Vec::new();
        for later in layer + 1..self.plan.layers.len() {
            let groups = self.plan.groups_for(later)?;
            if let Some(smallest) = groups
                .iter()
                .min_by_key(|group| (group.elements(), group.index))
            {
                for fragment in [&smallest.gate, &smallest.up, &smallest.down] {
                    reserve_uses.push(fragment_use(fragment, StorageTier::HostRam)?);
                }
            }
        }
        loop {
            let mut path_uses: Vec<ParamUse> = self.token_uses.clone();
            for group in &selected {
                for fragment in [&group.gate, &group.up, &group.down] {
                    path_uses.push(fragment_use(fragment, StorageTier::HostRam)?);
                }
            }
            path_uses.extend(reserve_uses.iter().cloned());
            let group_planned: u64 = selected.iter().map(|group| group.flops()).sum();
            let candidate_flops = planned_flops.saturating_add(group_planned);
            match self.try_admit(&path_uses, self.state.controller_flops, candidate_flops) {
                Ok(()) => {
                    planned_flops = candidate_flops;
                    break;
                }
                Err(error) if error.is_exceeded() && self.fallback && !self.mode.is_exact() => {
                    if selected.len() <= 1 {
                        return Err(RuntimeError::budget(format!(
                            "the within-budget fallback cannot drop below one group per layer, and one group plus the reserve for the remaining layers does not fit: {error}"
                        )));
                    }
                    let drop = least_scoring(&choices, &selected);
                    let position = selected
                        .iter()
                        .position(|group| group.index == drop)
                        .ok_or_else(|| {
                            RuntimeError::model("fallback could not locate the group to drop")
                        })?;
                    selected.remove(position);
                    self.state.fallback_applied = true;
                    self.mode.note_trimmed(1);
                }
                Err(error) => {
                    return Err(RuntimeError::budget(format!(
                        "budget gate refused layer {layer} before execution: {error}"
                    )));
                }
            }
        }
        self.state.admitted_flops = planned_flops;

        // Execute in ascending group order, whatever the scores were.
        selected.sort_by_key(|group| group.index);

        if !self.mode.is_exact() {
            // The approximate scope is the layer's selected set: the store still
            // refuses any read outside it, so the storage claim is per layer.
            let mut ids = Vec::with_capacity(selected.len() * 3);
            for group in &selected {
                ids.extend(group.shards());
            }
            self.reader.replace_scope(ExecutionScope::exact(ids)?);
        }

        let mut prefetched_ids: BTreeSet<selvedge_core::ShardId> = BTreeSet::new();
        if self.prefetch {
            let mut fragments: Vec<ShardFragment> = Vec::with_capacity(selected.len() * 3);
            for group in &selected {
                fragments.push(group.gate);
                fragments.push(group.up);
                fragments.push(group.down);
            }
            match self.read_order {
                ReadOrder::Plan => {}
                ReadOrder::Reverse => fragments.reverse(),
                ReadOrder::Physical => fragments.sort_by_key(|fragment| {
                    (
                        self.offsets.get(&fragment.id).copied().unwrap_or(0),
                        fragment.id,
                    )
                }),
            }
            // A prefetched shard is accounted once, here, from the tier its
            // bytes actually came from. The execution read that follows is a
            // cache hit and adds no second use.
            let prefetched = self.reader.prefetch(self.store, &fragments)?;
            for (fragment, device) in prefetched {
                let tier = if device {
                    StorageTier::Nvme
                } else {
                    StorageTier::HostRam
                };
                let use_value = fragment_use(&fragment, tier)?;
                self.token_uses.push(use_value.clone());
                self.ledger.record_use(use_value)?;
                prefetched_ids.insert(fragment.id);
            }
        }

        out.fill(0.0);
        for group in &selected {
            let width = usize::try_from(group.width())
                .map_err(|_| RuntimeError::memory("group width does not fit usize"))?;
            let hidden_usize = usize::try_from(hidden)
                .map_err(|_| RuntimeError::memory("hidden width does not fit usize"))?;
            if self.gate_buf.len() != width {
                self.gate_buf.clear();
                self.gate_buf.resize(width, 0.0);
            }
            if self.up_buf.len() != width {
                self.up_buf.clear();
                self.up_buf.resize(width, 0.0);
            }
            self.reader.read_group(self.store, group, hidden)?;
            let sources = self.reader.last_group_sources();
            for (fragment, device) in [
                (group.gate, sources[0]),
                (group.up, sources[1]),
                (group.down, sources[2]),
            ] {
                if prefetched_ids.contains(&fragment.id) {
                    continue;
                }
                let tier = if device {
                    StorageTier::Nvme
                } else {
                    StorageTier::HostRam
                };
                let use_value = fragment_use(&fragment, tier)?;
                self.token_uses.push(use_value.clone());
                self.ledger.record_use(use_value)?;
            }
            let tensors: &GroupTensors = self.reader.tensors();
            kernels::matvec_into(
                &tensors.gate,
                width,
                hidden_usize,
                x,
                self.gate_buf,
                self.threads,
            )?;
            kernels::matvec_into(
                &tensors.up,
                width,
                hidden_usize,
                x,
                self.up_buf,
                self.threads,
            )?;
            kernels::swiglu_act(self.gate_buf, self.up_buf)?;
            kernels::matvec_accumulate(&tensors.down, hidden_usize, width, self.gate_buf, out)?;
            trace.add_matvec(width as u64, hidden)?;
            trace.add_matvec(width as u64, hidden)?;
            trace.add_matvec(hidden, width as u64)?;
            trace.add_other((3 * width) as u64)?;
        }
        Ok(())
    }
}

fn fragment_use(fragment: &ShardFragment, tier: StorageTier) -> Result<ParamUse> {
    let tensor = ParamTensorId::new(format!("shard:{}", fragment.id.short()))?;
    let bytes = fragment
        .elements
        .checked_mul(4)
        .ok_or_else(|| RuntimeError::model("fragment byte count overflows"))?;
    ParamUse::once(
        ParamIdentity::new(tensor, 0..fragment.elements)?,
        bytes,
        tier,
    )
    .map_err(RuntimeError::from)
}

fn least_scoring(choices: &[GroupChoice], selected: &[&GroupPlan]) -> usize {
    let mut worst = selected[0].index;
    let mut worst_score = f32::INFINITY;
    for group in selected {
        let score = choices
            .iter()
            .find(|choice| choice.index == group.index)
            .map(|choice| choice.score)
            .unwrap_or(0.0);
        if score < worst_score || (score == worst_score && group.index > worst) {
            worst = group.index;
            worst_score = score;
        }
    }
    worst
}

fn load_tokenizer(config: &RunConfig) -> Result<Option<GreedyTokenizer>> {
    let needed = !matches!(config.prompt, PromptSource::Ids(_));
    let path = config.model.tokenizer.clone().or_else(|| {
        let candidate = config.model.dir.join("tokenizer.json");
        candidate.exists().then_some(candidate)
    });
    match (path, needed) {
        (Some(path), _) => Ok(Some(
            GreedyTokenizer::from_file(&path)?.with_stop_tokens(config.stop_tokens.iter().copied()),
        )),
        (None, true) => Err(RuntimeError::config(
            "tokenizer",
            "a text or chat prompt needs tokenizer.json",
        )),
        (None, false) => Ok(None),
    }
}

fn resolve_prompt(config: &RunConfig, tokenizer: Option<&GreedyTokenizer>) -> Result<Vec<u32>> {
    match &config.prompt {
        PromptSource::Ids(ids) => Ok(ids.clone()),
        PromptSource::Text(text) => tokenizer
            .ok_or_else(|| RuntimeError::config("tokenizer", "a text prompt needs tokenizer.json"))?
            .encode(text),
        PromptSource::Chat { user } => tokenizer
            .ok_or_else(|| RuntimeError::config("tokenizer", "a chat prompt needs tokenizer.json"))?
            .encode_chat(user),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn logits_digest_notices_one_bit() {
        let a = [1.0_f32, 2.0, 3.0];
        let mut b = a;
        b[1] = f32::from_bits(b[1].to_bits() ^ 1);
        assert_ne!(logits_digest(&a), logits_digest(&b));
        assert_eq!(logits_digest(&a), logits_digest(&a));
    }

    #[test]
    fn stop_tokens_are_read_from_the_generation_config() {
        let dir = std::env::temp_dir().join(format!("selvedge-stop-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("generation_config.json"),
            "{\"eos_token_id\": [151645, 151643]}",
        )
        .unwrap();
        assert_eq!(default_stop_tokens(&dir), vec![151643, 151645]);
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(default_stop_tokens(&dir).is_empty());
    }
}
