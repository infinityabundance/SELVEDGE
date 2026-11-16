//! `selvedge run`: one-device Qwen3 inference with bundle-backed MLP groups.
//!
//! Exit codes are meaningful:
//!
//! - `0`: the run completed and every gate passed;
//! - `1`: the run completed but a gate failed (a logits comparison outside
//!   tolerance), or the run was refused (budget, exactness, unsupported
//!   architecture) — never silently;
//! - `2`: the command line or a file could not be used at all.
//!
//! Nothing is approximated silently: an over-budget exact path, a bundle that is
//! not a complete partition, or a mismatch between the checkpoint and the bundle
//! is an error with a typed message.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, ValueEnum};
use selvedge_core::{Budget, ComplianceMode};
use selvedge_runtime::{
    ExecutionMode, ModelDescriptor, PromptSource, RunConfig, Runner, SelectorSpec, StreamConfig,
    Tolerance,
};
use selvedge_store::{PolicyKind, ReadOrder, VerificationMode};
use serde::Serialize;

use crate::CliResult;

/// Execution mode selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(crate) enum ModeArg {
    /// Every group of every layer, proven against the manifest's declaration.
    Exact,
    /// A selector-chosen subset, labelled approximate.
    Approximate,
}

/// Prefill strategy selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(crate) enum PrefillArg {
    /// Token at a time through the same path as decode. The reference oracle.
    Scalar,
    /// Prompt tokens as a batch through the same weights (exact mode only).
    Batched,
}

impl PrefillArg {
    const fn mode(self) -> selvedge_runtime::PrefillMode {
        match self {
            Self::Scalar => selvedge_runtime::PrefillMode::Scalar,
            Self::Batched => selvedge_runtime::PrefillMode::Batched,
        }
    }
}

/// Cache policy selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(crate) enum PolicyArg {
    /// No cache: every read goes to the bundle.
    None,
    /// Least recently used.
    Lru,
    /// Least frequently used.
    Lfu,
    /// SELVEDGE's least-stale policy.
    LeastStale,
    /// Pinned residency.
    Pinned,
}

impl PolicyArg {
    const fn policy(self) -> PolicyKind {
        match self {
            Self::None => PolicyKind::None,
            Self::Lru => PolicyKind::Lru,
            Self::Lfu => PolicyKind::Lfu,
            Self::LeastStale => PolicyKind::LeastStale,
            Self::Pinned => PolicyKind::Pinned,
        }
    }
}

/// Fetch-order selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(crate) enum OrderArg {
    /// Manifest order.
    Plan,
    /// Ascending physical offset.
    Physical,
    /// Reverse manifest order.
    Reverse,
}

impl OrderArg {
    const fn order(self) -> ReadOrder {
        match self {
            Self::Plan => ReadOrder::Plan,
            Self::Physical => ReadOrder::Physical,
            Self::Reverse => ReadOrder::Reverse,
        }
    }
}

#[derive(Debug, Args)]
pub(crate) struct RunArgs {
    /// Model directory holding config.json (and tokenizer.json).
    #[arg(long)]
    pub(crate) model: PathBuf,
    /// Bundle to execute.
    #[arg(long)]
    pub(crate) bundle: PathBuf,
    /// Execution mode.
    #[arg(long, value_enum, default_value = "exact")]
    pub(crate) mode: ModeArg,
    /// Groups kept per layer in approximate mode.
    #[arg(long, default_value_t = 1)]
    pub(crate) keep_groups: usize,
    /// Selector seed in approximate mode.
    #[arg(long, default_value_t = 0x5E1D_E6E)]
    pub(crate) selector_seed: u64,
    /// Raw text prompt, encoded with no chat template.
    #[arg(long, conflicts_with_all = ["prompt_ids", "chat"])]
    pub(crate) prompt: Option<String>,
    /// Explicit token ids, bypassing the tokenizer.
    #[arg(long = "prompt-ids", value_delimiter = ',', conflicts_with = "chat")]
    pub(crate) prompt_ids: Vec<u32>,
    /// User turn wrapped in the documented Qwen3 chat template.
    #[arg(long)]
    pub(crate) chat: Option<String>,
    /// Tokens to generate.
    #[arg(long, default_value_t = 16)]
    pub(crate) max_new_tokens: usize,
    /// Stop tokens; defaults to the model's generation_config.json.
    #[arg(long = "stop-token", value_delimiter = ',')]
    pub(crate) stop_token: Vec<u32>,
    /// Explicit tokenizer path.
    #[arg(long)]
    pub(crate) tokenizer: Option<PathBuf>,
    /// Cache capacity in bytes.
    #[arg(long, default_value_t = 0)]
    pub(crate) cache_bytes: u64,
    /// Byte budget for the **decoded** f32 weight residency, which is separate
    /// from `--cache-bytes` (verified bundle bytes). 0 disables it and
    /// reproduces the pre-P14.1 path, where every read was decoded again.
    #[arg(long, default_value_t = 1_500_000_000)]
    pub(crate) decode_cache_bytes: u64,
    /// Cache policy.
    #[arg(long, value_enum, default_value = "none")]
    pub(crate) policy: PolicyArg,
    /// Prefetch the selected shards before reading them for execution
    /// (synchronous: it cannot overlap compute).
    #[arg(long)]
    pub(crate) prefetch: bool,
    /// Prefetch on a separate fetch timeline, overlapping the kernels.
    #[arg(long)]
    pub(crate) async_prefetch: bool,
    /// Outstanding shard jobs the asynchronous fetcher may queue.
    #[arg(long, default_value_t = 8)]
    pub(crate) prefetch_depth: usize,
    /// How the prompt is executed.
    #[arg(long, value_enum, default_value = "scalar")]
    pub(crate) prefill: PrefillArg,
    /// Largest prompt batch the batched prefill forms.
    #[arg(long, default_value_t = 64)]
    pub(crate) prefill_batch: usize,
    /// Fetch order for prefetched shards.
    #[arg(long, value_enum, default_value = "plan")]
    pub(crate) read_order: OrderArg,
    /// Extra delay injected after every backing-tier read.
    #[arg(long, default_value_t = 0)]
    pub(crate) inject_delay_ns: u64,
    /// Worker threads for the dense kernels. Measured: one thread is faster at
    /// batch 1 with these kernels (see docs/RUNTIME.md), so one is the default.
    #[arg(long, default_value_t = 1)]
    pub(crate) threads: usize,
    /// Active-unique-parameter ceiling.
    #[arg(long, default_value_t = 1_000_000_000)]
    pub(crate) budget: u64,
    /// Label an over-budget run as a non-compliant teacher measurement.
    #[arg(long)]
    pub(crate) teacher: bool,
    /// Trim an over-budget approximate path to fit instead of refusing.
    #[arg(long)]
    pub(crate) fallback: bool,
    /// Write per-token accounting as JSON Lines here.
    #[arg(long)]
    pub(crate) accounting: Option<PathBuf>,
    /// Compare prefill logits against a raw little-endian f32 dump.
    #[arg(long)]
    pub(crate) compare_logits: Option<PathBuf>,
    /// Write this run's prefill logits as a raw little-endian f32 dump.
    #[arg(long)]
    pub(crate) dump_logits: Option<PathBuf>,
    /// Absolute tolerance for --compare-logits.
    #[arg(long, default_value_t = 1.0e-3)]
    pub(crate) tolerance_abs: f64,
    /// Relative tolerance for --compare-logits.
    #[arg(long, default_value_t = 1.0e-3)]
    pub(crate) tolerance_rel: f64,
    /// Emit the structured summary as JSON.
    #[arg(long)]
    pub(crate) json: bool,
}

#[derive(Debug, Serialize)]
struct FailureReport<'a> {
    schema: &'a str,
    ok: bool,
    error: String,
}

pub(crate) fn run(args: &RunArgs) -> CliResult {
    let prompt = match (&args.prompt, &args.chat, args.prompt_ids.is_empty()) {
        (Some(text), None, true) => PromptSource::Text(text.clone()),
        (None, Some(user), true) => PromptSource::Chat { user: user.clone() },
        (None, None, false) => PromptSource::Ids(args.prompt_ids.clone()),
        _ => {
            return Err("exactly one of --prompt, --chat or --prompt-ids must be given".into());
        }
    };

    let model = ModelDescriptor {
        dir: args.model.clone(),
        checkpoint: None,
        tokenizer: args.tokenizer.clone(),
    };
    let mode = match args.mode {
        ModeArg::Exact => ExecutionMode::Exact,
        ModeArg::Approximate => ExecutionMode::Approximate(SelectorSpec {
            keep_per_layer: args.keep_groups,
            seed: args.selector_seed,
        }),
    };
    let mut config = RunConfig::new(model, args.bundle.clone(), mode, prompt)?;
    config.stream = StreamConfig {
        policy: args.policy.policy(),
        cache_bytes: args.cache_bytes,
        prefetch: args.prefetch,
        async_prefetch: args.async_prefetch,
        prefetch_depth: args.prefetch_depth,
        read_order: args.read_order.order(),
        injected_read_delay_ns: args.inject_delay_ns,
        decode_cache_bytes: args.decode_cache_bytes,
        verification: VerificationMode::Always,
        ..StreamConfig::default()
    };
    config.prefill = args.prefill.mode();
    config.prefill_batch = args.prefill_batch;
    config.budget = Budget::new(
        args.budget,
        if args.teacher {
            ComplianceMode::ResearchNoncompliantTeacher
        } else {
            ComplianceMode::Compliance
        },
    )?;
    config.fallback = args.fallback;
    config.threads = args.threads;
    config.account = args.accounting.clone();
    config.max_new_tokens = args.max_new_tokens;
    config.compare_logits = args.compare_logits.clone();
    config.dump_logits = args.dump_logits.clone();
    config.tolerance = Tolerance {
        abs: args.tolerance_abs,
        rel: args.tolerance_rel,
    };
    let stop = if args.stop_token.is_empty() {
        selvedge_runtime::run::default_stop_tokens(&args.model)
    } else {
        args.stop_token.clone()
    };
    config.stop_tokens = stop;
    config.validate()?;

    // A text prompt without an explicit tokenizer path must still find one; the
    // runtime reports that clearly, but doing it here gives a better message.
    let mut runner = match Runner::open(config) {
        Ok(runner) => runner,
        Err(error) => {
            if args.json {
                let report = FailureReport {
                    schema: "selvedge/runtime-run/v1",
                    ok: false,
                    error: error.to_string(),
                };
                println!("{}", serde_json::to_string_pretty(&report)?);
            }
            eprintln!("selvedge run: {error}");
            return Ok(ExitCode::from(1));
        }
    };

    let outcome = match runner.run() {
        Ok(outcome) => outcome,
        Err(error) => {
            if args.json {
                let report = FailureReport {
                    schema: "selvedge/runtime-run/v1",
                    ok: false,
                    error: error.to_string(),
                };
                println!("{}", serde_json::to_string_pretty(&report)?);
            }
            eprintln!("selvedge run: {error}");
            return Ok(ExitCode::from(1));
        }
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&outcome.summary)?);
    } else {
        print_human(&outcome.summary);
    }
    if outcome.ok() {
        Ok(ExitCode::SUCCESS)
    } else {
        Ok(ExitCode::from(1))
    }
}

fn print_human(summary: &selvedge_runtime::account::RunSummary) {
    println!("mode            {}", summary.mode);
    println!("selector        {}", summary.selector);
    println!("bundle          {}", summary.bundle_id);
    println!("prompt tokens   {}", summary.prompt_tokens.len());
    println!("generated       {:?}", summary.generated_tokens);
    if !summary.generated_text.is_empty() {
        println!(
            "text            {}",
            summary.generated_text.replace('\n', "\\n")
        );
    }
    println!(
        "prefill         p50 {} us  p95 {} us",
        summary.prefill.p50_ns / 1_000,
        summary.prefill.p95_ns / 1_000
    );
    println!(
        "decode          p50 {} us  p95 {} us  ({} tok/s)",
        summary.decode.p50_ns / 1_000,
        summary.decode.p95_ns / 1_000,
        summary.decode_tokens_per_second
    );
    println!(
        "accounting      active_unique_max {} of {} limit",
        summary.totals.active_unique_max, summary.budget_limit
    );
    println!(
        "device bytes    {} total over {} tokens",
        summary.totals.bytes_device, summary.totals.tokens
    );
    println!("exactness       {}", summary.exactness["status"]);
    if let Some(comparison) = &summary.comparison {
        println!(
            "comparison      passed={} max_abs={} max_rel={} top1={}/{}",
            comparison.passed,
            comparison.max_abs,
            comparison.max_rel,
            comparison.top1_matches,
            comparison.positions
        );
    }
    println!("sequence digest {}", summary.sequence_digest);
}
