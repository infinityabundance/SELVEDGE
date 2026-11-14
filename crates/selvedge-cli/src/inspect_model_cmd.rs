//! `selvedge inspect-model`: exact parameter and FLOP accounting for a real
//! Qwen3 checkpoint.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Args;
use selvedge_convert::{
    AccountingOptions, DEFAULT_ACTIVE_CEILING, DEFAULT_GROUP_COUNT, ModelAccountingReport,
    account_model,
};
use serde::Serialize;

use crate::CliResult;

#[derive(Debug, Args)]
pub(crate) struct InspectModelArgs {
    /// Checkpoint safetensors file or directory. `config.json` must be beside
    /// it (or inside the directory).
    pub(crate) path: PathBuf,
    /// Emit machine-readable JSON.
    #[arg(long)]
    pub(crate) json: bool,
    /// Unique-active-parameter ceiling.
    #[arg(long, default_value_t = DEFAULT_ACTIVE_CEILING)]
    pub(crate) ceiling: u64,
    /// Group count for per-group arithmetic.
    #[arg(long, default_value_t = DEFAULT_GROUP_COUNT)]
    pub(crate) groups: u32,
    /// Comma-separated reference contexts for the attention-core estimate.
    #[arg(long, value_delimiter = ',')]
    pub(crate) contexts: Option<Vec<u64>>,
    /// Skip the streaming checkpoint digest.
    #[arg(long)]
    pub(crate) no_digest: bool,
    /// Write the report JSON to this path (receipt-shaped).
    #[arg(long)]
    pub(crate) report_out: Option<PathBuf>,
    /// Receipt stamp recorded verbatim in the report.
    #[arg(long)]
    pub(crate) stamp: Option<String>,
    /// Git revision recorded verbatim in the report.
    #[arg(long)]
    pub(crate) revision: Option<String>,
}

#[derive(Debug, Serialize)]
struct ErrorReport<'a> {
    schema: &'a str,
    ok: bool,
    error: String,
}

/// Run `selvedge inspect-model`.
///
/// # Errors
///
/// Propagates checkpoint, config and serialization failures.
pub(crate) fn run(args: &InspectModelArgs) -> CliResult {
    match run_inner(args) {
        Ok(code) => Ok(code),
        Err(error) => {
            if args.json {
                let report = ErrorReport {
                    schema: "selvedge/inspect-model-error/v1",
                    ok: false,
                    error: error.to_string(),
                };
                println!("{}", serde_json::to_string_pretty(&report)?);
                Ok(ExitCode::FAILURE)
            } else {
                Err(error)
            }
        }
    }
}

fn run_inner(args: &InspectModelArgs) -> CliResult {
    let mut options = AccountingOptions {
        ceiling: args.ceiling,
        group_count: args.groups,
        compute_digest: !args.no_digest,
        stamp: args.stamp.clone(),
        revision: args.revision.clone(),
        ..AccountingOptions::default()
    };
    if let Some(contexts) = &args.contexts {
        options.reference_contexts = contexts.clone();
    }
    let report = account_model(&args.path, &options)?;
    if let Some(path) = &args.report_out {
        crate::convert_cmd::write_report(path, &report)?;
    }
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_human(&report);
    }
    Ok(if report.ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

fn print_human(report: &ModelAccountingReport) {
    let model = &report.model;
    println!(
        "model={} type={} layers={} hidden={} intermediate={} heads={} kv_heads={} head_dim={} vocab={}",
        model.path,
        model.model_type,
        model.layers,
        model.hidden,
        model.intermediate,
        model.heads,
        model.kv_heads,
        model.head_dim,
        model.vocab,
    );
    println!(
        "config_sha256={} checkpoint_digest={} files={}",
        model.config_sha256,
        model.checkpoint_digest.as_deref().unwrap_or("<skipped>"),
        model.checkpoint_files,
    );
    println!(
        "tie config={} lm_head_present={} identical={} treated_as_shared={} compared_elements={}",
        report.tie.config_tie_word_embeddings,
        report.tie.lm_head_present,
        report.tie.lm_head_identical_to_embedding,
        report.tie.treated_as_shared,
        report.tie.compared_elements,
    );
    println!(
        "totals embedding={} attention={} norm={} ffn={} lm_head={}",
        report.totals.embedding,
        report.totals.attention,
        report.totals.norm,
        report.totals.ffn,
        report.totals.lm_head,
    );
    let budget = &report.budget;
    println!(
        "unique={} checkpoint_tensor_elements={} backbone={} removable_ffn={} ceiling={} headroom={}",
        budget.unique_learned_elements,
        budget.checkpoint_tensor_elements,
        budget.always_active_backbone_elements,
        budget.removable_ffn_elements,
        budget.ceiling,
        budget.headroom_elements,
    );
    println!(
        "verdict={} binding={} dense_compliant={} backbone_fits={} max_ffn={} fraction={:.6} channels/layer={} whole_groups/layer={} full_ffn_layers={:.3}",
        budget.verdict,
        budget.ceiling_binding,
        budget.dense_execution_compliant,
        budget.backbone_fits,
        budget.max_compliant_ffn_elements,
        budget.max_compliant_ffn_fraction,
        budget.max_compliant_channels_per_layer,
        budget.max_compliant_whole_groups_per_layer,
        budget.max_compliant_full_ffn_layers,
    );
    println!("{}", budget.explanation);
    let flops = &report.flops;
    println!(
        "flops/token backbone={} ffn_all={} dense={} (estimates)",
        flops.per_token.always_on_backbone,
        flops.per_token.ffn_total_all_layers_estimate,
        flops.per_token.dense_total_at_first_context_estimate,
    );
    println!(
        "flops/group width={} group_count={} ffn_matmul={} always_on_plus_one_group={}",
        flops.groups.group_width,
        flops.groups.group_count,
        flops.groups.matmul_flops_per_full_group,
        flops.groups.always_on_plus_one_group,
    );
}
