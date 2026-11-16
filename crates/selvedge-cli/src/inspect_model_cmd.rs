//! `selvedge inspect-model`: exact parameter and FLOP accounting for a real
//! Qwen3 or Qwen3.5 checkpoint.
//!
//! Qwen3 uses accounting schema v1 (flat config, one tower). Qwen3.5 uses
//! schema v2: nested `text_config`, a separate vision tower, a multi-token
//! prediction head, a declared execution path and a ceiling certificate. The
//! dispatch is on `model_type` read from the checkpoint's `config.json`; there
//! is no fallback, so a config the auditor does not understand fails closed.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Args;
use selvedge_convert::{
    AccountingOptions, ActivePath, DEFAULT_ACTIVE_CEILING, DEFAULT_GROUP_COUNT,
    ModelAccountingReport, Qwen35AccountingReport, Qwen35Options, TieAssertion, account_model,
    account_model_qwen35, detect_model_type,
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
    /// Group count for per-group arithmetic (Qwen3 only).
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
    /// Declare the execution path: `text-only` or `text-and-mtp` (Qwen3.5 only).
    #[arg(long)]
    pub(crate) active_path: Option<String>,
    /// Certify the declared active set against this ceiling (Qwen3.5 only).
    #[arg(long)]
    pub(crate) certify_ceiling: Option<u64>,
    /// Declare the embedding and output projection tied (Qwen3.5 only).
    #[arg(long, conflicts_with = "assume_untied")]
    pub(crate) assume_tied: bool,
    /// Declare the embedding and output projection untied (Qwen3.5 only).
    #[arg(long)]
    pub(crate) assume_untied: bool,
    /// Comma-separated `name=elements` totals the auditor must reproduce exactly
    /// (`text`, `text-checkpoint`, `vision`, `mtp`, `checkpoint`,
    /// `checkpoint-unique`, `active`; Qwen3.5 only).
    #[arg(long, value_delimiter = ',')]
    pub(crate) expect_totals: Option<Vec<String>>,
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

fn parse_expectations(raw: &Option<Vec<String>>) -> selvedge_convert::Result<Vec<(String, u64)>> {
    let mut out = Vec::new();
    if let Some(items) = raw {
        for item in items {
            let (name, value) = item.split_once('=').ok_or_else(|| {
                selvedge_convert::ConvertError::plan(format!(
                    "--expect-totals entry {item:?} is not name=elements"
                ))
            })?;
            let elements: u64 = value.trim().parse().map_err(|_| {
                selvedge_convert::ConvertError::plan(format!(
                    "--expect-totals value {value:?} is not an integer"
                ))
            })?;
            out.push((name.trim().to_owned(), elements));
        }
    }
    Ok(out)
}

fn run_inner(args: &InspectModelArgs) -> CliResult {
    let model_type = detect_model_type(&args.path)?;
    let v2_flag = args.active_path.is_some()
        || args.certify_ceiling.is_some()
        || args.assume_tied
        || args.assume_untied
        || args.expect_totals.is_some();
    match model_type.as_str() {
        "qwen3" => {
            if v2_flag {
                return Err(selvedge_convert::ConvertError::plan(
                    "--active-path, --certify-ceiling, --assume-tied, --assume-untied and \
                     --expect-totals are Qwen3.5 (schema v2) options; this checkpoint is qwen3",
                )
                .into());
            }
            run_v1(args)
        }
        "qwen3_5" | "qwen3_5_text" => run_v2(args),
        other => Err(selvedge_convert::ConvertError::inconsistent(format!(
            "inspect-model accounts model_type \"qwen3\" and \"qwen3_5\"; config says {other:?}"
        ))
        .into()),
    }
}

fn run_v1(args: &InspectModelArgs) -> CliResult {
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

fn run_v2(args: &InspectModelArgs) -> CliResult {
    let active_path = match &args.active_path {
        Some(value) => ActivePath::parse(value)?,
        None => ActivePath::TextOnly,
    };
    let tie_assertion = if args.assume_tied {
        TieAssertion::Tied
    } else if args.assume_untied {
        TieAssertion::Untied
    } else {
        TieAssertion::FromConfig
    };
    if args.contexts.is_some() {
        return Err(selvedge_convert::ConvertError::plan(
            "--contexts asks for a FLOP estimate; no FLOP formula is defined for the Qwen3.5 \
             hybrid path, so the request is refused rather than answered with an invented number",
        )
        .into());
    }
    let options = Qwen35Options {
        ceiling: args.ceiling,
        active_path,
        tie_assertion,
        certify_ceiling: args.certify_ceiling,
        expect_totals: parse_expectations(&args.expect_totals)?,
        compute_digest: !args.no_digest,
        stamp: args.stamp.clone(),
        revision: args.revision.clone(),
    };
    let report = account_model_qwen35(&args.path, &options)?;
    if let Some(path) = &args.report_out {
        crate::convert_cmd::write_report(path, &report)?;
    }
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_human_qwen35(&report);
    }
    Ok(if report.ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

fn print_human_qwen35(report: &Qwen35AccountingReport) {
    let model = &report.model;
    let text = &model.text;
    println!(
        "model={} type={} schema={} layers={} hidden={} intermediate={} vocab={}",
        model.path,
        model.model_type,
        report.schema,
        text.layers,
        text.hidden,
        text.intermediate,
        text.vocab,
    );
    println!(
        "text: {} linear_attention + {} full_attention at {:?}; delta heads {}x{} conv {}; attention {}q/{}kv x {} (rotary {} mRoPE {:?}); eps={}",
        text.linear_layers,
        text.attention_layers,
        text.attention_layer_indices,
        text.linear_key_heads,
        text.linear_key_head_dim,
        text.conv_kernel,
        text.attention_heads,
        text.attention_kv_heads,
        text.attention_head_dim,
        text.rotary_dim,
        text.mrope_section,
        text.rms_norm_eps,
    );
    if let Some(vision) = &model.vision {
        println!(
            "vision: depth={} hidden={} intermediate={} heads={} out={} patch={} merge={} positions={}",
            vision.depth,
            vision.hidden,
            vision.intermediate,
            vision.heads,
            vision.out_hidden,
            vision.patch_size,
            vision.spatial_merge_size,
            vision.position_embeddings,
        );
    }
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
    for (name, tower) in &report.towers {
        println!(
            "tower={} tensors={} checkpoint_elements={} unique_elements={} ffn_elements={} always_active={} executed={}",
            name,
            tower.tensors,
            tower.checkpoint_elements,
            tower.unique_elements,
            tower.ffn_elements,
            tower.always_active_elements,
            tower.executed_by_declared_path,
        );
    }
    let checkpoint = &report.checkpoint;
    println!(
        "checkpoint tensors={} elements={} bytes={} unique={} index_total_bytes={:?} index_matches={:?}",
        checkpoint.tensors,
        checkpoint.tensor_elements,
        checkpoint.tensor_bytes,
        checkpoint.unique_elements,
        checkpoint.index_total_bytes,
        checkpoint.index_total_matches,
    );
    for entry in &checkpoint.by_dtype {
        println!(
            "dtype={} tensors={} elements={} bytes={}",
            entry.dtype, entry.tensors, entry.elements, entry.bytes
        );
    }
    let path = &report.active_path;
    println!(
        "active_path={} executed={:?} excluded={:?} unique_active={} verdict={}",
        path.declared,
        path.executed_towers,
        path.excluded_towers,
        path.unique_active_elements,
        path.verdict,
    );
    for excluded in &path.excluded_elements {
        println!(
            "excluded tower={} elements={} reason={}",
            excluded.tower, excluded.elements, excluded.reason
        );
    }
    for shared in &path.shared_reads {
        println!(
            "shared_read tensor={} towers={:?} reason={}",
            shared.tensor, shared.towers, shared.reason
        );
    }
    let cert = &report.certification;
    for term in &cert.arithmetic {
        println!("  {} {} {}", term.sign, term.elements, term.label);
    }
    println!(
        "ceiling={} requested={} active={} headroom={} verdict={}",
        cert.ceiling,
        cert.requested,
        cert.active_unique_elements,
        cert.headroom_elements,
        cert.verdict,
    );
    for check in &report.expectations {
        println!(
            "expect name={} expected={} observed={} matches={}",
            check.name, check.expected, check.observed, check.matches
        );
    }
    println!("{}", cert.explanation);
    if report.refusals.is_empty() {
        println!("verdict=ok");
    } else {
        println!("verdict=refused ({}):", report.refusals.len());
        for refusal in &report.refusals {
            println!("  {refusal}");
        }
    }
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
