//! `selvedge atlas fit|select|compare` — the Atlas selector tools.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, Subcommand};
use selvedge_atlas::{
    AtlasWeights, CompareInput, FitData, SelectorKind, compare as compare_models,
    fit as fit_models, read_fixtures, select_named,
};
use serde::Serialize;

/// Arguments for the Atlas commands.
#[derive(Debug, Args)]
pub(crate) struct AtlasArgs {
    #[command(subcommand)]
    command: AtlasCommand,
}

#[derive(Debug, Subcommand)]
enum AtlasCommand {
    /// Fit the unary and interaction models from the exported normal equations.
    Fit(FitArgs),
    /// Run one selector over exported pre-decision records.
    Select(SelectArgs),
    /// Verify matched realized budgets and compute paired comparisons.
    Compare(CompareArgs),
    /// Measure selector inference latency over the exported fixtures.
    Cost(CostArgs),
}

#[derive(Debug, Args)]
struct FitArgs {
    /// Fit-data JSON exported by `oracle/m2_selectors.py prepare`.
    #[arg(long)]
    fit_data: PathBuf,
    /// Output weights JSON.
    #[arg(long)]
    out: PathBuf,
    /// Override the pairwise mixing weight (the Python export carries it).
    #[arg(long)]
    lambda_mix: Option<f64>,
    /// Print the weights JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
struct SelectArgs {
    /// Weights JSON.
    #[arg(long)]
    weights: PathBuf,
    /// Fixture records JSON.
    #[arg(long)]
    inputs: PathBuf,
    /// Selector name (magnitude, frequency, random, static-magnitude,
    /// static-frequency, learned-unary, learned-interaction).
    #[arg(long, default_value = "learned-unary")]
    selector: String,
    /// Groups to select per layer.
    #[arg(long, default_value_t = 5)]
    k: usize,
    /// Fail if any decision differs from the fixture's expected decision.
    #[arg(long)]
    check_expected: bool,
    /// Print the full JSON report.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
struct CostArgs {
    /// Weights JSON.
    #[arg(long)]
    weights: PathBuf,
    /// Fixture records JSON.
    #[arg(long)]
    inputs: PathBuf,
    /// Decisions per selector.
    #[arg(long, default_value_t = 10_000)]
    repeat: usize,
    /// Groups to select per layer.
    #[arg(long, default_value_t = 5)]
    k: usize,
    /// Print the JSON report.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Serialize)]
struct CostRow {
    selector: String,
    decisions: usize,
    total_seconds: f64,
    nanoseconds_per_decision: f64,
    derived_flops_per_decision: u64,
}

#[derive(Debug, Args)]
struct CompareArgs {
    /// Comparison input JSON (budgets, realized allocations, per-item KL).
    #[arg(long)]
    input: PathBuf,
    /// Bootstrap resamples.
    #[arg(long, default_value_t = 10_000)]
    resamples: usize,
    /// Print the full JSON report.
    #[arg(long)]
    json: bool,
}

#[derive(Serialize)]
struct SelectDecision {
    index: usize,
    layer: u32,
    selected: Vec<u32>,
    expected: Option<Vec<u32>>,
    matches: Option<bool>,
}

#[derive(Serialize)]
struct SelectReport {
    schema: &'static str,
    selector: String,
    k: usize,
    decisions: Vec<SelectDecision>,
    mismatches: usize,
}

fn load_json<T: serde::de::DeserializeOwned>(
    path: &PathBuf,
) -> Result<T, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}

fn write_json<T: Serialize>(path: &PathBuf, value: &T) -> Result<(), Box<dyn std::error::Error>> {
    let text = serde_json::to_string_pretty(value)?;
    std::fs::write(path, format!("{text}\n"))?;
    Ok(())
}

/// Run one Atlas subcommand.
pub(crate) fn run(args: &AtlasArgs) -> Result<ExitCode, Box<dyn std::error::Error>> {
    match &args.command {
        AtlasCommand::Fit(fit) => {
            let mut data: FitData = load_json(&fit.fit_data)?;
            if let Some(value) = fit.lambda_mix {
                data.lambda_mix = value;
            }
            let weights = fit_models(&data)?;
            write_json(&fit.out, &weights)?;
            println!(
                "atlas fit: {} layers, {} groups, {} controller elements -> {}",
                weights.layers,
                weights.groups,
                weights.parameter_elements,
                fit.out.display()
            );
            if fit.json {
                println!("{}", serde_json::to_string_pretty(&weights)?);
            }
            Ok(ExitCode::SUCCESS)
        }
        AtlasCommand::Select(select) => {
            let weights: AtlasWeights = load_json(&select.weights)?;
            weights.validate()?;
            let fixtures = read_fixtures(&select.inputs)?;
            let kind = SelectorKind::parse(&select.selector)?;
            let mut decisions = Vec::with_capacity(fixtures.records.len());
            let mut mismatches = 0_usize;
            for (index, record) in fixtures.records.iter().enumerate() {
                let input = record.input();
                input.validate(weights.layers as u32, weights.groups as u32)?;
                let selected = select_named(kind, &input, &weights, select.k)
                    .map_err(|error| format!("record {index}: {error}"))?;
                let expected = match kind {
                    SelectorKind::Magnitude => Some(record.expected.magnitude.clone()),
                    SelectorKind::Frequency => Some(record.expected.frequency.clone()),
                    SelectorKind::Random => Some(record.expected.random.clone()),
                    SelectorKind::StaticMagnitude => Some(record.expected.static_magnitude.clone()),
                    SelectorKind::StaticFrequency => Some(record.expected.static_frequency.clone()),
                    SelectorKind::LearnedUnary => Some(record.expected.learned_unary.clone()),
                    SelectorKind::LearnedInteraction => {
                        Some(record.expected.learned_interaction.clone())
                    }
                };
                let matches = expected.as_ref().map(|value| *value == selected);
                if select.check_expected && matches != Some(true) {
                    mismatches += 1;
                }
                decisions.push(SelectDecision {
                    index,
                    layer: record.layer,
                    selected,
                    expected,
                    matches,
                });
            }
            let report = SelectReport {
                schema: "selvedge/atlas/select-v1",
                selector: kind.name().to_string(),
                k: select.k,
                decisions,
                mismatches,
            };
            if select.json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!(
                    "atlas select: {} records, selector={}, mismatches={}",
                    report.decisions.len(),
                    report.selector,
                    report.mismatches
                );
            }
            if select.check_expected && mismatches > 0 {
                Ok(ExitCode::from(1))
            } else {
                Ok(ExitCode::SUCCESS)
            }
        }
        AtlasCommand::Cost(cost) => {
            let weights: AtlasWeights = load_json(&cost.weights)?;
            weights.validate()?;
            let fixtures = read_fixtures(&cost.inputs)?;
            let kinds = [
                SelectorKind::Magnitude,
                SelectorKind::Frequency,
                SelectorKind::Random,
                SelectorKind::StaticMagnitude,
                SelectorKind::StaticFrequency,
                SelectorKind::LearnedUnary,
                SelectorKind::LearnedInteraction,
            ];
            let mut rows = Vec::new();
            for kind in kinds {
                let inputs: Vec<_> = fixtures
                    .records
                    .iter()
                    .map(|record| record.input())
                    .collect();
                let start = std::time::Instant::now();
                for index in 0..cost.repeat {
                    let input = &inputs[index % inputs.len()];
                    let _ = select_named(kind, input, &weights, cost.k)
                        .map_err(|error| format!("{}: {error}", kind.name()))?;
                }
                let seconds = start.elapsed().as_secs_f64();
                let flops = match kind {
                    SelectorKind::StaticMagnitude | SelectorKind::StaticFrequency => 0,
                    SelectorKind::Magnitude | SelectorKind::Frequency => 2 * 5,
                    SelectorKind::Random => 8 * 5,
                    SelectorKind::LearnedUnary => 2 * 6 * 5,
                    SelectorKind::LearnedInteraction => 2 * 6 * 5 + 2 * 5 * 25 * 5,
                };
                rows.push(CostRow {
                    selector: kind.name().to_string(),
                    decisions: cost.repeat,
                    total_seconds: seconds,
                    nanoseconds_per_decision: seconds * 1e9 / cost.repeat as f64,
                    derived_flops_per_decision: flops,
                });
            }
            let report = serde_json::json!({
                "schema": "selvedge/atlas/cost-v1",
                "weights": cost.weights.display().to_string(),
                "records": fixtures.records.len(),
                "k": cost.k,
                "rows": rows,
            });
            if cost.json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                for row in report["rows"]
                    .as_array()
                    .map(|rows| rows.as_slice())
                    .unwrap_or(&[])
                {
                    println!(
                        "atlas cost: {:<22} {:.1} ns/decision  {} derived flops",
                        row["selector"].as_str().unwrap_or("?"),
                        row["nanoseconds_per_decision"].as_f64().unwrap_or(0.0),
                        row["derived_flops_per_decision"].as_u64().unwrap_or(0)
                    );
                }
            }
            Ok(ExitCode::SUCCESS)
        }
        AtlasCommand::Compare(compare) => {
            let input: CompareInput = load_json(&compare.input)?;
            let report = compare_models(&input, compare.resamples)?;
            let text = serde_json::to_string_pretty(&report)?;
            if compare.json {
                println!("{text}");
            } else {
                println!(
                    "atlas compare: ok={} arms={} comparisons={} ceiling={}",
                    report.ok,
                    report.arms.len(),
                    report.comparisons.len(),
                    report.ceiling
                );
                for arm in &report.arms {
                    println!(
                        "  {:<30} active={} delta={} matched={} budget={}",
                        arm.id,
                        arm.active_unique_elements,
                        arm.active_delta_from_baseline,
                        arm.matched,
                        arm.budget
                    );
                }
                for entry in &report.comparisons {
                    if let (Some(relative), Some(ci)) =
                        (entry.relative_reduction, entry.relative_ci)
                    {
                        println!(
                            "  {:<30} vs {} relative={:.4} CI=[{:.4}, {:.4}] passes={}",
                            entry.arm,
                            entry.baseline,
                            relative,
                            ci[0],
                            ci[1],
                            entry.passes_five_percent
                        );
                    }
                }
            }
            if report.ok {
                Ok(ExitCode::SUCCESS)
            } else {
                Ok(ExitCode::from(1))
            }
        }
    }
}
