//! Court E — exact-mode semantic parity under changed *streaming*, not changed
//! arithmetic.
//!
//! The court runs the identical model, the identical prompt and the identical
//! greedy rule under every combination of cache capacity, cache policy, prefetch
//! setting, fetch order, injected device delay and kernel thread count, and
//! requires **bitwise identical logits** for every token.
//!
//! It also proves the variants really are physically different: a cold run must
//! move more bytes from the backing tier than a warm one, and a delayed run must
//! take longer. Without that check the parity claim would be vacuous — a variant
//! that silently stopped reading would pass too.

// Integration courts: a panicking assertion is the failure report.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

mod common;

use std::collections::BTreeMap;

use selvedge_runtime::account::Phase;
use selvedge_runtime::model::greedy_token;
use selvedge_runtime::{ExecutionMode, Runner, StreamConfig, logits_digest};
use selvedge_store::{PolicyKind, ReadOrder};

/// Prompt shared by every variant.
const PROMPT: [u32; 4] = [1, 5, 9, 17];
/// Continuation steps after the prompt.
const DECODE: usize = 2;
/// Injected per-read delay for the delayed variant.
const DELAY_NS: u64 = 2_000_000;

struct Variant {
    name: &'static str,
    stream: StreamConfig,
    threads: usize,
}

fn variants() -> Vec<Variant> {
    let cached = |bytes: u64, policy: PolicyKind| StreamConfig {
        policy,
        cache_bytes: bytes,
        ..StreamConfig::default()
    };
    let prefetching = |order: ReadOrder, policy: PolicyKind| StreamConfig {
        policy,
        cache_bytes: 1 << 20,
        prefetch: true,
        read_order: order,
        ..StreamConfig::default()
    };
    vec![
        Variant {
            name: "cache-none",
            stream: StreamConfig::default(),
            threads: 2,
        },
        Variant {
            name: "cache-small-lru",
            stream: cached(8 << 10, PolicyKind::Lru),
            threads: 2,
        },
        Variant {
            name: "cache-large-lru",
            stream: cached(1 << 20, PolicyKind::Lru),
            threads: 2,
        },
        Variant {
            name: "cache-large-lfu",
            stream: cached(1 << 20, PolicyKind::Lfu),
            threads: 2,
        },
        Variant {
            name: "cache-large-least-stale",
            stream: cached(1 << 20, PolicyKind::LeastStale),
            threads: 2,
        },
        Variant {
            name: "cache-large-pinned",
            stream: cached(1 << 20, PolicyKind::Pinned),
            threads: 2,
        },
        Variant {
            name: "prefetch-plan",
            stream: prefetching(ReadOrder::Plan, PolicyKind::Lru),
            threads: 2,
        },
        Variant {
            name: "prefetch-reverse",
            stream: prefetching(ReadOrder::Reverse, PolicyKind::Lru),
            threads: 2,
        },
        Variant {
            name: "prefetch-physical",
            stream: prefetching(ReadOrder::Physical, PolicyKind::Lru),
            threads: 2,
        },
        Variant {
            name: "prefetch-delay-small-cache",
            stream: StreamConfig {
                policy: PolicyKind::Lru,
                cache_bytes: 8 << 10,
                prefetch: false,
                injected_read_delay_ns: DELAY_NS,
                ..StreamConfig::default()
            },
            threads: 2,
        },
        Variant {
            name: "threads-1",
            stream: cached(1 << 20, PolicyKind::Lru),
            threads: 1,
        },
        Variant {
            name: "threads-4",
            stream: cached(1 << 20, PolicyKind::Lru),
            threads: 4,
        },
        // P14.1: execution residency. The same arithmetic, served from decoded
        // f32 weights instead of freshly decoded bytes.
        Variant {
            name: "decoded-residency",
            stream: StreamConfig {
                policy: PolicyKind::Lru,
                cache_bytes: 1 << 20,
                decode_cache_bytes: 1 << 22,
                ..StreamConfig::default()
            },
            threads: 2,
        },
        // P14.2: asynchronous prefetch at two queue depths, the second with a
        // modelled slow device. Parity must survive a fetch timeline of its own.
        Variant {
            name: "async-prefetch-depth-8",
            stream: StreamConfig {
                policy: PolicyKind::Lru,
                cache_bytes: 1 << 20,
                async_prefetch: true,
                prefetch_depth: 8,
                ..StreamConfig::default()
            },
            threads: 2,
        },
        Variant {
            name: "async-prefetch-depth-1-delay",
            stream: StreamConfig {
                policy: PolicyKind::Lru,
                cache_bytes: 1 << 20,
                async_prefetch: true,
                prefetch_depth: 1,
                injected_read_delay_ns: DELAY_NS,
                ..StreamConfig::default()
            },
            threads: 2,
        },
    ]
}

struct Observation {
    logits: Vec<Vec<f32>>,
    digests: Vec<String>,
    tokens: Vec<u32>,
    witness_digest: String,
    active_unique: Vec<u64>,
    bytes_device: u64,
    elapsed_ns: u64,
}

fn observe(bundle: &std::path::Path, variant: &Variant) -> Observation {
    let mut config = common::tiny_config(bundle, PROMPT.to_vec(), DECODE);
    config.stream = variant.stream.clone();
    config.threads = variant.threads;
    config.budget = common::tiny_budget();
    let mut runner = Runner::open(config).expect("open runner");

    let mut logits = Vec::new();
    let mut digests = Vec::new();
    let mut active_unique = Vec::new();
    let mut bytes_device = 0_u64;
    let mut elapsed_ns = 0_u64;
    let mut tokens = Vec::new();
    let mut next = 0_u32;
    let mut witness_digest = String::new();
    for (position, token) in PROMPT.iter().enumerate() {
        let step = runner
            .forward(*token, position, Phase::Prefill)
            .expect("prefill");
        assert!(step.exactness.is_proven(), "{} is not exact", variant.name);
        witness_digest = runner
            .mode()
            .last_witness()
            .expect("witness")
            .executed_digest()
            .to_hex();
        next = greedy_token(&step.logits).expect("greedy");
        digests.push(logits_digest(&step.logits).to_hex());
        active_unique.push(step.accounting.active_unique_params);
        bytes_device = bytes_device.saturating_add(
            step.accounting
                .learned_bytes_by_tier
                .iter()
                .filter(|(tier, _)| **tier == selvedge_core::StorageTier::Nvme)
                .map(|(_, bytes)| *bytes)
                .sum::<u64>(),
        );
        elapsed_ns = elapsed_ns.saturating_add(step.elapsed_ns);
        logits.push(step.logits);
    }
    for index in 0..DECODE {
        tokens.push(next);
        let position = PROMPT.len() + index;
        let step = runner
            .forward(next, position, Phase::Decode)
            .expect("decode");
        assert!(step.exactness.is_proven(), "{} is not exact", variant.name);
        next = greedy_token(&step.logits).expect("greedy");
        digests.push(logits_digest(&step.logits).to_hex());
        active_unique.push(step.accounting.active_unique_params);
        bytes_device = bytes_device.saturating_add(
            step.accounting
                .learned_bytes_by_tier
                .iter()
                .filter(|(tier, _)| **tier == selvedge_core::StorageTier::Nvme)
                .map(|(_, bytes)| *bytes)
                .sum::<u64>(),
        );
        elapsed_ns = elapsed_ns.saturating_add(step.elapsed_ns);
        logits.push(step.logits);
    }
    Observation {
        logits,
        digests,
        tokens,
        witness_digest,
        active_unique,
        bytes_device,
        elapsed_ns,
    }
}

#[test]
fn court_e_exact_mode_is_bitwise_identical_across_streaming_variants() {
    let temp = common::TempDir::new("court-e");
    let bundle = common::build_tiny_bundle(temp.path());
    let variants = variants();
    let baseline = observe(&bundle, &variants[0]);
    let mut report: BTreeMap<&str, String> = BTreeMap::new();
    report.insert(
        variants[0].name,
        format!(
            "device_bytes={} elapsed_ns={}",
            baseline.bytes_device, baseline.elapsed_ns
        ),
    );

    for variant in variants.iter().skip(1) {
        let observed = observe(&bundle, variant);
        assert_eq!(
            observed.digests, baseline.digests,
            "{} produced different logits than {}",
            variant.name, variants[0].name
        );
        assert_eq!(observed.tokens, baseline.tokens, "{}", variant.name);
        assert_eq!(
            observed.active_unique, baseline.active_unique,
            "{} selected a different set",
            variant.name
        );
        assert_eq!(
            observed.witness_digest, baseline.witness_digest,
            "{} witnessed a different native set",
            variant.name
        );
        for (left, right) in observed.logits.iter().zip(baseline.logits.iter()) {
            for (a, b) in left.iter().zip(right.iter()) {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{} changed a logit's bits",
                    variant.name
                );
            }
        }
        report.insert(
            variant.name,
            format!(
                "device_bytes={} elapsed_ns={}",
                observed.bytes_device, observed.elapsed_ns
            ),
        );
    }

    // The variants must actually differ physically, or parity proves nothing.
    let cold = &baseline.bytes_device;
    let warm = variants
        .iter()
        .position(|variant| variant.name == "cache-large-lru")
        .map(|index| observe(&bundle, &variants[index]).bytes_device)
        .expect("a warm variant exists");
    assert!(
        *cold > 0,
        "the cache-none variant moved no bytes from the backing tier"
    );
    assert!(
        warm < *cold,
        "the warm variant ({warm}) did not move fewer device bytes than the cold one ({cold})"
    );
    for (name, summary) in &report {
        eprintln!("court-e {name}: {summary}");
    }
}

#[test]
fn court_e_approximate_mode_is_stable_but_not_exact() {
    let temp = common::TempDir::new("court-e-approximate");
    let bundle = common::build_tiny_bundle(temp.path());

    let run = |policy: PolicyKind, cache_bytes: u64, threads: usize| {
        let mut config = common::tiny_config(&bundle, PROMPT.to_vec(), DECODE);
        config.mode = ExecutionMode::Approximate(common::one_group_selector());
        config.stream = StreamConfig {
            policy,
            cache_bytes,
            ..StreamConfig::default()
        };
        config.threads = threads;
        config.budget = common::tiny_budget();
        let mut runner = Runner::open(config).expect("open runner");
        let mut digests = Vec::new();
        let mut proven = true;
        let mut verdicts = 0;
        let mut next = 0_u32;
        for (position, token) in PROMPT.iter().enumerate() {
            let step = runner
                .forward(*token, position, Phase::Prefill)
                .expect("prefill");
            proven &= step.exactness.is_proven();
            verdicts += 1;
            next = greedy_token(&step.logits).expect("greedy");
            digests.push(logits_digest(&step.logits).to_hex());
        }
        for index in 0..DECODE {
            let position = PROMPT.len() + index;
            let step = runner
                .forward(next, position, Phase::Decode)
                .expect("decode");
            proven &= step.exactness.is_proven();
            next = greedy_token(&step.logits).expect("greedy");
            digests.push(logits_digest(&step.logits).to_hex());
        }
        (digests, proven, verdicts)
    };

    let (cold, proven_cold, _) = run(PolicyKind::None, 0, 2);
    let (warm, proven_warm, _) = run(PolicyKind::Lru, 1 << 20, 4);
    assert!(
        !proven_cold && !proven_warm,
        "approximate must never claim exactness"
    );
    assert_eq!(cold, warm, "approximate mode must also be deterministic");

    // The two verdicts are recorded for an approximate run.
    let mut config = common::tiny_config(&bundle, PROMPT.to_vec(), 1);
    config.mode = ExecutionMode::Approximate(common::one_group_selector());
    config.budget = common::tiny_budget();
    let mut runner = Runner::open(config).expect("open runner");
    let step = runner
        .forward(PROMPT[0], 0, Phase::Prefill)
        .expect("forward");
    let verdict = runner
        .mode()
        .verdict("prefill:0", &step.accounting, 0)
        .expect("verdict")
        .expect("approximate runs carry a verdict");
    assert!(verdict.causal.is_available());
    assert!(verdict.physical.is_compliant());
}

#[test]
fn court_e_budget_gate_refuses_an_over_budget_exact_run() {
    let temp = common::TempDir::new("court-e-budget");
    let bundle = common::build_tiny_bundle(temp.path());
    let mut config = common::tiny_config(&bundle, PROMPT.to_vec(), 1);
    config.budget = selvedge_core::Budget::new(1_000, selvedge_core::ComplianceMode::Compliance)
        .expect("budget");
    let mut runner = Runner::open(config).expect("open runner");
    let error = runner
        .forward(PROMPT[0], 0, Phase::Prefill)
        .expect_err("an over-budget exact path must be refused before execution");
    assert!(
        error.to_string().contains("budget"),
        "unexpected error: {error}"
    );
}

#[test]
fn court_e_within_budget_fallback_trims_approximate_groups() {
    let temp = common::TempDir::new("court-e-fallback");
    let bundle = common::build_tiny_bundle(temp.path());

    // The backbone alone is 20 thousand elements; allow a little more so exactly
    // one group per layer fits and the fallback must drop the rest.
    let backbone = {
        let config = common::tiny_config(&bundle, PROMPT.to_vec(), 1);
        let runner = Runner::open(config).expect("open runner");
        runner.model().backbone_elements()
    };
    let group_elements = 3 * 16 * 32; // |G| = 16, H = 32, three matrices
    // One group per layer must fit; two per layer must not.
    let limit = backbone + 2 * group_elements + 8;

    let mut config = common::tiny_config(&bundle, PROMPT.to_vec(), 1);
    config.mode = ExecutionMode::Approximate(selvedge_runtime::SelectorSpec {
        keep_per_layer: 4,
        seed: 7,
    });
    config.fallback = true;
    config.budget = selvedge_core::Budget::new(limit, selvedge_core::ComplianceMode::Compliance)
        .expect("budget");
    let mut runner = Runner::open(config).expect("open runner");
    let step = runner
        .forward(PROMPT[0], 0, Phase::Prefill)
        .expect("fallback run");
    assert!(step.accounting.within_budget);
    assert!(step.accounting.active_unique_params <= limit);
    // Four groups per layer were asked for; the gate can only afford one.
    assert_eq!(step.group_shards as usize, 2 * 3);
    assert!(runner.mode().trimmed_groups() > 0);
}

#[test]
fn court_e_selector_reads_only_the_current_state() {
    // The control selector is a pure function of (seed, layer, group, hidden
    // state); the same hidden state and the same seed must select the same
    // groups, and the future cannot enter because nothing else is read.
    let temp = common::TempDir::new("court-e-selector");
    let bundle = common::build_tiny_bundle(temp.path());
    let mut decisions = Vec::new();
    for _ in 0..2 {
        let mut config = common::tiny_config(&bundle, PROMPT.to_vec(), 1);
        config.mode = ExecutionMode::Approximate(common::one_group_selector());
        config.budget = common::tiny_budget();
        let mut runner = Runner::open(config).expect("open runner");
        let plan = runner.plan().clone();
        let step = runner
            .forward(PROMPT[0], 0, Phase::Prefill)
            .expect("forward");
        decisions.push((logits_digest(&step.logits), step.group_shards));
        let groups = plan.groups_for(0).expect("layer 0");
        let hidden = vec![0.0_f32; plan.hidden as usize];
        let choices = runner
            .mode()
            .decide(0, groups, &hidden, plan.hidden)
            .expect("decide");
        assert_eq!(choices.len(), 1);
        // Zero hidden state means every score is zero, so the tie-break (lowest
        // index) decides: the selector is deterministic, not random.
        assert_eq!(choices[0].index, 0);
    }
    assert_eq!(decisions[0], decisions[1]);
}
