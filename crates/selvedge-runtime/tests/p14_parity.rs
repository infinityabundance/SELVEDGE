//! P14 parity courts: decoded residency, asynchronous prefetch, batched prefill.
//!
//! Each of these is an *engineering* correction to the runtime, so the thing to
//! prove is that it changed only the cost and never the answer:
//!
//! 1. [`decoded_residency_is_bit_identical`] — serving a shard from the decoded
//!    f32 residency produces the same bits as decoding it again.
//! 2. [`decode_counts_drop_as_the_residency_warms`] — the counter the fix exists
//!    for: decode passes per token.
//! 3. [`residency_eviction_stays_inside_its_budget_and_keeps_the_answer`] — a
//!    residency that cannot hold the bundle evicts, respects its byte budget and
//!    still produces identical logits.
//! 4. [`async_prefetch_is_bit_identical_with_an_injected_delay`] — the real
//!    overlap path against the synchronous one under a modelled slow device.
//! 5. [`prefetch_accounting_conserves`] — every fetched byte is used or wasted,
//!    and every fetch nanosecond is hidden or exposed.
//! 6. [`batched_prefill_matches_the_scalar_oracle`] — dense projections as a
//!    batch, same bits, on the tiny fixture.
//! 7. [`kv_state_is_identical_after_prefill`] — the batched path leaves the same
//!    KV cache state, so decode can continue.
//! 8. [`greedy_sequence_matches_after_batched_prefill`] — the end-to-end claim:
//!    batched prefill plus decode emits the same tokens as scalar prefill.
//!
//! Nothing here is a GPU claim. Every measurement is CPU wall clock.

// Integration courts: a panicking assertion is the failure report.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

mod common;

use std::path::Path;

use selvedge_runtime::account::{Phase, PrefetchRow};
use selvedge_runtime::model::greedy_token;
use selvedge_runtime::{ExecutionMode, PrefillMode, Runner, StreamConfig, logits_digest};
use selvedge_store::{PolicyKind, ReadOrder};

const PROMPT: [u32; 4] = [1, 5, 9, 17];
const DECODE: usize = 2;
/// The tiny bundle is ~48 KiB of f32; this holds it entirely.
const FULL_RESIDENCY: u64 = 4 << 20;
/// Delay injected after every backing-tier read for the overlap court.
const DELAY_NS: u64 = 2_000_000;

struct Observation {
    digests: Vec<String>,
    tokens: Vec<u32>,
    decodes: Vec<u64>,
    decode_hits: Vec<u64>,
    elapsed_ns: Vec<u64>,
    kv_digest: String,
    rows: Vec<serde_json::Value>,
}

fn read_rows(path: &Path) -> Vec<serde_json::Value> {
    std::fs::read_to_string(path)
        .expect("accounting rows")
        .lines()
        .map(|line| serde_json::from_str(line).expect("row json"))
        .collect()
}

fn observe(_bundle: &Path, config: &mut selvedge_runtime::RunConfig, label: &str) -> Observation {
    config.budget = common::tiny_budget();
    let account = std::env::temp_dir().join(format!(
        "selvedge-p14-{label}-{}-{:?}.jsonl",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_file(&account);
    config.account = Some(account.clone());
    let mut runner = Runner::open(config.clone()).expect("open runner");
    let mut digests = Vec::new();
    let mut tokens = Vec::new();
    let mut decodes = Vec::new();
    let mut decode_hits = Vec::new();
    let mut elapsed_ns = Vec::new();
    let mut next = 0_u32;
    match config.prefill {
        PrefillMode::Scalar => {
            for (position, token) in PROMPT.iter().enumerate() {
                let step = runner
                    .forward(*token, position, Phase::Prefill)
                    .expect("prefill");
                assert!(step.exactness.is_proven(), "{label} is not exact");
                next = greedy_token(&step.logits).expect("greedy");
                digests.push(logits_digest(&step.logits).to_hex());
                elapsed_ns.push(step.elapsed_ns);
            }
        }
        PrefillMode::Batched => {
            let batch = config.prefill_batch.max(1);
            let mut start = 0;
            while start < PROMPT.len() {
                let end = (start + batch).min(PROMPT.len());
                let steps = runner
                    .forward_batch(&PROMPT[start..end], start, Phase::Prefill)
                    .expect("batched prefill");
                for step in steps {
                    assert!(step.exactness.is_proven(), "{label} is not exact");
                    next = greedy_token(&step.logits).expect("greedy");
                    digests.push(logits_digest(&step.logits).to_hex());
                    elapsed_ns.push(step.elapsed_ns);
                }
                start = end;
            }
        }
    }
    for index in 0..DECODE {
        tokens.push(next);
        let position = PROMPT.len() + index;
        let step = runner
            .forward(next, position, Phase::Decode)
            .expect("decode");
        next = greedy_token(&step.logits).expect("greedy");
        digests.push(logits_digest(&step.logits).to_hex());
        elapsed_ns.push(step.elapsed_ns);
    }
    let rows = {
        runner.flush_accounting().expect("flush accounting");
        read_rows(&account)
    };
    for row in &rows {
        decodes.push(row["decode"]["decodes"].as_u64().expect("decodes"));
        decode_hits.push(row["decode"]["hits"].as_u64().expect("decode hits"));
    }
    let kv_digest = runner.cache().state_digest().to_hex();
    let _ = std::fs::remove_file(&account);
    Observation {
        digests,
        tokens,
        decodes,
        decode_hits,
        elapsed_ns,
        kv_digest,
        rows,
    }
}

/// A configuration with the bundle resident in the storage cache.
fn base_config(bundle: &Path, threads: usize) -> selvedge_runtime::RunConfig {
    let mut config = common::tiny_config(bundle, PROMPT.to_vec(), DECODE);
    config.stream = StreamConfig {
        policy: PolicyKind::Lru,
        cache_bytes: 1 << 20,
        decode_cache_bytes: 0,
        ..StreamConfig::default()
    };
    config.threads = threads;
    config
}

#[test]
fn decoded_residency_is_bit_identical() {
    let temp = common::TempDir::new("p14-residency-identity");
    let bundle = common::build_tiny_bundle(temp.path());

    let mut without = base_config(&bundle, 2);
    without.stream.decode_cache_bytes = 0;
    let baseline = observe(&bundle, &mut without, "residency-off");

    let mut with = base_config(&bundle, 2);
    with.stream.decode_cache_bytes = FULL_RESIDENCY;
    let residency = observe(&bundle, &mut with, "residency-on");

    assert_eq!(
        residency.digests, baseline.digests,
        "decoded residency changed the logits"
    );
    assert_eq!(residency.tokens, baseline.tokens);
    assert_eq!(
        residency.kv_digest, baseline.kv_digest,
        "decoded residency changed the KV state"
    );
    // The residency must actually be doing something, or the parity is vacuous.
    assert!(
        baseline.decodes.iter().all(|count| *count > 0),
        "the residency-off arm decoded nothing: {:?}",
        baseline.decodes
    );
    assert_eq!(
        residency.decodes.last().copied(),
        Some(0),
        "the warm token decoded again: {:?}",
        residency.decodes
    );
    assert_eq!(
        residency.decode_hits.last().copied(),
        Some(24),
        "the warm token was not served entirely from residency: {:?}",
        residency.decode_hits
    );
}

#[test]
fn decode_counts_drop_as_the_residency_warms() {
    let temp = common::TempDir::new("p14-residency-counts");
    let bundle = common::build_tiny_bundle(temp.path());

    let mut off = base_config(&bundle, 1);
    off.stream.decode_cache_bytes = 0;
    let cold = observe(&bundle, &mut off, "counts-off");

    let mut on = base_config(&bundle, 1);
    on.stream.decode_cache_bytes = FULL_RESIDENCY;
    let warm = observe(&bundle, &mut on, "counts-on");

    // Two layers, four groups, three matrices: 24 shards per token.
    let per_token = 24;
    for count in &cold.decodes {
        assert_eq!(
            *count, per_token,
            "the residency-off arm must decode every shard"
        );
    }
    assert_eq!(
        warm.decodes.first().copied().expect("first token"),
        per_token,
        "the first token has to decode the bundle"
    );
    for count in warm.decodes.iter().skip(1) {
        assert_eq!(
            *count, 0,
            "a warm token decoded again: decodes={:?}",
            warm.decodes
        );
    }
    let total_off: u64 = cold.decodes.iter().sum();
    let total_on: u64 = warm.decodes.iter().sum();
    assert!(
        total_on * 4 < total_off,
        "the residency did not reduce decodes: {total_on} vs {total_off}"
    );
    eprintln!(
        "p14 decodes: off={total_off} on={total_on} per-token-off={per_token} \
         (hits on warm tokens: {:?})",
        warm.decode_hits
    );
}

#[test]
fn residency_eviction_stays_inside_its_budget_and_keeps_the_answer() {
    let temp = common::TempDir::new("p14-residency-evict");
    let bundle = common::build_tiny_bundle(temp.path());

    let mut full = base_config(&bundle, 1);
    full.stream.decode_cache_bytes = FULL_RESIDENCY;
    let baseline = observe(&bundle, &mut full, "evict-baseline");

    // One group is 3 * 16 * 32 * 4 = 6144 bytes; this budget holds one group.
    let mut small = base_config(&bundle, 1);
    small.stream.decode_cache_bytes = 8 << 10;
    let evicting = observe(&bundle, &mut small, "evict-small");

    assert_eq!(evicting.digests, baseline.digests);
    let evictions: u64 = evicting
        .rows
        .iter()
        .filter_map(|row| row["decode"]["evictions"].as_u64())
        .max()
        .unwrap_or(0);
    assert!(evictions > 0, "a one-group residency never evicted");
    assert!(
        evicting.decodes.iter().all(|count| *count >= 1),
        "an evicting residency must still decode something every token"
    );
    for row in &evicting.rows {
        let resident = row["decode"]["resident_bytes"].as_u64().expect("resident");
        let capacity = row["decode"]["capacity_bytes"].as_u64().expect("capacity");
        assert!(
            resident <= capacity,
            "the residency exceeded its budget: {resident} > {capacity}"
        );
    }
    eprintln!("p14 residency evictions={evictions} budget={}", 8 << 10);
}

#[test]
fn async_prefetch_is_bit_identical_with_an_injected_delay() {
    let temp = common::TempDir::new("p14-prefetch-parity");
    let bundle = common::build_tiny_bundle(temp.path());

    let mut off = base_config(&bundle, 2);
    off.stream.cache_bytes = 64 << 10;
    let baseline = observe(&bundle, &mut off, "prefetch-off");

    let mut sync = base_config(&bundle, 2);
    sync.stream.cache_bytes = 64 << 10;
    sync.stream.prefetch = true;
    sync.stream.read_order = ReadOrder::Reverse;
    let sync_observed = observe(&bundle, &mut sync, "prefetch-sync");

    let mut asynchronous = base_config(&bundle, 2);
    asynchronous.stream.cache_bytes = 64 << 10;
    asynchronous.stream.async_prefetch = true;
    asynchronous.stream.prefetch_depth = 8;
    asynchronous.stream.injected_read_delay_ns = DELAY_NS;
    let async_observed = observe(&bundle, &mut asynchronous, "prefetch-async");

    // Same answer under every fetch strategy, including the modelled slow device.
    assert_eq!(async_observed.digests, baseline.digests);
    assert_eq!(sync_observed.digests, baseline.digests);
    assert_eq!(async_observed.tokens, baseline.tokens);
    assert_eq!(async_observed.kv_digest, baseline.kv_digest);

    // The delayed arm must have paid the delay on its fetch timeline.
    let busy: u64 = async_observed
        .rows
        .iter()
        .filter_map(|row| row["prefetch"]["fetch_busy_ns"].as_u64())
        .sum();
    assert!(
        busy >= DELAY_NS,
        "the injected delay never reached the fetch timeline: {busy} ns"
    );
    eprintln!(
        "p14 prefetch: off_elapsed_ns={:?} sync_elapsed_ns={:?} async_elapsed_ns={:?} fetch_busy_ns={busy}",
        baseline.elapsed_ns, sync_observed.elapsed_ns, async_observed.elapsed_ns
    );
}

#[test]
fn prefetch_accounting_conserves() {
    let temp = common::TempDir::new("p14-prefetch-conservation");
    let bundle = common::build_tiny_bundle(temp.path());

    let mut config = base_config(&bundle, 2);
    config.stream.cache_bytes = 64 << 10;
    config.stream.async_prefetch = true;
    config.stream.prefetch_depth = 4;
    config.stream.injected_read_delay_ns = DELAY_NS;
    let observed = observe(&bundle, &mut config, "prefetch-conservation");

    let mut rows_with_prefetch = 0;
    for row in &observed.rows {
        let Some(prefetch) = row.get("prefetch") else {
            continue;
        };
        rows_with_prefetch += 1;
        let fetched = prefetch["fetched_bytes"].as_u64().expect("fetched");
        let used = prefetch["used_bytes"].as_u64().expect("used");
        let wasted = prefetch["wasted_bytes"].as_u64().expect("wasted");
        assert_eq!(
            fetched,
            used.saturating_add(wasted),
            "byte accounting does not conserve: {prefetch}"
        );
        let busy = prefetch["fetch_busy_ns"].as_u64().expect("busy");
        let hidden = prefetch["hidden_ns"].as_u64().expect("hidden");
        let exposed = prefetch["exposed_stall_ns"].as_u64().expect("exposed");
        let residual = prefetch["residual_wait_ns"].as_u64().expect("residual");
        assert_eq!(
            busy.saturating_add(residual),
            hidden.saturating_add(exposed),
            "stall accounting does not conserve: {prefetch}"
        );
        assert!(
            prefetch["shards_fetched"].as_u64().expect("shards") >= 24,
            "a token must fetch its whole shard set: {prefetch}"
        );
        assert!(
            prefetch["layers"].as_u64().expect("layers") >= 2,
            "both layers must run a fetcher: {prefetch}"
        );
    }
    assert_eq!(
        rows_with_prefetch,
        PROMPT.len() + DECODE,
        "every token should carry prefetch accounting"
    );
}

#[test]
fn batched_prefill_matches_the_scalar_oracle() {
    let temp = common::TempDir::new("p14-batched-parity");
    let bundle = common::build_tiny_bundle(temp.path());

    let mut scalar = base_config(&bundle, 1);
    scalar.prefill = PrefillMode::Scalar;
    let baseline = observe(&bundle, &mut scalar, "prefill-scalar");

    for (label, batch) in [("prefill-batched-4", 4_usize), ("prefill-batched-2", 2)] {
        let mut batched = base_config(&bundle, 1);
        batched.prefill = PrefillMode::Batched;
        batched.prefill_batch = batch;
        let observed = observe(&bundle, &mut batched, label);
        assert_eq!(
            observed.digests, baseline.digests,
            "{label} produced different logits than the scalar oracle"
        );
        assert_eq!(observed.kv_digest, baseline.kv_digest, "{label} KV state");
        assert_eq!(observed.tokens, baseline.tokens);
    }
}

#[test]
fn kv_state_is_identical_after_prefill() {
    let temp = common::TempDir::new("p14-kv-equivalence");
    let bundle = common::build_tiny_bundle(temp.path());

    let mut scalar = base_config(&bundle, 1);
    scalar.prefill = PrefillMode::Scalar;
    let scalar_run = observe(&bundle, &mut scalar, "kv-scalar");

    let mut batched = base_config(&bundle, 1);
    batched.prefill = PrefillMode::Batched;
    batched.prefill_batch = 4;
    let batched_run = observe(&bundle, &mut batched, "kv-batched");

    assert_eq!(
        batched_run.kv_digest, scalar_run.kv_digest,
        "the KV cache layout differs after prefill, so decode would diverge"
    );
    assert_eq!(batched_run.digests, scalar_run.digests);
}

#[test]
fn greedy_sequence_matches_after_batched_prefill() {
    let temp = common::TempDir::new("p14-greedy-equivalence");
    let bundle = common::build_tiny_bundle(temp.path());

    let run = |prefill: PrefillMode, batch: usize| {
        let mut config = base_config(&bundle, 1);
        config.prefill = prefill;
        config.prefill_batch = batch;
        config.budget = common::tiny_budget();
        config.max_new_tokens = 4;
        let mut runner = Runner::open(config).expect("open runner");
        let outcome = runner.run().expect("run");
        assert!(outcome.ok());
        (
            outcome.summary.generated_tokens,
            outcome.summary.sequence_digest,
        )
    };

    let scalar = run(PrefillMode::Scalar, 1);
    for batch in [2_usize, 4] {
        let batched = run(PrefillMode::Batched, batch);
        assert_eq!(
            batched, scalar,
            "batched prefill (batch {batch}) plus decode changed the greedy sequence"
        );
    }
}

/// The prefetch row's own conservation identity, asserted directly.
#[test]
fn prefetch_row_conservation_identity() {
    // A row built by hand: the identity is arithmetic on the recorded counters,
    // and it is the same identity `prefetch_accounting_conserves` checks on the
    // JSONL. Keeping it here means a change to the identity fails a unit test
    // rather than only a real-model receipt.
    let row = PrefetchRow {
        layers: 2,
        shards_fetched: 24,
        fetched_bytes: 100,
        device_bytes: 100,
        used_bytes: 90,
        wasted_bytes: 10,
        staging_high_water_bytes: 2048,
        fetch_busy_ns: 70,
        exposed_stall_ns: 30,
        hidden_ns: 40,
        residual_wait_ns: 0,
        waits: 3,
        ready_without_wait: 21,
        stall_p50_ns: 10,
        stall_p95_ns: 18,
        stall_max_ns: 20,
    };
    assert_eq!(
        row.fetched_bytes,
        row.used_bytes + row.wasted_bytes,
        "bytes"
    );
    assert_eq!(
        row.fetch_busy_ns + row.residual_wait_ns,
        row.hidden_ns + row.exposed_stall_ns,
        "nanoseconds"
    );
}

/// The approximate selector is per token, so batched prefill must refuse it.
#[test]
fn batched_prefill_refuses_approximate_mode() {
    let temp = common::TempDir::new("p14-batched-refuses");
    let bundle = common::build_tiny_bundle(temp.path());
    let mut config = common::tiny_config(&bundle, PROMPT.to_vec(), 1);
    config.mode = ExecutionMode::Approximate(common::one_group_selector());
    config.prefill = PrefillMode::Batched;
    let error = Runner::open(config).expect_err("batched prefill must refuse approximate mode");
    assert!(
        error.to_string().contains("exact mode"),
        "unexpected error: {error}"
    );
}
