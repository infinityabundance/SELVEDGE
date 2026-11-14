//! Offline tests for exact parameter accounting on a tiny Qwen3-shaped
//! checkpoint. The real checkpoints are exercised by court receipts; these
//! tests pin the arithmetic, the tie verification and every rejection path.

#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::path::Path;

use selvedge_convert::accounting::{
    AccountingOptions, DEFAULT_ACTIVE_CEILING, account_model, category_totals,
};
use selvedge_convert::synthetic::{
    SyntheticDType, SyntheticModel, SyntheticTensor, write_safetensors,
};
use selvedge_core::testutil::TempDir;

const HIDDEN: u64 = 4;
const INTERMEDIATE: u64 = 3;
const HEADS: u64 = 2;
const KV_HEADS: u64 = 1;
const HEAD_DIM: u64 = 2;
const LAYERS: u64 = 1;
const VOCAB: u64 = 5;
const Q_WIDTH: u64 = HEADS * HEAD_DIM;

fn values(shape: &[u64], seed: f32) -> Vec<f32> {
    let count: u64 = shape.iter().product();
    (0..count)
        .map(|index| seed + index as f32 * 0.125)
        .collect()
}

fn tensor(shape: &[u64], seed: f32) -> SyntheticTensor {
    SyntheticTensor {
        shape: shape.to_vec(),
        values: values(shape, seed),
    }
}

fn base_tensors() -> BTreeMap<String, SyntheticTensor> {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_owned(),
        tensor(&[VOCAB, HIDDEN], 0.0),
    );
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_owned(),
        tensor(&[Q_WIDTH, HIDDEN], 1.0),
    );
    tensors.insert(
        "model.layers.0.self_attn.k_proj.weight".to_owned(),
        tensor(&[KV_HEADS * HEAD_DIM, HIDDEN], 2.0),
    );
    tensors.insert(
        "model.layers.0.self_attn.v_proj.weight".to_owned(),
        tensor(&[KV_HEADS * HEAD_DIM, HIDDEN], 3.0),
    );
    tensors.insert(
        "model.layers.0.self_attn.o_proj.weight".to_owned(),
        tensor(&[HIDDEN, Q_WIDTH], 4.0),
    );
    tensors.insert(
        "model.layers.0.self_attn.q_norm.weight".to_owned(),
        tensor(&[HEAD_DIM], 5.0),
    );
    tensors.insert(
        "model.layers.0.self_attn.k_norm.weight".to_owned(),
        tensor(&[HEAD_DIM], 6.0),
    );
    tensors.insert(
        "model.layers.0.input_layernorm.weight".to_owned(),
        tensor(&[HIDDEN], 7.0),
    );
    tensors.insert(
        "model.layers.0.post_attention_layernorm.weight".to_owned(),
        tensor(&[HIDDEN], 8.0),
    );
    tensors.insert(
        "model.layers.0.mlp.gate_proj.weight".to_owned(),
        tensor(&[INTERMEDIATE, HIDDEN], 9.0),
    );
    tensors.insert(
        "model.layers.0.mlp.up_proj.weight".to_owned(),
        tensor(&[INTERMEDIATE, HIDDEN], 10.0),
    );
    tensors.insert(
        "model.layers.0.mlp.down_proj.weight".to_owned(),
        tensor(&[HIDDEN, INTERMEDIATE], 11.0),
    );
    tensors.insert("model.norm.weight".to_owned(), tensor(&[HIDDEN], 12.0));
    tensors
}

fn model(tensors: BTreeMap<String, SyntheticTensor>) -> SyntheticModel {
    SyntheticModel {
        hidden: HIDDEN,
        intermediate: INTERMEDIATE,
        seed: 0x5E1A_ED6E,
        tensors,
        maps: Vec::new(),
    }
}

fn write_config(dir: &Path, tie: bool) {
    let config = format!(
        "{{\n  \"architectures\": [\"Qwen3ForCausalLM\"],\n  \"model_type\": \"qwen3\",\n  \
         \"hidden_size\": {HIDDEN},\n  \"intermediate_size\": {INTERMEDIATE},\n  \
         \"num_hidden_layers\": {LAYERS},\n  \"num_attention_heads\": {HEADS},\n  \
         \"num_key_value_heads\": {KV_HEADS},\n  \"head_dim\": {HEAD_DIM},\n  \
         \"vocab_size\": {VOCAB},\n  \"tie_word_embeddings\": {tie}\n}}\n"
    );
    std::fs::write(dir.join("config.json"), config).unwrap();
}

fn checkpoint(dir: &Path, tensors: BTreeMap<String, SyntheticTensor>, tie: bool) {
    write_config(dir, tie);
    write_safetensors(
        &model(tensors),
        SyntheticDType::F32,
        &dir.join("model.safetensors"),
    )
    .unwrap();
}

fn tied_tensors() -> BTreeMap<String, SyntheticTensor> {
    let mut tensors = base_tensors();
    let embed = tensors.get("model.embed_tokens.weight").unwrap().clone();
    tensors.insert("lm_head.weight".to_owned(), embed);
    tensors
}

#[test]
fn tiny_qwen3_accounting_is_exact_and_tie_aware() {
    let dir = TempDir::new("accounting-tied").unwrap();
    checkpoint(dir.path(), tied_tensors(), true);
    let report = account_model(dir.path(), &AccountingOptions::default()).unwrap();

    assert!(report.ok);
    assert_eq!(report.model.layers, LAYERS);
    assert_eq!(report.model.head_dim, HEAD_DIM);
    assert_eq!(report.model.kv_heads, KV_HEADS);
    assert!(report.tie.lm_head_present);
    assert!(report.tie.lm_head_identical_to_embedding);
    assert!(report.tie.treated_as_shared);
    assert_eq!(report.tie.compared_elements, VOCAB * HIDDEN);

    let totals = &report.totals;
    assert_eq!(totals.embedding, VOCAB * HIDDEN);
    let attention =
        (Q_WIDTH * HIDDEN) + 2 * (KV_HEADS * HEAD_DIM * HIDDEN) + (HIDDEN * Q_WIDTH) + 2 * HEAD_DIM;
    assert_eq!(totals.attention, attention);
    assert_eq!(totals.norm, 2 * HIDDEN + HIDDEN);
    assert_eq!(totals.ffn, 3 * INTERMEDIATE * HIDDEN);
    assert_eq!(totals.lm_head, VOCAB * HIDDEN);

    let checkpoint_elements =
        VOCAB * HIDDEN + attention + totals.norm + totals.ffn + VOCAB * HIDDEN;
    assert_eq!(
        report.budget.checkpoint_tensor_elements,
        checkpoint_elements
    );
    assert_eq!(
        report.budget.unique_learned_elements,
        checkpoint_elements - VOCAB * HIDDEN
    );
    assert_eq!(
        report.budget.always_active_backbone_elements,
        totals.embedding + totals.attention + totals.norm
    );
    assert_eq!(report.budget.removable_ffn_elements, totals.ffn);
    assert_eq!(report.budget.verdict, "not_binding");
    assert!(!report.budget.ceiling_binding);
    assert!(report.budget.dense_execution_compliant);

    let categories = category_totals(&report);
    assert_eq!(categories.len(), 5);
}

#[test]
fn untied_pair_counts_twice_and_ceiling_verdict_tracks_it() {
    let dir = TempDir::new("accounting-untied").unwrap();
    let mut tensors = base_tensors();
    tensors.insert("lm_head.weight".to_owned(), tensor(&[VOCAB, HIDDEN], 30.0));
    checkpoint(dir.path(), tensors, false);
    let report = account_model(dir.path(), &AccountingOptions::default()).unwrap();
    assert!(!report.tie.treated_as_shared);
    assert!(!report.tie.lm_head_identical_to_embedding);
    let unique = report.budget.checkpoint_tensor_elements;
    assert_eq!(report.budget.unique_learned_elements, unique);
    assert_eq!(
        report.budget.always_active_backbone_elements,
        report.totals.embedding
            + report.totals.attention
            + report.totals.norm
            + report.totals.lm_head
    );
}

#[test]
fn a_ceiling_between_backbone_and_dense_is_binding() {
    let dir = TempDir::new("accounting-binding").unwrap();
    checkpoint(dir.path(), tied_tensors(), true);
    let full = account_model(dir.path(), &AccountingOptions::default()).unwrap();
    let backbone = full.budget.always_active_backbone_elements;
    let unique = full.budget.unique_learned_elements;
    assert!(backbone < unique);

    let ceiling = backbone + 16;
    let options = AccountingOptions {
        ceiling,
        compute_digest: false,
        ..AccountingOptions::default()
    };
    let report = account_model(dir.path(), &options).unwrap();
    assert_eq!(report.budget.verdict, "binding_backbone_fits");
    assert!(report.budget.ceiling_binding);
    assert!(!report.budget.dense_execution_compliant);
    assert!(report.budget.backbone_fits);
    assert_eq!(report.budget.max_compliant_ffn_elements, 16);
    assert!((report.budget.max_compliant_ffn_fraction - 16.0 / 36.0).abs() < 1e-12);
}

#[test]
fn a_ceiling_below_the_backbone_is_impossible() {
    let dir = TempDir::new("accounting-impossible").unwrap();
    checkpoint(dir.path(), tied_tensors(), true);
    let options = AccountingOptions {
        ceiling: 10,
        compute_digest: false,
        ..AccountingOptions::default()
    };
    let report = account_model(dir.path(), &options).unwrap();
    assert_eq!(report.budget.verdict, "impossible_backbone_over_ceiling");
    assert!(!report.budget.backbone_fits);
    assert_eq!(report.budget.max_compliant_ffn_elements, 0);
    assert_eq!(report.budget.max_compliant_ffn_fraction, 0.0);
}

#[test]
fn tied_config_with_different_bytes_is_rejected() {
    let dir = TempDir::new("accounting-bad-tie").unwrap();
    let mut tensors = base_tensors();
    tensors.insert("lm_head.weight".to_owned(), tensor(&[VOCAB, HIDDEN], 40.0));
    checkpoint(dir.path(), tensors, true);
    let error = account_model(dir.path(), &AccountingOptions::default()).unwrap_err();
    assert!(error.to_string().contains("tie_word_embeddings=true"));
}

#[test]
fn unclassified_tensor_is_rejected() {
    let dir = TempDir::new("accounting-unexpected").unwrap();
    let mut tensors = tied_tensors();
    tensors.insert("model.extra.weight".to_owned(), tensor(&[1], 0.0));
    checkpoint(dir.path(), tensors, true);
    let error = account_model(dir.path(), &AccountingOptions::default()).unwrap_err();
    assert!(error.to_string().contains("does not classify"));
}

#[test]
fn a_shape_that_disagrees_with_config_is_rejected() {
    let dir = TempDir::new("accounting-shape").unwrap();
    let mut tensors = tied_tensors();
    tensors.insert(
        "model.layers.0.mlp.gate_proj.weight".to_owned(),
        tensor(&[INTERMEDIATE + 1, HIDDEN], 0.0),
    );
    checkpoint(dir.path(), tensors, true);
    let error = account_model(dir.path(), &AccountingOptions::default()).unwrap_err();
    assert!(error.to_string().contains("expected"));
}

#[test]
fn default_ceiling_matches_the_mandate() {
    assert_eq!(DEFAULT_ACTIVE_CEILING, 1_000_000_000);
}
