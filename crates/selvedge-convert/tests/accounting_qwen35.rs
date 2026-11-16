//! Offline tests for Qwen3.5 accounting (schema v2): tower separation, the
//! declared execution path, tie verification by absence and by bytes, and every
//! refusal the auditor promises.
//!
//! The tiny checkpoint here is shaped by the real config's rules (nested
//! `text_config`, `layer_types`, a vision tower, an MTP head) but with
//! single-digit dimensions. The real 0.8B checkpoint is exercised by the rust
//! court receipt, which is the evidence for the 0.8B numbers.

#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::path::Path;

use selvedge_convert::accounting_qwen35::{
    ActivePath, Qwen35Options, TieAssertion, account_model_qwen35,
};
use selvedge_convert::detect_model_type;
use selvedge_convert::synthetic::{
    SyntheticDType, SyntheticModel, SyntheticTensor, write_safetensors,
};
use selvedge_core::testutil::TempDir;

const HIDDEN: u64 = 4;
const INTERMEDIATE: u64 = 3;
const VOCAB: u64 = 5;
const LAYERS: u64 = 2;
const ATTENTION_HEADS: u64 = 2;
const ATTENTION_KV_HEADS: u64 = 1;
const ATTENTION_HEAD_DIM: u64 = 2;
const Q_OUT: u64 = ATTENTION_HEADS * ATTENTION_HEAD_DIM * 2; // output gate fused
const KV_OUT: u64 = ATTENTION_KV_HEADS * ATTENTION_HEAD_DIM;
const LIN_KEY_HEADS: u64 = 2;
const LIN_KEY_DIM: u64 = 1;
const LIN_VALUE_HEADS: u64 = 2;
const LIN_VALUE_DIM: u64 = 1;
const CONV_DIM: u64 = 2 * LIN_KEY_HEADS * LIN_KEY_DIM + LIN_VALUE_HEADS * LIN_VALUE_DIM;
const VALUE_WIDTH: u64 = LIN_VALUE_HEADS * LIN_VALUE_DIM;
const CONV_KERNEL: u64 = 2;
const MTP_LAYERS: u64 = 1;
const VISION_HIDDEN: u64 = 2;
const VISION_INTERMEDIATE: u64 = 3;
const VISION_OUT_HIDDEN: u64 = 4;
const VISION_DEPTH: u64 = 1;
const VISION_POSITIONS: u64 = 2;

fn tensor(shape: &[u64], seed: f32) -> SyntheticTensor {
    let count: u64 = shape.iter().product();
    SyntheticTensor {
        shape: shape.to_vec(),
        values: (0..count).map(|index| seed + index as f32 * 0.25).collect(),
    }
}

fn config_json(tie: bool, layer_types: [&str; 2], mtp_layers: u64, vision: bool) -> String {
    let vision_json = if vision {
        r#",
  "vision_config": {
    "depth": 1,
    "hidden_act": "gelu_pytorch_tanh",
    "hidden_size": 2,
    "in_channels": 1,
    "intermediate_size": 3,
    "num_heads": 1,
    "num_position_embeddings": 2,
    "out_hidden_size": 4,
    "patch_size": 1,
    "spatial_merge_size": 1,
    "temporal_patch_size": 1
  }"#
        .to_owned()
    } else {
        String::new()
    };
    format!(
        r#"{{
  "architectures": ["Qwen3_5ForConditionalGeneration"],
  "model_type": "qwen3_5",
  "tie_word_embeddings": {tie},
  "text_config": {{
    "attn_output_gate": true,
    "full_attention_interval": 2,
    "head_dim": {ATTENTION_HEAD_DIM},
    "hidden_size": {HIDDEN},
    "intermediate_size": {INTERMEDIATE},
    "layer_types": ["{lt0}", "{lt1}"],
    "linear_conv_kernel_dim": {CONV_KERNEL},
    "linear_key_head_dim": {LIN_KEY_DIM},
    "linear_num_key_heads": {LIN_KEY_HEADS},
    "linear_num_value_heads": {LIN_VALUE_HEADS},
    "linear_value_head_dim": {LIN_VALUE_DIM},
    "mlp_only_layers": [],
    "model_type": "qwen3_5_text",
    "mtp_num_hidden_layers": {mtp_layers},
    "mtp_use_dedicated_embeddings": false,
    "num_attention_heads": {ATTENTION_HEADS},
    "num_hidden_layers": {LAYERS},
    "num_key_value_heads": {ATTENTION_KV_HEADS},
    "rms_norm_eps": 1e-06,
    "rope_parameters": {{
      "mrope_interleaved": true,
      "mrope_section": [1, 1],
      "rope_theta": 10000000.0,
      "rope_type": "default",
      "partial_rotary_factor": 1.0
    }},
    "tie_word_embeddings": {tie},
    "vocab_size": {VOCAB}
  }}{vision_json}
}}
"#,
        lt0 = layer_types[0],
        lt1 = layer_types[1],
    )
}

fn text_tensors() -> BTreeMap<String, SyntheticTensor> {
    let mut tensors = BTreeMap::new();
    let mut seed = 0.0_f32;
    let mut push =
        |tensors: &mut BTreeMap<String, SyntheticTensor>, key: String, shape: Vec<u64>| {
            tensors.insert(key, tensor(&shape, seed));
            seed += 1.0;
        };
    push(
        &mut tensors,
        "model.language_model.embed_tokens.weight".to_owned(),
        vec![VOCAB, HIDDEN],
    );
    for layer in 0..LAYERS {
        let prefix = format!("model.language_model.layers.{layer}");
        push(
            &mut tensors,
            format!("{prefix}.input_layernorm.weight"),
            vec![HIDDEN],
        );
        push(
            &mut tensors,
            format!("{prefix}.post_attention_layernorm.weight"),
            vec![HIDDEN],
        );
        if layer == 1 {
            push(
                &mut tensors,
                format!("{prefix}.self_attn.q_proj.weight"),
                vec![Q_OUT, HIDDEN],
            );
            push(
                &mut tensors,
                format!("{prefix}.self_attn.k_proj.weight"),
                vec![KV_OUT, HIDDEN],
            );
            push(
                &mut tensors,
                format!("{prefix}.self_attn.v_proj.weight"),
                vec![KV_OUT, HIDDEN],
            );
            push(
                &mut tensors,
                format!("{prefix}.self_attn.o_proj.weight"),
                vec![HIDDEN, ATTENTION_HEADS * ATTENTION_HEAD_DIM],
            );
            push(
                &mut tensors,
                format!("{prefix}.self_attn.q_norm.weight"),
                vec![ATTENTION_HEAD_DIM],
            );
            push(
                &mut tensors,
                format!("{prefix}.self_attn.k_norm.weight"),
                vec![ATTENTION_HEAD_DIM],
            );
        } else {
            push(
                &mut tensors,
                format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                vec![CONV_DIM, HIDDEN],
            );
            push(
                &mut tensors,
                format!("{prefix}.linear_attn.in_proj_z.weight"),
                vec![VALUE_WIDTH, HIDDEN],
            );
            push(
                &mut tensors,
                format!("{prefix}.linear_attn.in_proj_a.weight"),
                vec![LIN_VALUE_HEADS, HIDDEN],
            );
            push(
                &mut tensors,
                format!("{prefix}.linear_attn.in_proj_b.weight"),
                vec![LIN_VALUE_HEADS, HIDDEN],
            );
            push(
                &mut tensors,
                format!("{prefix}.linear_attn.conv1d.weight"),
                vec![CONV_DIM, 1, CONV_KERNEL],
            );
            push(
                &mut tensors,
                format!("{prefix}.linear_attn.A_log"),
                vec![LIN_VALUE_HEADS],
            );
            push(
                &mut tensors,
                format!("{prefix}.linear_attn.dt_bias"),
                vec![LIN_VALUE_HEADS],
            );
            push(
                &mut tensors,
                format!("{prefix}.linear_attn.norm.weight"),
                vec![LIN_VALUE_DIM],
            );
            push(
                &mut tensors,
                format!("{prefix}.linear_attn.out_proj.weight"),
                vec![HIDDEN, VALUE_WIDTH],
            );
        }
        push(
            &mut tensors,
            format!("{prefix}.mlp.gate_proj.weight"),
            vec![INTERMEDIATE, HIDDEN],
        );
        push(
            &mut tensors,
            format!("{prefix}.mlp.up_proj.weight"),
            vec![INTERMEDIATE, HIDDEN],
        );
        push(
            &mut tensors,
            format!("{prefix}.mlp.down_proj.weight"),
            vec![HIDDEN, INTERMEDIATE],
        );
    }
    push(
        &mut tensors,
        "model.language_model.norm.weight".to_owned(),
        vec![HIDDEN],
    );
    tensors
}

fn mtp_tensors() -> BTreeMap<String, SyntheticTensor> {
    let mut tensors = BTreeMap::new();
    let mut seed = 100.0_f32;
    let mut push =
        |tensors: &mut BTreeMap<String, SyntheticTensor>, key: String, shape: Vec<u64>| {
            tensors.insert(key, tensor(&shape, seed));
            seed += 1.0;
        };
    push(
        &mut tensors,
        "mtp.fc.weight".to_owned(),
        vec![HIDDEN, 2 * HIDDEN],
    );
    push(
        &mut tensors,
        "mtp.pre_fc_norm_embedding.weight".to_owned(),
        vec![HIDDEN],
    );
    push(
        &mut tensors,
        "mtp.pre_fc_norm_hidden.weight".to_owned(),
        vec![HIDDEN],
    );
    push(&mut tensors, "mtp.norm.weight".to_owned(), vec![HIDDEN]);
    for layer in 0..MTP_LAYERS {
        let prefix = format!("mtp.layers.{layer}");
        push(
            &mut tensors,
            format!("{prefix}.input_layernorm.weight"),
            vec![HIDDEN],
        );
        push(
            &mut tensors,
            format!("{prefix}.post_attention_layernorm.weight"),
            vec![HIDDEN],
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.q_proj.weight"),
            vec![Q_OUT, HIDDEN],
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.k_proj.weight"),
            vec![KV_OUT, HIDDEN],
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.v_proj.weight"),
            vec![KV_OUT, HIDDEN],
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.o_proj.weight"),
            vec![HIDDEN, ATTENTION_HEADS * ATTENTION_HEAD_DIM],
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.q_norm.weight"),
            vec![ATTENTION_HEAD_DIM],
        );
        push(
            &mut tensors,
            format!("{prefix}.self_attn.k_norm.weight"),
            vec![ATTENTION_HEAD_DIM],
        );
        push(
            &mut tensors,
            format!("{prefix}.mlp.gate_proj.weight"),
            vec![INTERMEDIATE, HIDDEN],
        );
        push(
            &mut tensors,
            format!("{prefix}.mlp.up_proj.weight"),
            vec![INTERMEDIATE, HIDDEN],
        );
        push(
            &mut tensors,
            format!("{prefix}.mlp.down_proj.weight"),
            vec![HIDDEN, INTERMEDIATE],
        );
    }
    tensors
}

fn vision_tensors() -> BTreeMap<String, SyntheticTensor> {
    let mut tensors = BTreeMap::new();
    let mut seed = 200.0_f32;
    let mut push =
        |tensors: &mut BTreeMap<String, SyntheticTensor>, key: String, shape: Vec<u64>| {
            tensors.insert(key, tensor(&shape, seed));
            seed += 1.0;
        };
    push(
        &mut tensors,
        "model.visual.patch_embed.proj.weight".to_owned(),
        vec![VISION_HIDDEN, 1, 1, 1, 1],
    );
    push(
        &mut tensors,
        "model.visual.patch_embed.proj.bias".to_owned(),
        vec![VISION_HIDDEN],
    );
    push(
        &mut tensors,
        "model.visual.pos_embed.weight".to_owned(),
        vec![VISION_POSITIONS, VISION_HIDDEN],
    );
    for block in 0..VISION_DEPTH {
        let prefix = format!("model.visual.blocks.{block}");
        for name in ["norm1", "norm2"] {
            push(
                &mut tensors,
                format!("{prefix}.{name}.weight"),
                vec![VISION_HIDDEN],
            );
            push(
                &mut tensors,
                format!("{prefix}.{name}.bias"),
                vec![VISION_HIDDEN],
            );
        }
        push(
            &mut tensors,
            format!("{prefix}.attn.qkv.weight"),
            vec![3 * VISION_HIDDEN, VISION_HIDDEN],
        );
        push(
            &mut tensors,
            format!("{prefix}.attn.qkv.bias"),
            vec![3 * VISION_HIDDEN],
        );
        push(
            &mut tensors,
            format!("{prefix}.attn.proj.weight"),
            vec![VISION_HIDDEN, VISION_HIDDEN],
        );
        push(
            &mut tensors,
            format!("{prefix}.attn.proj.bias"),
            vec![VISION_HIDDEN],
        );
        push(
            &mut tensors,
            format!("{prefix}.mlp.linear_fc1.weight"),
            vec![VISION_INTERMEDIATE, VISION_HIDDEN],
        );
        push(
            &mut tensors,
            format!("{prefix}.mlp.linear_fc1.bias"),
            vec![VISION_INTERMEDIATE],
        );
        push(
            &mut tensors,
            format!("{prefix}.mlp.linear_fc2.weight"),
            vec![VISION_HIDDEN, VISION_INTERMEDIATE],
        );
        push(
            &mut tensors,
            format!("{prefix}.mlp.linear_fc2.bias"),
            vec![VISION_HIDDEN],
        );
    }
    push(
        &mut tensors,
        "model.visual.merger.norm.weight".to_owned(),
        vec![VISION_HIDDEN],
    );
    push(
        &mut tensors,
        "model.visual.merger.norm.bias".to_owned(),
        vec![VISION_HIDDEN],
    );
    push(
        &mut tensors,
        "model.visual.merger.linear_fc1.weight".to_owned(),
        vec![VISION_HIDDEN, VISION_HIDDEN],
    );
    push(
        &mut tensors,
        "model.visual.merger.linear_fc1.bias".to_owned(),
        vec![VISION_HIDDEN],
    );
    push(
        &mut tensors,
        "model.visual.merger.linear_fc2.weight".to_owned(),
        vec![VISION_OUT_HIDDEN, VISION_HIDDEN],
    );
    push(
        &mut tensors,
        "model.visual.merger.linear_fc2.bias".to_owned(),
        vec![VISION_OUT_HIDDEN],
    );
    tensors
}

fn all_tensors() -> BTreeMap<String, SyntheticTensor> {
    let mut tensors = text_tensors();
    tensors.extend(mtp_tensors());
    tensors.extend(vision_tensors());
    tensors
}

fn write_checkpoint(
    dir: &Path,
    tensors: BTreeMap<String, SyntheticTensor>,
    tie: bool,
    layer_types: [&str; 2],
    mtp_layers: u64,
    vision: bool,
) {
    std::fs::write(
        dir.join("config.json"),
        config_json(tie, layer_types, mtp_layers, vision),
    )
    .unwrap();
    write_safetensors(
        &SyntheticModel {
            hidden: HIDDEN,
            intermediate: INTERMEDIATE,
            seed: 0x5133_3500,
            tensors,
            maps: Vec::new(),
        },
        SyntheticDType::F32,
        &dir.join("model.safetensors"),
    )
    .unwrap();
}

/// Elements the config rules imply, restated independently of the auditor.
fn expected_totals() -> (u64, u64, u64, u64) {
    let embed = VOCAB * HIDDEN;
    let mut text = embed + HIDDEN; // final norm
    // Layer 0: linear attention + FFN.
    let linear = CONV_DIM * HIDDEN
        + VALUE_WIDTH * HIDDEN
        + 2 * LIN_VALUE_HEADS * HIDDEN
        + CONV_DIM * CONV_KERNEL
        + 2 * LIN_VALUE_HEADS
        + LIN_VALUE_DIM
        + HIDDEN * VALUE_WIDTH;
    let ffn = 3 * INTERMEDIATE * HIDDEN;
    text += 2 * HIDDEN + linear + ffn;
    // Layer 1: gated attention + FFN.
    let attention = Q_OUT * HIDDEN
        + 2 * KV_OUT * HIDDEN
        + HIDDEN * (ATTENTION_HEADS * ATTENTION_HEAD_DIM)
        + 2 * ATTENTION_HEAD_DIM;
    text += 2 * HIDDEN + attention + ffn;

    let mtp_attention = Q_OUT * HIDDEN
        + 2 * KV_OUT * HIDDEN
        + HIDDEN * (ATTENTION_HEADS * ATTENTION_HEAD_DIM)
        + 2 * ATTENTION_HEAD_DIM;
    let mtp = HIDDEN * 2 * HIDDEN + 3 * HIDDEN + 2 * HIDDEN + mtp_attention + ffn;

    let vision_blocks = VISION_DEPTH
        * (4 * VISION_HIDDEN
            + 3 * VISION_HIDDEN * VISION_HIDDEN
            + 3 * VISION_HIDDEN
            + VISION_HIDDEN * VISION_HIDDEN
            + VISION_HIDDEN
            + VISION_INTERMEDIATE * VISION_HIDDEN
            + VISION_INTERMEDIATE
            + VISION_HIDDEN * VISION_INTERMEDIATE
            + VISION_HIDDEN);
    let vision = VISION_HIDDEN
        + VISION_HIDDEN
        + VISION_POSITIONS * VISION_HIDDEN
        + vision_blocks
        + 2 * VISION_HIDDEN
        + VISION_HIDDEN * VISION_HIDDEN
        + VISION_HIDDEN
        + VISION_OUT_HIDDEN * VISION_HIDDEN
        + VISION_OUT_HIDDEN;

    (text, vision, mtp, text + vision + mtp)
}

const LINEAR_THEN_FULL: [&str; 2] = ["linear_attention", "full_attention"];

#[test]
fn tiny_qwen35_separates_towers_and_verifies_the_tie_by_absence() {
    let dir = TempDir::new("qwen35-towers").unwrap();
    let (text, vision, mtp, total) = expected_totals();
    write_checkpoint(
        dir.path(),
        all_tensors(),
        true,
        LINEAR_THEN_FULL,
        MTP_LAYERS,
        true,
    );
    let report = account_model_qwen35(dir.path(), &Qwen35Options::default()).unwrap();

    assert!(report.ok, "refusals: {:?}", report.refusals);
    assert_eq!(report.schema, "selvedge/model-accounting/v2");
    assert_eq!(report.model.text.layers, LAYERS);
    assert_eq!(report.model.text.attention_layers, 1);
    assert_eq!(report.model.text.linear_layers, 1);
    assert_eq!(report.model.text.attention_layer_indices, vec![1]);
    assert_eq!(report.model.text.rotary_dim, ATTENTION_HEAD_DIM);
    assert_eq!(report.model.text.conv_kernel, CONV_KERNEL);

    // Tie by absence: config says tied and there is no lm_head.weight.
    assert!(!report.tie.lm_head_present);
    assert!(report.tie.treated_as_shared);
    assert_eq!(report.tie.compared_elements, 0);

    assert_eq!(report.towers["text"].checkpoint_elements, text);
    assert_eq!(report.towers["text"].unique_elements, text);
    assert_eq!(report.towers["vision"].checkpoint_elements, vision);
    assert_eq!(report.towers["mtp"].checkpoint_elements, mtp);
    assert_eq!(report.checkpoint.tensor_elements, total);
    assert_eq!(report.checkpoint.unique_elements, total);
    assert_eq!(report.checkpoint.tensor_bytes, total * 4);
    // Text FFN is the removable set: two layers of gate+up+down.
    assert_eq!(
        report.towers["text"].ffn_elements,
        2 * 3 * INTERMEDIATE * HIDDEN
    );
}

#[test]
fn text_only_excludes_the_vision_tower_and_the_mtp_head() {
    let dir = TempDir::new("qwen35-text-only").unwrap();
    let (text, vision, mtp, total) = expected_totals();
    write_checkpoint(
        dir.path(),
        all_tensors(),
        true,
        LINEAR_THEN_FULL,
        MTP_LAYERS,
        true,
    );
    let report = account_model_qwen35(dir.path(), &Qwen35Options::default()).unwrap();

    let path = &report.active_path;
    assert_eq!(path.declared, "text-only");
    assert_eq!(path.executed_towers, vec!["text"]);
    assert_eq!(path.excluded_towers, vec!["vision", "mtp"]);
    assert_eq!(path.unique_active_elements, text);
    assert_eq!(path.verdict, "consistent");
    let excluded: u64 = path
        .excluded_elements
        .iter()
        .map(|entry| entry.elements)
        .sum();
    assert_eq!(excluded, vision + mtp);
    assert_eq!(path.unique_active_elements + excluded, total);
    // The MTP head reads the language model's embedding.
    assert_eq!(path.shared_reads.len(), 1);
    assert_eq!(
        path.shared_reads[0].tensor,
        "model.language_model.embed_tokens.weight"
    );
    // The certificate is produced even when not explicitly requested.
    assert_eq!(report.certification.ceiling, 1_000_000_000);
    assert_eq!(report.certification.verdict, "certified");
    assert_eq!(
        report.certification.active_unique_elements,
        path.unique_active_elements
    );
}

#[test]
fn mtp_included_adds_the_head_once_and_certifies_the_union() {
    let dir = TempDir::new("qwen35-mtp").unwrap();
    let (text, _vision, mtp, _total) = expected_totals();
    write_checkpoint(
        dir.path(),
        all_tensors(),
        true,
        LINEAR_THEN_FULL,
        MTP_LAYERS,
        true,
    );
    let options = Qwen35Options {
        active_path: ActivePath::TextAndMtp,
        certify_ceiling: Some(text + mtp),
        ..Qwen35Options::default()
    };
    let report = account_model_qwen35(dir.path(), &options).unwrap();
    assert!(report.ok, "refusals: {:?}", report.refusals);
    assert_eq!(report.active_path.executed_towers, vec!["text", "mtp"]);
    assert_eq!(report.active_path.excluded_towers, vec!["vision"]);
    assert_eq!(report.active_path.unique_active_elements, text + mtp);
    assert_eq!(report.certification.verdict, "certified");
    assert_eq!(report.certification.headroom_elements, 0);
    // The arithmetic must re-add to the reported active count.
    let plus: u64 = report
        .certification
        .arithmetic
        .iter()
        .filter(|term| term.sign == "+")
        .map(|term| term.elements)
        .sum();
    assert_eq!(plus, text + mtp);
}

#[test]
fn a_certificate_above_the_ceiling_fails_closed() {
    let dir = TempDir::new("qwen35-ceiling").unwrap();
    let (text, _vision, _mtp, _total) = expected_totals();
    write_checkpoint(
        dir.path(),
        all_tensors(),
        true,
        LINEAR_THEN_FULL,
        MTP_LAYERS,
        true,
    );
    let options = Qwen35Options {
        certify_ceiling: Some(text - 1),
        ..Qwen35Options::default()
    };
    let report = account_model_qwen35(dir.path(), &options).unwrap();
    assert!(!report.ok);
    assert_eq!(report.certification.verdict, "exceeds_ceiling");
    assert_eq!(report.certification.headroom_elements, -1);
    assert!(
        report
            .active_path
            .excluded_elements
            .iter()
            .all(|entry| entry.elements > 0)
    );
    assert!(
        report
            .refusals
            .iter()
            .any(|refusal| refusal.contains("above the"))
    );
}

#[test]
fn declaring_a_tie_against_a_distinct_lm_head_is_refused() {
    let dir = TempDir::new("qwen35-tie-refused").unwrap();
    let mut tensors = all_tensors();
    // An untied checkpoint: lm_head.weight present with its own bytes.
    tensors.insert("lm_head.weight".to_owned(), tensor(&[VOCAB, HIDDEN], 900.0));
    write_checkpoint(
        dir.path(),
        tensors,
        false,
        LINEAR_THEN_FULL,
        MTP_LAYERS,
        true,
    );
    // The config is untied and the bytes agree, so a plain account succeeds and
    // counts both matrices.
    let report = account_model_qwen35(dir.path(), &Qwen35Options::default()).unwrap();
    assert!(report.ok, "refusals: {:?}", report.refusals);
    assert!(report.tie.lm_head_present);
    assert!(!report.tie.lm_head_identical_to_embedding);
    assert!(!report.tie.treated_as_shared);
    assert_eq!(report.tie.compared_elements, VOCAB * HIDDEN);

    // The same bytes with a declared tie are refused, and the arithmetic is
    // still reported.
    let options = Qwen35Options {
        tie_assertion: TieAssertion::Tied,
        ..Qwen35Options::default()
    };
    let refused = account_model_qwen35(dir.path(), &options).unwrap();
    assert!(!refused.ok);
    assert_eq!(refused.active_path.verdict, "inconsistent");
    assert!(
        refused
            .refusals
            .iter()
            .any(|refusal| refusal.contains("declared the weights tied"))
    );
    assert_eq!(
        refused.active_path.unique_active_elements,
        expected_totals().0 + VOCAB * HIDDEN
    );
}

#[test]
fn declaring_untied_without_an_lm_head_is_refused() {
    let dir = TempDir::new("qwen35-untied-refused").unwrap();
    write_checkpoint(
        dir.path(),
        all_tensors(),
        true,
        LINEAR_THEN_FULL,
        MTP_LAYERS,
        true,
    );
    let options = Qwen35Options {
        tie_assertion: TieAssertion::Untied,
        ..Qwen35Options::default()
    };
    let report = account_model_qwen35(dir.path(), &options).unwrap();
    assert!(!report.ok);
    assert!(
        report
            .refusals
            .iter()
            .any(|refusal| refusal.contains("no lm_head.weight exists"))
    );
}

#[test]
fn declaring_mtp_on_a_checkpoint_without_mtp_is_refused() {
    let dir = TempDir::new("qwen35-mtp-absent").unwrap();
    let mut tensors = text_tensors();
    tensors.extend(vision_tensors());
    // mtp_num_hidden_layers = 0 and no mtp tensors.
    write_checkpoint(dir.path(), tensors, true, LINEAR_THEN_FULL, 0, true);
    assert!(!report_has_mtp(dir.path()));

    let options = Qwen35Options {
        active_path: ActivePath::TextAndMtp,
        ..Qwen35Options::default()
    };
    let report = account_model_qwen35(dir.path(), &options).unwrap();
    assert!(!report.ok);
    assert_eq!(report.active_path.verdict, "inconsistent");
    assert!(
        report
            .refusals
            .iter()
            .any(|refusal| refusal.contains("no such weights")
                || refusal.contains("contains no mtp weights"))
    );
}

fn report_has_mtp(path: &Path) -> bool {
    let report = account_model_qwen35(path, &Qwen35Options::default()).unwrap();
    report.towers.contains_key("mtp")
}

#[test]
fn a_vision_tensor_without_a_vision_config_is_unclassified() {
    let dir = TempDir::new("qwen35-vision-orphan").unwrap();
    write_checkpoint(
        dir.path(),
        all_tensors(),
        true,
        LINEAR_THEN_FULL,
        MTP_LAYERS,
        false, // config declares no vision tower
    );
    let error = account_model_qwen35(dir.path(), &Qwen35Options::default()).unwrap_err();
    assert!(
        error.to_string().contains("does not classify"),
        "unexpected error: {error}"
    );
}

#[test]
fn a_hub_total_that_disagrees_with_the_header_sum_is_refused() {
    let dir = TempDir::new("qwen35-hub-total").unwrap();
    write_checkpoint(
        dir.path(),
        all_tensors(),
        true,
        LINEAR_THEN_FULL,
        MTP_LAYERS,
        true,
    );
    let (_text, _vision, _mtp, total) = expected_totals();
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        format!(
            "{{\"metadata\":{{\"total_size\":{}}},\"weight_map\":{{}}}}",
            total * 4 + 8
        ),
    )
    .unwrap();
    let error = account_model_qwen35(dir.path(), &Qwen35Options::default()).unwrap_err();
    assert!(
        error.to_string().contains("Hub") || error.to_string().contains("total bytes"),
        "unexpected error: {error}"
    );

    // With the true byte total the auditor accepts and records the match.
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        format!(
            "{{\"metadata\":{{\"total_size\":{}}},\"weight_map\":{{}}}}",
            total * 4
        ),
    )
    .unwrap();
    let report = account_model_qwen35(dir.path(), &Qwen35Options::default()).unwrap();
    assert_eq!(report.checkpoint.index_total_bytes, Some(total * 4));
    assert_eq!(report.checkpoint.index_total_matches, Some(true));
}

#[test]
fn layer_types_must_agree_with_the_attention_interval() {
    let dir = TempDir::new("qwen35-layer-types").unwrap();
    // Interval 2 implies layer 1 is full attention; declaring both linear is a
    // contradiction and must not be silently resolved either way.
    write_checkpoint(
        dir.path(),
        all_tensors(),
        true,
        ["linear_attention", "linear_attention"],
        MTP_LAYERS,
        true,
    );
    let error = account_model_qwen35(dir.path(), &Qwen35Options::default()).unwrap_err();
    assert!(
        error.to_string().contains("full_attention_interval"),
        "unexpected error: {error}"
    );
}

#[test]
fn expected_totals_must_match_exactly() {
    let dir = TempDir::new("qwen35-expect").unwrap();
    let (text, vision, mtp, total) = expected_totals();
    write_checkpoint(
        dir.path(),
        all_tensors(),
        true,
        LINEAR_THEN_FULL,
        MTP_LAYERS,
        true,
    );
    let options = Qwen35Options {
        expect_totals: vec![
            ("text".to_owned(), text),
            ("vision".to_owned(), vision),
            ("mtp".to_owned(), mtp),
            ("checkpoint".to_owned(), total),
        ],
        ..Qwen35Options::default()
    };
    let report = account_model_qwen35(dir.path(), &options).unwrap();
    assert!(report.ok, "refusals: {:?}", report.refusals);
    assert!(report.expectations.iter().all(|check| check.matches));

    let wrong = Qwen35Options {
        expect_totals: vec![("checkpoint".to_owned(), total + 1)],
        ..Qwen35Options::default()
    };
    let refused = account_model_qwen35(dir.path(), &wrong).unwrap();
    assert!(!refused.ok);
    assert_eq!(refused.expectations[0].observed, total);
}

#[test]
fn model_type_detection_reads_the_nested_config() {
    let dir = TempDir::new("qwen35-detect").unwrap();
    write_checkpoint(
        dir.path(),
        all_tensors(),
        true,
        LINEAR_THEN_FULL,
        MTP_LAYERS,
        true,
    );
    assert_eq!(detect_model_type(dir.path()).unwrap(), "qwen3_5");
}

#[test]
fn an_unknown_active_path_is_rejected() {
    assert!(ActivePath::parse("text-only").is_ok());
    assert!(ActivePath::parse("text-and-mtp").is_ok());
    assert!(ActivePath::parse("vision").is_err());
    assert!(ActivePath::parse("").is_err());
}

#[test]
fn dedicated_mtp_embeddings_are_refused_rather_than_guessed() {
    let dir = TempDir::new("qwen35-mtp-dedicated").unwrap();
    write_checkpoint(
        dir.path(),
        all_tensors(),
        true,
        LINEAR_THEN_FULL,
        MTP_LAYERS,
        true,
    );
    let config = std::fs::read_to_string(dir.path().join("config.json")).unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        config.replace(
            "\"mtp_use_dedicated_embeddings\": false",
            "\"mtp_use_dedicated_embeddings\": true",
        ),
    )
    .unwrap();
    let error = account_model_qwen35(dir.path(), &Qwen35Options::default()).unwrap_err();
    assert!(
        error.to_string().contains("mtp_use_dedicated_embeddings"),
        "unexpected error: {error}"
    );
}
