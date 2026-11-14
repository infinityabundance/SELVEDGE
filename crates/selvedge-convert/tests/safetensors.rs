//! Byte-level safetensors parser tests and property tests.

#![allow(clippy::unwrap_used)]

use std::path::Path;

use proptest::prelude::*;
use selvedge_convert::error::ConvertError;
use selvedge_convert::safetensors::{Safetensors, parse_bytes};
use selvedge_convert::synthetic::{
    SYNTHETIC_HIDDEN, SYNTHETIC_INTERMEDIATE, SYNTHETIC_SEED, SyntheticDType, encode_safetensors,
    swiglu_model,
};
use selvedge_convert::tensor_source::TensorSource;

fn valid_image() -> Vec<u8> {
    let model = swiglu_model(SYNTHETIC_HIDDEN, SYNTHETIC_INTERMEDIATE, SYNTHETIC_SEED).unwrap();
    encode_safetensors(&model, SyntheticDType::F32).unwrap()
}

fn header_only(header: &str) -> Vec<u8> {
    let padded = header.len().div_ceil(8) * 8;
    let mut out = Vec::new();
    out.extend_from_slice(&(padded as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.resize(8 + padded, b' ');
    out
}

fn with_data(header: &str, data: &[u8]) -> Vec<u8> {
    let mut out = header_only(header);
    out.extend_from_slice(data);
    out
}

#[test]
fn valid_image_parses_and_reads_exact_regions() {
    let image = valid_image();
    let mut store = Safetensors::from_source(image.as_slice()).unwrap();
    let keys = store.tensor_keys();
    assert_eq!(keys.len(), 3);
    for key in keys {
        let descriptor = store.descriptor(&key).unwrap();
        assert_eq!(descriptor.shape.len(), 2);
        let region = store
            .read_region_f32(&key, 0..descriptor.shape[0], 0..descriptor.shape[1])
            .unwrap();
        assert_eq!(region.len() as u64, descriptor.elements);
    }
}

#[test]
fn file_reader_uses_positional_reads_without_mapping() {
    let image = valid_image();
    let dir = selvedge_core::testutil::TempDir::new("selvedge-safetensors").unwrap();
    let path = dir.path().join("tiny.safetensors");
    std::fs::write(&path, &image).unwrap();
    let mut store = Safetensors::open(&path).unwrap();
    let key = "model.layers.0.mlp.gate_proj.weight".to_owned();
    let descriptor = store.descriptor(&key).unwrap();
    let rows = store.read_region_f32(&key, 1..3, 2..6).unwrap();
    assert_eq!(rows.len(), 2 * 4);
    let full = store
        .read_region_f32(&key, 0..descriptor.shape[0], 0..descriptor.shape[1])
        .unwrap();
    for row in 0..2 {
        for col in 0..4 {
            assert_eq!(
                rows[row * 4 + col],
                full[(row + 1) * descriptor.shape[1] as usize + col + 2]
            );
        }
    }
}

#[test]
fn zero_size_tensor_is_accepted() {
    let image = header_only(r#"{"empty":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}"#);
    let index = parse_bytes(&image).unwrap();
    assert_eq!(index.tensors["empty"].elements, 0);
    assert_eq!(index.data_len, 0);
}

#[test]
fn truncations_and_oversized_headers_are_rejected() {
    for cut in 0..8 {
        let image = &valid_image()[..cut];
        assert!(parse_bytes(image).is_err(), "cut {cut} was accepted");
    }
    let mut image = u64::MAX.to_le_bytes().to_vec();
    image.extend_from_slice(b"{}");
    assert!(matches!(
        parse_bytes(&image),
        Err(ConvertError::HeaderTooLarge { .. })
    ));
    let mut image = 9_u64.to_le_bytes().to_vec();
    image.extend_from_slice(&[b' '; 9]);
    assert!(matches!(
        parse_bytes(&image),
        Err(ConvertError::HeaderMisaligned { .. })
    ));
}

#[test]
fn malformed_json_and_non_utf8_are_rejected() {
    let bad_json = header_only("{not json}");
    assert!(matches!(
        parse_bytes(&bad_json),
        Err(ConvertError::HeaderMalformed { .. })
    ));
    let mut non_utf8 = 8_u64.to_le_bytes().to_vec();
    non_utf8.extend_from_slice(&[0xff; 8]);
    assert!(matches!(
        parse_bytes(&non_utf8),
        Err(ConvertError::HeaderNotUtf8 | ConvertError::HeaderMalformed { .. })
    ));
}

#[test]
fn duplicate_keys_are_rejected() {
    let image = header_only(
        r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
    );
    assert!(matches!(
        parse_bytes(&image),
        Err(ConvertError::HeaderMalformed { .. })
    ));
}

#[test]
fn unsupported_and_packed_dtypes_are_rejected() {
    for dtype in ["F64", "F4", "I4", "U4", "BOOL", "F8_E4M3", "F8_E5M2"] {
        let image = header_only(&format!(
            r#"{{"a":{{"dtype":"{dtype}","shape":[1],"data_offsets":[0,0]}}}}"#
        ));
        assert!(
            matches!(
                parse_bytes(&image),
                Err(ConvertError::UnsupportedDType { .. })
            ),
            "dtype {dtype} was not rejected"
        );
    }
}

#[test]
fn offset_shape_and_overlap_failures_are_rejected() {
    let reversed = with_data(
        r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}}"#,
        &[0_u8; 8],
    );
    assert!(matches!(
        parse_bytes(&reversed),
        Err(ConvertError::OffsetOutOfBounds { .. })
    ));

    let out_of_bounds = with_data(
        r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
        &[0_u8; 4],
    );
    assert!(matches!(
        parse_bytes(&out_of_bounds),
        Err(ConvertError::OffsetOutOfBounds { .. })
    ));

    let overlap = with_data(
        r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}}"#,
        &[0_u8; 16],
    );
    assert!(matches!(
        parse_bytes(&overlap),
        Err(ConvertError::OverlappingExtents { .. })
    ));

    let byte_mismatch = with_data(
        r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#,
        &[0_u8; 4],
    );
    assert!(matches!(
        parse_bytes(&byte_mismatch),
        Err(ConvertError::ByteLengthMismatch { .. })
    ));

    let absurd = header_only(
        r#"{"a":{"dtype":"F32","shape":[18446744073709551615,2],"data_offsets":[0,0]}}"#,
    );
    assert!(matches!(
        parse_bytes(&absurd),
        Err(ConvertError::InvalidShape { .. })
    ));
}

#[test]
fn unknown_tensor_fields_are_rejected() {
    let image =
        header_only(r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4],"nonsense":true}}"#);
    assert!(matches!(
        parse_bytes(&image),
        Err(ConvertError::HeaderMalformed { .. })
    ));
}

#[test]
fn metadata_must_be_a_string_map() {
    let image = header_only(r#"{"__metadata__":{"ok":"yes","bad":1}}"#);
    assert!(matches!(
        parse_bytes(&image),
        Err(ConvertError::HeaderMalformed { .. })
    ));
    let image = header_only(r#"{"__metadata__":{"format":"pt"}}"#);
    let index = parse_bytes(&image).unwrap();
    assert_eq!(index.metadata["format"], "pt");
    assert!(index.tensors.is_empty());
}

#[test]
fn read_tensor_bytes_honours_the_cap() {
    let image = valid_image();
    let store = Safetensors::from_source(image.as_slice()).unwrap();
    let key = "model.layers.0.mlp.gate_proj.weight";
    assert!(store.read_tensor_bytes(key, 4).is_err());
    let bytes = store.read_tensor_bytes(key, 1 << 20).unwrap();
    assert_eq!(
        bytes.len() as u64,
        store.header().tensors[key].end - store.header().tensors[key].begin
    );
}

proptest! {
    #[test]
    fn arbitrary_bytes_never_panic(bytes in proptest::collection::vec(any::<u8>(), 0..2048)) {
        let _ = parse_bytes(&bytes);
        if let Ok(mut store) = Safetensors::from_source(bytes.as_slice()) {
            for key in store.tensor_keys() {
                let _ = store.read_region_f32(&key, 0..1, 0..1);
            }
        }
    }

    #[test]
    fn single_byte_mutations_never_panic(
        mutations in proptest::collection::vec((any::<usize>(), any::<u8>()), 1..5)
    ) {
        let mut bytes = valid_image();
        for (index, value) in mutations {
            if bytes.is_empty() {
                break;
            }
            let slot = index % bytes.len();
            bytes[slot] = value;
        }
        let _ = parse_bytes(&bytes);
        if let Ok(mut store) = Safetensors::from_source(bytes.as_slice()) {
            for key in store.tensor_keys() {
                let _ = store.read_region_f32(&key, 0..1, 0..1);
            }
        }
    }
}

#[test]
fn writes_are_deterministic_across_runs() {
    let first = valid_image();
    let second = valid_image();
    assert_eq!(first, second);
    let dir = selvedge_core::testutil::TempDir::new("selvedge-safetensors-det").unwrap();
    let path = dir.path().join("a.safetensors");
    std::fs::write(&path, &first).unwrap();
    let store = Safetensors::open(Path::new(&path)).unwrap();
    assert_eq!(store.tensor_keys().len(), 3);
    assert!(store.header().metadata.contains_key("source"));
}
