//! The real-activation tolerance policy must be documented, not just coded.
//!
//! `docs/ARCHITECTURE.md` publishes the tolerance constants from the
//! real-parity receipt. This test fails if the code and the document drift
//! apart, so the named Court A scenario
//! `bf16_roundtrip_tolerance_is_documented` has a mechanical backstop.

#![allow(clippy::unwrap_used)]

use std::path::PathBuf;

use selvedge_convert::{
    BF16_PARITY_ABS, BF16_PARITY_REL, BF16_PARITY_SCALE, F32_PARITY_ABS, F32_PARITY_REL,
    F32_PARITY_SCALE, TOLERANCE_POLICY,
};

fn architecture_doc() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("docs/ARCHITECTURE.md");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()))
}

fn rendered(value: f64) -> String {
    format!("{value}")
}

#[test]
fn architecture_documents_every_tolerance_constant() {
    let doc = architecture_doc();
    for value in [
        F32_PARITY_ABS,
        F32_PARITY_REL,
        F32_PARITY_SCALE,
        BF16_PARITY_ABS,
        BF16_PARITY_REL,
        BF16_PARITY_SCALE,
    ] {
        let needle = rendered(value);
        assert!(
            doc.contains(&needle),
            "docs/ARCHITECTURE.md does not contain the tolerance value {needle}"
        );
    }
    assert!(doc.contains("cancellation"));
    assert!(TOLERANCE_POLICY.contains("docs/ARCHITECTURE.md"));
}
