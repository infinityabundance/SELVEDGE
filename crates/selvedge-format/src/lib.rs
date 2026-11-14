//! Versioned shard container format for SELVEDGE.
//!
//! A bundle is `magic || version || manifest_len || manifest JSON || padding ||
//! payload`. The manifest is self-describing, every shard id is recomputed from
//! its semantic fields during validation, and every payload extent is checked
//! for bounds, overlap, alignment and SHA-256. Publication is atomic within one
//! directory. See `docs/SHARD-FORMAT.md` for the byte layout, versioning rules
//! and threat model.
//!
//! Invariants:
//!
//! - validation never allocates from an untrusted length before checking it
//!   against a configured cap;
//! - a hash mismatch, malformed extent, duplicate id or version skew is a typed
//!   error, never a warning;
//! - a published bundle is either absent or complete: the writer fsyncs the
//!   complete temporary file, validates it, then links/renames it into place.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod catalog;
pub mod error;
pub mod manifest;
pub mod memory;
pub mod reader;
pub mod sidecar;
pub mod validation;
pub mod writer;

pub use catalog::{BundleCatalog, CatalogEntry, ReloadVerdict};
pub use error::{FormatError, Result};
pub use manifest::{
    AuxExtent, CalibrationProvenance, DEFAULT_ALIGNMENT, HARD_MAX_MANIFEST_BYTES, HEADER_LEN,
    MAGIC, MANIFEST_SCHEMA, MANIFEST_SCHEMA_VERSION, MAX_SHARDS, MIN_ALIGNMENT, Manifest,
    QuantizationSpec, ShardRecord,
};
pub use memory::MemoryBundle;
pub use reader::BundleFile;
pub use sidecar::{SIDECAR_SCHEMA, Sidecar, publish_sidecar, read_sidecar, validate_sidecar};
pub use validation::{
    BundleHeader, ValidationOptions, ValidationReport, VerifyMode, validate_bundle, validate_bytes,
    validate_reader,
};
pub use writer::{
    PublishedBundle, WriteOptions, encode_bundle_bytes, publish_bundle, publish_bundle_from_reader,
};
