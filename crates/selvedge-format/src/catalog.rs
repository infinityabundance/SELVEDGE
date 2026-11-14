//! Bundle catalog and hot-reload conflict detection.
//!
//! A catalog tracks the manifest digest of each published bundle id. Offering
//! the same id with different content is rejected and does not mutate the
//! catalog; offering identical content is idempotent.

use std::collections::BTreeMap;

use selvedge_core::{Sha256Digest, canonical_json_sha256};
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::manifest::Manifest;

/// Catalog entry for one published bundle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogEntry {
    /// Bundle id.
    pub bundle_id: String,
    /// SHA-256 of the canonical manifest JSON.
    pub manifest_sha256: Sha256Digest,
    /// Model digest.
    pub model_digest: Sha256Digest,
    /// Payload length.
    pub payload_len: u64,
    /// Number of shard records.
    pub shard_count: u64,
    /// Sorted unique compatibility ids.
    pub compatibility_ids: Vec<String>,
}

impl CatalogEntry {
    /// Build an entry from a manifest.
    ///
    /// # Errors
    ///
    /// Rejects an invalid manifest or canonicalization failure.
    pub fn from_manifest(manifest: &Manifest) -> Result<Self> {
        manifest.validate()?;
        Ok(Self {
            bundle_id: manifest.bundle_id.clone(),
            manifest_sha256: canonical_json_sha256(manifest)?,
            model_digest: manifest.model.digest(),
            payload_len: manifest.payload_len,
            shard_count: manifest.shards.len() as u64,
            compatibility_ids: manifest.compatibility_ids(),
        })
    }
}

/// Result of offering a manifest to the catalog.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReloadVerdict {
    /// A new bundle id was accepted.
    AcceptedNew,
    /// The identical manifest is already known; no change.
    AcceptedIdentical,
    /// The bundle id exists with different content; the offer was rejected.
    RejectedConflict {
        /// Entry already installed.
        existing: Box<CatalogEntry>,
        /// Entry that was refused.
        incoming: Box<CatalogEntry>,
    },
}

impl ReloadVerdict {
    /// Whether the catalog accepted the offer.
    #[must_use]
    pub fn is_accepted(&self) -> bool {
        matches!(self, Self::AcceptedNew | Self::AcceptedIdentical)
    }
}

/// In-memory catalog of bundle ids and their manifest digests.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct BundleCatalog {
    entries: BTreeMap<String, CatalogEntry>,
}

impl BundleCatalog {
    /// Empty catalog.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Installed entries.
    #[must_use]
    pub fn entries(&self) -> &BTreeMap<String, CatalogEntry> {
        &self.entries
    }

    /// Look up a bundle id.
    #[must_use]
    pub fn get(&self, bundle_id: &str) -> Option<&CatalogEntry> {
        self.entries.get(bundle_id)
    }

    /// Number of installed bundles.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the catalog is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Offer a manifest for hot reload.
    ///
    /// # Errors
    ///
    /// Rejects an invalid manifest.
    pub fn offer(&mut self, manifest: &Manifest) -> Result<ReloadVerdict> {
        let incoming = CatalogEntry::from_manifest(manifest)?;
        if let Some(existing) = self.entries.get(&incoming.bundle_id) {
            if existing == &incoming {
                return Ok(ReloadVerdict::AcceptedIdentical);
            }
            return Ok(ReloadVerdict::RejectedConflict {
                existing: Box::new(existing.clone()),
                incoming: Box::new(incoming),
            });
        }
        self.entries.insert(incoming.bundle_id.clone(), incoming);
        Ok(ReloadVerdict::AcceptedNew)
    }
}
