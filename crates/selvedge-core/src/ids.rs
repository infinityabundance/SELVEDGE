//! Content-derived identifiers: model digests and shard ids.
//!
//! Both definitions are versioned and documented here because every later
//! phase depends on them being stable. Integers are little-endian and strings
//! are length-prefixed; no map iteration order is ever hashed.

use core::fmt;
use core::str::FromStr;

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::canonical::CanonicalWriter;
use crate::error::{CoreError, Result};
use crate::hash::Sha256Digest;
use crate::tensor::{ChannelSet, DType, QuantScheme, TensorRole};

/// Shard-container format version implemented by this repository.
pub const FORMAT_VERSION: u16 = 1;

/// Domain tag for shard identity derivation.
pub const SHARD_ID_DOMAIN: &str = "selvedge/shard-id/v1";

/// Domain tag for the normalized checkpoint fileset digest.
pub const MODEL_DIGEST_DOMAIN: &str = "selvedge/model-digest/v1";

/// One checkpoint file in the normalized model identity input.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointFile {
    /// Normalized relative path within the checkpoint fileset.
    pub path: String,
    /// SHA-256 of the file's bytes.
    pub sha256: Sha256Digest,
    /// File length in bytes.
    pub length: u64,
}

/// Identity of the source checkpoint plus optional upstream and tokenizer
/// provenance.
///
/// The digest is defined as `SHA-256(canonical encoding of the fileset)` where
/// the fileset is the list of [`CheckpointFile`] entries sorted by path,
/// encoded in the `selvedge/model-digest/v1` domain as
/// `u64 count` followed by, per file, length-prefixed `path`, 32 raw digest
/// bytes and little-endian `u64 length`. The upstream revision string and the
/// tokenizer digest are recorded verbatim; changing either changes the shard
/// identity even if the weight digest is unchanged.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelDigest {
    digest: Sha256Digest,
    upstream_revision: Option<String>,
    tokenizer_digest: Option<Sha256Digest>,
}

impl ModelDigest {
    /// Construct from an already-computed checkpoint digest.
    ///
    /// # Errors
    ///
    /// Rejects an empty upstream revision string.
    pub fn from_parts(
        digest: Sha256Digest,
        upstream_revision: Option<String>,
        tokenizer_digest: Option<Sha256Digest>,
    ) -> Result<Self> {
        if let Some(revision) = &upstream_revision {
            if revision.is_empty() {
                return Err(CoreError::invalid(
                    "upstream_revision",
                    "revision string is empty",
                ));
            }
            if revision.len() > 256 {
                return Err(CoreError::invalid(
                    "upstream_revision",
                    "revision string longer than 256 bytes",
                ));
            }
        }
        Ok(Self {
            digest,
            upstream_revision,
            tokenizer_digest,
        })
    }

    /// Compute the digest from a checkpoint fileset.
    ///
    /// # Errors
    ///
    /// Rejects duplicate paths and invalid revision strings.
    pub fn of_fileset(files: &[CheckpointFile]) -> Result<Self> {
        let mut sorted: Vec<&CheckpointFile> = files.iter().collect();
        sorted.sort_by(|left, right| left.path.cmp(&right.path));
        for pair in sorted.windows(2) {
            if pair[0].path == pair[1].path {
                return Err(CoreError::invalid(
                    "checkpoint_fileset",
                    format!("duplicate path {:?}", pair[0].path),
                ));
            }
        }
        let mut writer = CanonicalWriter::new(MODEL_DIGEST_DOMAIN)?;
        writer.write_u64(sorted.len() as u64);
        for file in &sorted {
            writer.write_str(&file.path);
            writer.write_bytes(file.sha256.as_bytes());
            writer.write_u64(file.length);
        }
        Self::from_parts(Sha256Digest::of(&writer.into_bytes()), None, None)
    }

    /// Raw checkpoint digest.
    #[must_use]
    pub fn digest(&self) -> Sha256Digest {
        self.digest
    }

    /// Upstream revision string, if recorded.
    #[must_use]
    pub fn upstream_revision(&self) -> Option<&str> {
        self.upstream_revision.as_deref()
    }

    /// Tokenizer digest, if recorded.
    #[must_use]
    pub fn tokenizer_digest(&self) -> Option<Sha256Digest> {
        self.tokenizer_digest
    }
}

impl fmt::Display for ModelDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.digest.to_hex())
    }
}

/// The exact fields that determine a shard's identity.
///
/// Two shards with different bytes but the same identity fields are a
/// format/developer error; the format validator recomputes every id and fails
/// closed on a mismatch.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShardIdentity {
    /// Container format version.
    pub format_version: u16,
    /// Source model identity.
    pub model: ModelDigest,
    /// Tensor role.
    pub role: TensorRole,
    /// Optional layer index.
    pub layer: Option<u32>,
    /// Channel set.
    pub channels: ChannelSet,
    /// Logical axis the channel set applies to.
    pub channel_axis: u32,
    /// Element type.
    pub dtype: DType,
    /// Quantization scheme, when the dtype requires one.
    pub quant: Option<QuantScheme>,
    /// Repacking or transform identifier.
    pub transform_id: String,
}

impl ShardIdentity {
    /// Validate the identity fields.
    ///
    /// # Errors
    ///
    /// Rejects an unsupported dtype, malformed channels or empty transform id.
    pub fn validate(&self) -> Result<()> {
        if self.format_version == 0 {
            return Err(CoreError::invalid("format_version", "version is zero"));
        }
        self.channels.validate()?;
        self.dtype.require_supported()?;
        if self.transform_id.is_empty() {
            return Err(CoreError::invalid("transform_id", "transform id is empty"));
        }
        if self.transform_id.len() > 128 {
            return Err(CoreError::invalid(
                "transform_id",
                "transform id longer than 128 bytes",
            ));
        }
        Ok(())
    }
}

/// SHA-256 identity of one shard.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ShardId(Sha256Digest);

impl ShardId {
    /// Wrap an existing digest.
    #[must_use]
    pub const fn from_digest(digest: Sha256Digest) -> Self {
        Self(digest)
    }

    /// Derive a shard id from its identity fields.
    ///
    /// Encoding (all integers little-endian):
    ///
    /// ```text
    /// domain("selvedge/shard-id/v1")
    /// u16  format_version
    /// 32B  model.digest
    /// u8   has_upstream_revision, [str revision]
    /// u8   has_tokenizer_digest, [32B tokenizer digest]
    /// str  role (stable name)
    /// u8   has_layer, [u32 layer]
    /// str  "range" | "list"
    ///      range: u64 start, u64 end
    ///      list:  u64 count, then u64 index per entry
    /// u32  channel_axis
    /// str  dtype stable name
    /// str  "none" | quant scheme stable name
    /// str  transform_id
    /// ```
    ///
    /// # Errors
    ///
    /// Rejects invalid identity fields and unsupported dtypes.
    pub fn derive(identity: &ShardIdentity) -> Result<Self> {
        identity.validate()?;
        let mut writer = CanonicalWriter::new(SHARD_ID_DOMAIN)?;
        writer.write_u16(identity.format_version);
        writer.write_bytes(identity.model.digest().as_bytes());
        if let Some(revision) = identity.model.upstream_revision() {
            writer.write_bool(true);
            writer.write_str(revision);
        } else {
            writer.write_bool(false);
        }
        if let Some(tokenizer) = identity.model.tokenizer_digest() {
            writer.write_bool(true);
            writer.write_bytes(tokenizer.as_bytes());
        } else {
            writer.write_bool(false);
        }
        writer.write_str(&identity.role.to_string());
        writer.write_opt_u32(identity.layer);
        match &identity.channels {
            ChannelSet::Range { start, end } => {
                writer.write_str("range");
                writer.write_u64(*start);
                writer.write_u64(*end);
            }
            ChannelSet::List { indices } => {
                writer.write_str("list");
                writer.write_u64(indices.len() as u64);
                for index in indices {
                    writer.write_u64(*index);
                }
            }
        }
        writer.write_u32(identity.channel_axis);
        writer.write_str(&identity.dtype.stable_name());
        match &identity.quant {
            Some(scheme) => writer.write_str(&scheme.stable_name()),
            None => writer.write_str("none"),
        }
        writer.write_str(&identity.transform_id);
        Ok(Self(Sha256Digest::of(&writer.into_bytes())))
    }

    /// Underlying digest.
    #[must_use]
    pub const fn digest(&self) -> Sha256Digest {
        self.0
    }

    /// Lowercase hex without prefix.
    #[must_use]
    pub fn to_hex(&self) -> String {
        self.0.to_hex()
    }

    /// First 12 hex characters, for tables and logs.
    #[must_use]
    pub fn short(&self) -> String {
        self.0.to_hex().chars().take(12).collect()
    }

    /// Parse `sha256:<hex>` or bare 64-digit hex.
    ///
    /// # Errors
    ///
    /// Rejects malformed digests.
    pub fn parse(input: &str) -> Result<Self> {
        Ok(Self(Sha256Digest::parse_hex(input)?))
    }
}

impl fmt::Debug for ShardId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ShardId(")?;
        f.write_str(&self.0.to_hex())?;
        f.write_str(")")
    }
}

impl fmt::Display for ShardId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sha256:{}", self.0.to_hex())
    }
}

impl FromStr for ShardId {
    type Err = CoreError;

    fn from_str(input: &str) -> Result<Self> {
        Self::parse(input)
    }
}

impl Serialize for ShardId {
    fn serialize<S: Serializer>(&self, serializer: S) -> core::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for ShardId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> core::result::Result<Self, D::Error> {
        struct ShardVisitor;

        impl Visitor<'_> for ShardVisitor {
            type Value = ShardId;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("sha256:<64 hex digits>")
            }

            fn visit_str<E: de::Error>(self, value: &str) -> core::result::Result<Self::Value, E> {
                ShardId::parse(value).map_err(E::custom)
            }
        }

        deserializer.deserialize_str(ShardVisitor)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn model() -> ModelDigest {
        ModelDigest::from_parts(Sha256Digest::of(b"model"), Some("rev1".to_owned()), None).unwrap()
    }

    fn identity() -> ShardIdentity {
        ShardIdentity {
            format_version: FORMAT_VERSION,
            model: model(),
            role: TensorRole::MlpGate,
            layer: Some(3),
            channels: ChannelSet::range(16, 32).unwrap(),
            channel_axis: 0,
            dtype: DType::F32,
            quant: None,
            transform_id: "identity".to_owned(),
        }
    }

    #[test]
    fn shard_id_is_stable_and_display_round_trips() {
        let first = ShardId::derive(&identity()).unwrap();
        let second = ShardId::derive(&identity()).unwrap();
        assert_eq!(first, second);
        assert_eq!(ShardId::parse(&first.to_string()).unwrap(), first);
        assert_eq!(first.short().len(), 12);
    }

    #[test]
    fn every_identity_field_changes_the_id() {
        let base = ShardId::derive(&identity()).unwrap();
        let mut changed = identity();
        changed.layer = Some(4);
        assert_ne!(ShardId::derive(&changed).unwrap(), base);

        let mut changed = identity();
        changed.channels = ChannelSet::list([16_u64, 18, 20]).unwrap();
        assert_ne!(ShardId::derive(&changed).unwrap(), base);

        let mut changed = identity();
        changed.quant = Some(QuantScheme::Symmetric { block_size: 32 });
        changed.dtype = DType::I8;
        assert_ne!(ShardId::derive(&changed).unwrap(), base);

        let mut changed = identity();
        changed.transform_id = "f32-to-f16".to_owned();
        assert_ne!(ShardId::derive(&changed).unwrap(), base);

        let mut changed = identity();
        changed.model =
            ModelDigest::from_parts(Sha256Digest::of(b"model"), Some("rev2".to_owned()), None)
                .unwrap();
        assert_ne!(ShardId::derive(&changed).unwrap(), base);
    }

    #[test]
    fn model_digest_is_order_independent() {
        let files_a = vec![
            CheckpointFile {
                path: "b.safetensors".to_owned(),
                sha256: Sha256Digest::of(b"b"),
                length: 2,
            },
            CheckpointFile {
                path: "a.safetensors".to_owned(),
                sha256: Sha256Digest::of(b"a"),
                length: 1,
            },
        ];
        let mut files_b = files_a.clone();
        files_b.reverse();
        assert_eq!(
            ModelDigest::of_fileset(&files_a).unwrap(),
            ModelDigest::of_fileset(&files_b).unwrap()
        );
    }

    #[test]
    fn model_digest_rejects_duplicate_paths() {
        let file = CheckpointFile {
            path: "a".to_owned(),
            sha256: Sha256Digest::of(b"a"),
            length: 1,
        };
        assert!(ModelDigest::of_fileset(&[file.clone(), file]).is_err());
    }
}
