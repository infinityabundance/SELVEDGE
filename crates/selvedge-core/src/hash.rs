//! SHA-256 digest values, hex parsing and incremental hashing.
//!
//! Digests are stored as raw bytes and rendered as lowercase hex. Parsing
//! accepts an optional `sha256:` prefix but requires exactly 64 hex digits.

use core::fmt;
use core::str::FromStr;

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};

use crate::error::{CoreError, Result};

/// Number of raw bytes in a SHA-256 digest.
pub const SHA256_LEN: usize = 32;

/// A validated SHA-256 digest.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Sha256Digest([u8; SHA256_LEN]);

impl Sha256Digest {
    /// Hash a byte slice.
    #[must_use]
    pub fn of(bytes: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        let out = hasher.finalize();
        let mut raw = [0_u8; SHA256_LEN];
        raw.copy_from_slice(&out);
        Self(raw)
    }

    /// Wrap raw digest bytes.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; SHA256_LEN]) -> Self {
        Self(bytes)
    }

    /// Parse 64 hex digits, optionally prefixed with `sha256:`.
    pub fn parse_hex(input: &str) -> Result<Self> {
        let hex = input.strip_prefix("sha256:").unwrap_or(input);
        if hex.len() != SHA256_LEN * 2 {
            return Err(CoreError::InvalidDigest {
                reason: format!("expected 64 hex digits, found {}", hex.len()),
            });
        }
        let mut raw = [0_u8; SHA256_LEN];
        let (chunks, _remainder) = hex.as_bytes().as_chunks::<2>();
        for (index, chunk) in chunks.iter().enumerate() {
            let hi = hex_nibble(chunk[0]).ok_or_else(|| CoreError::InvalidDigest {
                reason: format!("non-hex character in digest: {:?}", chunk[0] as char),
            })?;
            let lo = hex_nibble(chunk[1]).ok_or_else(|| CoreError::InvalidDigest {
                reason: format!("non-hex character in digest: {:?}", chunk[1] as char),
            })?;
            raw[index] = (hi << 4) | lo;
        }
        Ok(Self(raw))
    }

    /// Raw bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; SHA256_LEN] {
        &self.0
    }

    /// Lowercase hex rendering (no prefix).
    #[must_use]
    pub fn to_hex(&self) -> String {
        let mut out = String::with_capacity(SHA256_LEN * 2);
        for byte in self.0 {
            out.push(HEX[(byte >> 4) as usize] as char);
            out.push(HEX[(byte & 0x0f) as usize] as char);
        }
        out
    }
}

const HEX: &[u8; 16] = b"0123456789abcdef";

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

impl fmt::Debug for Sha256Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("sha256:")?;
        f.write_str(&self.to_hex())
    }
}

impl fmt::Display for Sha256Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl FromStr for Sha256Digest {
    type Err = CoreError;

    fn from_str(input: &str) -> Result<Self> {
        Self::parse_hex(input)
    }
}

impl Serialize for Sha256Digest {
    fn serialize<S: Serializer>(&self, serializer: S) -> core::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for Sha256Digest {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> core::result::Result<Self, D::Error> {
        struct DigestVisitor;

        impl Visitor<'_> for DigestVisitor {
            type Value = Sha256Digest;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a 64-character hex sha256 digest")
            }

            fn visit_str<E: de::Error>(self, value: &str) -> core::result::Result<Self::Value, E> {
                Sha256Digest::parse_hex(value).map_err(E::custom)
            }
        }

        deserializer.deserialize_str(DigestVisitor)
    }
}

/// Incremental SHA-256 builder for streaming payload verification.
#[derive(Default)]
pub struct Sha256Builder {
    inner: Sha256,
}

impl Sha256Builder {
    /// Start a new digest.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed bytes into the digest.
    pub fn update(&mut self, bytes: &[u8]) {
        self.inner.update(bytes);
    }

    /// Finish and return the digest.
    #[must_use]
    pub fn finish(self) -> Sha256Digest {
        let out = self.inner.finalize();
        let mut raw = [0_u8; SHA256_LEN];
        raw.copy_from_slice(&out);
        Sha256Digest(raw)
    }
}

impl fmt::Debug for Sha256Builder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Sha256Builder { .. }")
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn empty_vector_matches_known_sha256() {
        assert_eq!(
            Sha256Digest::of(b"").to_hex(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn parse_and_display_round_trip() {
        let digest = Sha256Digest::of(b"selvedge");
        let text = digest.to_hex();
        assert_eq!(Sha256Digest::parse_hex(&text).unwrap(), digest);
        let prefixed = format!("sha256:{text}");
        assert_eq!(Sha256Digest::parse_hex(&prefixed).unwrap(), digest);
        assert_eq!(Sha256Digest::from_str(&text).unwrap(), digest);
    }

    #[test]
    fn rejects_wrong_length_and_non_hex() {
        assert!(Sha256Digest::parse_hex("abc").is_err());
        let bad = "z".repeat(64);
        assert!(Sha256Digest::parse_hex(&bad).is_err());
    }

    #[test]
    fn incremental_matches_one_shot() {
        let mut builder = Sha256Builder::new();
        builder.update(b"selve");
        builder.update(b"dge");
        assert_eq!(builder.finish(), Sha256Digest::of(b"selvedge"));
    }
}
