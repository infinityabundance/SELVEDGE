//! Canonical byte and JSON encodings used for every content-derived identity.
//!
//! The byte writer starts with a domain separation tag, writes lengths and
//! integers in explicit little-endian form, and never depends on hash-map
//! iteration order. The JSON canonicalizer sorts object keys recursively. Both
//! are stable across platforms and across runs.

use serde::Serialize;
use serde_json::{Map, Value};

use crate::error::{CoreError, Result};
use crate::hash::Sha256Digest;

/// Domain separation prefix for the canonical byte writer.
pub const CANONICAL_DOMAIN_TAG: &[u8] = b"SELVEDGE-CANONICAL-V1\0";

/// Append-only canonical byte writer.
#[derive(Debug, Clone)]
pub struct CanonicalWriter {
    buf: Vec<u8>,
}

impl CanonicalWriter {
    /// Start a canonical encoding in the named domain.
    ///
    /// # Errors
    ///
    /// Fails when the domain is longer than 65,535 bytes.
    pub fn new(domain: &str) -> Result<Self> {
        let len = u16::try_from(domain.len()).map_err(|_| {
            CoreError::invalid("canonical_domain", "domain longer than 65535 bytes")
        })?;
        let mut buf = Vec::with_capacity(CANONICAL_DOMAIN_TAG.len() + 2 + domain.len());
        buf.extend_from_slice(CANONICAL_DOMAIN_TAG);
        buf.extend_from_slice(&len.to_le_bytes());
        buf.extend_from_slice(domain.as_bytes());
        Ok(Self { buf })
    }

    /// Write a boolean as one byte (`0` or `1`).
    pub fn write_bool(&mut self, value: bool) {
        self.write_u8(u8::from(value));
    }

    /// Write one byte.
    pub fn write_u8(&mut self, value: u8) {
        self.buf.push(value);
    }

    /// Write a little-endian `u16`.
    pub fn write_u16(&mut self, value: u16) {
        self.buf.extend_from_slice(&value.to_le_bytes());
    }

    /// Write a little-endian `u32`.
    pub fn write_u32(&mut self, value: u32) {
        self.buf.extend_from_slice(&value.to_le_bytes());
    }

    /// Write a little-endian `u64`.
    pub fn write_u64(&mut self, value: u64) {
        self.buf.extend_from_slice(&value.to_le_bytes());
    }

    /// Write an optional `u32` as a presence byte followed by the value.
    pub fn write_opt_u32(&mut self, value: Option<u32>) {
        match value {
            Some(inner) => {
                self.write_bool(true);
                self.write_u32(inner);
            }
            None => self.write_bool(false),
        }
    }

    /// Write a length-prefixed byte string (64-bit little-endian length).
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        self.write_u64(bytes.len() as u64);
        self.buf.extend_from_slice(bytes);
    }

    /// Write a length-prefixed UTF-8 string.
    pub fn write_str(&mut self, value: &str) {
        self.write_bytes(value.as_bytes());
    }

    /// Finish the encoding.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }
}

/// Canonical compact JSON for any serializable value.
///
/// Object keys are sorted lexicographically at every depth; array order is
/// preserved because it is semantic. Numbers are rendered by `serde_json`.
///
/// # Errors
///
/// Fails when the value cannot be represented as JSON.
pub fn canonical_json<T: Serialize>(value: &T) -> Result<String> {
    let value =
        serde_json::to_value(value).map_err(|err| CoreError::invalid("json", err.to_string()))?;
    canonical_json_value(&value)
}

/// Canonical compact JSON for an already-parsed value.
///
/// # Errors
///
/// Fails only if a string cannot be re-encoded, which `serde_json` does not do
/// for valid UTF-8 values, but the error is propagated rather than ignored.
pub fn canonical_json_value(value: &Value) -> Result<String> {
    let mut out = String::new();
    write_value(value, &mut out)?;
    Ok(out)
}

/// SHA-256 of the canonical JSON encoding of a value.
///
/// # Errors
///
/// Propagates failures from [`canonical_json`].
pub fn canonical_json_sha256<T: Serialize>(value: &T) -> Result<Sha256Digest> {
    Ok(Sha256Digest::of(canonical_json(value)?.as_bytes()))
}

fn write_value(value: &Value, out: &mut String) -> Result<()> {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(number) => out.push_str(&number.to_string()),
        Value::String(string) => out.push_str(&encode_json_string(string)?),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_value(item, out)?;
            }
            out.push(']');
        }
        Value::Object(map) => write_object(map, out)?,
    }
    Ok(())
}

fn write_object(map: &Map<String, Value>, out: &mut String) -> Result<()> {
    let mut keys: Vec<&String> = map.keys().collect();
    keys.sort_unstable();
    out.push('{');
    for (index, key) in keys.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&encode_json_string(key)?);
        out.push(':');
        let value = map.get(*key).ok_or_else(|| {
            CoreError::invalid("canonical_json", "object key vanished while sorting")
        })?;
        write_value(value, out)?;
    }
    out.push('}');
    Ok(())
}

fn encode_json_string(value: &str) -> Result<String> {
    serde_json::to_string(value)
        .map_err(|err| CoreError::invalid("canonical_json", err.to_string()))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn integers_are_little_endian_and_length_prefixed() {
        let mut writer = CanonicalWriter::new("test").unwrap();
        writer.write_u16(0x0102);
        writer.write_u32(0x0304_0506);
        writer.write_u64(0x0708_090a_0b0c_0d0e);
        writer.write_str("hi");
        let bytes = writer.into_bytes();
        assert!(bytes.ends_with(&[
            0x02, 0x01, 0x06, 0x05, 0x04, 0x03, 0x0e, 0x0d, 0x0c, 0x0b, 0x0a, 0x09, 0x08, 0x07,
            0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, b'h', b'i',
        ]));
    }

    #[test]
    fn key_order_does_not_change_canonical_json_or_hash() {
        let first: Value =
            serde_json::from_str(r#"{"b":2,"a":{"d":4,"c":[3,1]},"e":null}"#).unwrap();
        let second: Value =
            serde_json::from_str(r#"{"e":null,"a":{"c":[3,1],"d":4},"b":2}"#).unwrap();
        assert_eq!(
            canonical_json_value(&first).unwrap(),
            canonical_json_value(&second).unwrap()
        );
        assert_eq!(
            canonical_json_sha256(&first).unwrap(),
            canonical_json_sha256(&second).unwrap()
        );
    }

    #[test]
    fn struct_field_order_is_canonicalised() {
        #[derive(serde::Serialize)]
        struct Forward {
            alpha: u32,
            beta: u32,
        }
        #[derive(serde::Serialize)]
        struct Reversed {
            beta: u32,
            alpha: u32,
        }
        let forward = canonical_json(&Forward { alpha: 1, beta: 2 }).unwrap();
        let reversed = canonical_json(&Reversed { beta: 2, alpha: 1 }).unwrap();
        assert_eq!(forward, reversed);
        assert_eq!(forward, r#"{"alpha":1,"beta":2}"#);
    }

    #[test]
    fn canonical_json_matches_expected_shape() {
        let value = json!({"z": [1, 2, 3], "a": {"y": true, "x": "s"}});
        assert_eq!(
            canonical_json_value(&value).unwrap(),
            r#"{"a":{"x":"s","y":true},"z":[1,2,3]}"#
        );
    }
}
