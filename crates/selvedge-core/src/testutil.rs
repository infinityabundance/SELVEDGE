//! Test-support filesystem helpers.
//!
//! These helpers exist so court tests create and remove bounded temporary
//! directories without adding a dependency. They are not used by any library
//! accounting path.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{CoreError, Result};

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

/// A temporary directory removed on drop.
#[derive(Debug)]
pub struct TempDir {
    path: PathBuf,
}

impl TempDir {
    /// Create a directory under `SELVEDGE_TEST_TMP`, then `TMPDIR`, then
    /// `/tmp`.
    ///
    /// # Errors
    ///
    /// Rejects an invalid prefix and propagates directory-creation failures
    /// (including a full bounded `/tmp`).
    pub fn new(prefix: &str) -> Result<Self> {
        if prefix.is_empty()
            || prefix.len() > 64
            || !prefix
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
        {
            return Err(CoreError::invalid(
                "temp_prefix",
                "prefix must be 1..=64 bytes of A-Za-z0-9._-",
            ));
        }
        let base = std::env::var_os("SELVEDGE_TEST_TMP")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("TMPDIR").map(PathBuf::from))
            .unwrap_or_else(|| PathBuf::from("/tmp"));
        let pid = std::process::id();
        let mut last_error = None;
        for _ in 0..64 {
            let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
            let path = base.join(format!("{prefix}-{pid}-{id}"));
            match std::fs::create_dir(&path) {
                Ok(()) => return Ok(Self { path }),
                Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
                    last_error = Some(err);
                }
                Err(err) => {
                    return Err(CoreError::Io(err));
                }
            }
        }
        Err(CoreError::Io(last_error.unwrap_or_else(|| {
            std::io::Error::other("could not allocate a unique temporary directory")
        })))
    }

    /// Directory path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// A path inside the directory.
    #[must_use]
    pub fn join(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::TempDir;

    #[test]
    fn tempdir_is_created_and_removed() {
        let path = {
            let dir = TempDir::new("selvedge-core").unwrap();
            let path = dir.path().to_path_buf();
            assert!(path.is_dir());
            path
        };
        assert!(!path.exists());
    }

    #[test]
    fn rejects_bad_prefix() {
        assert!(TempDir::new("../escape").is_err());
        assert!(TempDir::new("").is_err());
    }
}
