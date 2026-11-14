//! Configuration loading shared by the CLI commands.

use std::path::PathBuf;

use selvedge_core::{Config, CoreError};

/// Load a TOML configuration, or the validated defaults when no path is given.
///
/// # Errors
///
/// Rejects unreadable files and malformed or non-compliant configurations.
pub(crate) fn load(path: &Option<PathBuf>) -> Result<Config, CoreError> {
    match path {
        None => Ok(Config::default()),
        Some(path) => {
            let text = std::fs::read_to_string(path).map_err(|err| CoreError::Config {
                reason: format!("cannot read {}: {err}", path.display()),
            })?;
            Config::from_toml_str(&text)
        }
    }
}
