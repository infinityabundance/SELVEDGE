//! Retry policy.

use crate::config::Settings;

pub fn retries() -> u32 {
    Settings::default().retries
}
