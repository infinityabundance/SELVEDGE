//! Typed configuration with explicit hard limits and a canonical hash.

use serde::{Deserialize, Serialize};

use crate::canonical::canonical_json_sha256;
use crate::error::{CoreError, Result};
use crate::ledger::{Budget, ComplianceMode};
use crate::units::align_up;

/// Version of the configuration schema.
pub const CONFIG_SCHEMA_VERSION: u16 = 1;

/// Absolute limits this build will never exceed, regardless of configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardLimits {
    /// Largest accepted per-token active-unique-parameter ceiling.
    pub active_unique_ceiling: u64,
    /// Largest accepted resident memory cap.
    pub resident_cap_ceiling: u64,
    /// Largest accepted temporary-storage budget.
    pub tmp_budget_ceiling: u64,
    /// Smallest accepted shard alignment.
    pub min_alignment: u64,
    /// Largest accepted shard alignment.
    pub max_alignment: u64,
}

impl Default for HardLimits {
    fn default() -> Self {
        Self {
            active_unique_ceiling: 1_000_000_000,
            resident_cap_ceiling: 256 * 1024 * 1024 * 1024,
            tmp_budget_ceiling: 256 * 1024 * 1024 * 1024,
            min_alignment: 512,
            max_alignment: 1024 * 1024,
        }
    }
}

/// Budget configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BudgetConfig {
    /// Per-token active-unique-parameter ceiling.
    pub active_unique_limit: u64,
    /// Compliance or labelled teacher mode.
    pub mode: ComplianceMode,
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self {
            active_unique_limit: 1_000_000_000,
            mode: ComplianceMode::Compliance,
        }
    }
}

/// Memory and temporary-storage caps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryConfig {
    /// Maximum simultaneously resident learned bytes.
    pub resident_cap_bytes: u64,
    /// Maximum bytes this program may place in temporary storage.
    pub tmp_budget_bytes: u64,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            resident_cap_bytes: 4 * 1024 * 1024 * 1024,
            tmp_budget_bytes: 2 * 1024 * 1024 * 1024,
        }
    }
}

/// Storage and validator limits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StorageConfig {
    /// Shard extent alignment in bytes. 4096 is a documented default and a
    /// benchmark target, not a claim that it is optimal for every device.
    pub alignment: u64,
    /// Largest manifest this build will parse.
    pub max_manifest_bytes: u64,
    /// Largest bundle file this build will read.
    pub max_bundle_bytes: u64,
    /// Whether validation verifies payload hashes by default.
    pub verify_sha256: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            alignment: 4096,
            max_manifest_bytes: 8 * 1024 * 1024,
            max_bundle_bytes: 8 * 1024 * 1024 * 1024,
            verify_sha256: true,
        }
    }
}

/// Doctor thresholds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DoctorConfig {
    /// Free space below which `selvedge doctor` fails.
    pub min_free_bytes: u64,
}

impl Default for DoctorConfig {
    fn default() -> Self {
        Self {
            min_free_bytes: 64 * 1024 * 1024,
        }
    }
}

/// Complete typed configuration.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Configuration schema version.
    pub schema_version: u16,
    /// Budget configuration.
    pub budget: BudgetConfig,
    /// Memory limits.
    pub memory: MemoryConfig,
    /// Storage/validator limits.
    pub storage: StorageConfig,
    /// Doctor thresholds.
    pub doctor: DoctorConfig,
    /// Absolute limits.
    pub hard_limits: HardLimits,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            schema_version: CONFIG_SCHEMA_VERSION,
            budget: BudgetConfig::default(),
            memory: MemoryConfig::default(),
            storage: StorageConfig::default(),
            doctor: DoctorConfig::default(),
            hard_limits: HardLimits::default(),
        }
    }
}

impl Config {
    /// Parse TOML and validate.
    ///
    /// # Errors
    ///
    /// Rejects malformed TOML, unknown fields and every limit violation
    /// reported by [`Config::validate`].
    pub fn from_toml_str(input: &str) -> Result<Self> {
        let config: Self = toml::from_str(input).map_err(|err| CoreError::Config {
            reason: err.to_string(),
        })?;
        config.validate()?;
        Ok(config)
    }

    /// Validate all hard limits.
    ///
    /// # Errors
    ///
    /// Rejects: an unknown schema version; a zero budget; a budget above the
    /// ceiling; zero or oversized memory/tmp caps; a non-power-of-two alignment
    /// outside the configured range; zero or oversized manifest/bundle caps;
    /// and a zero doctor threshold.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != CONFIG_SCHEMA_VERSION {
            return Err(CoreError::Config {
                reason: format!(
                    "schema version {} is not supported (expected {CONFIG_SCHEMA_VERSION})",
                    self.schema_version
                ),
            });
        }
        let limits = &self.hard_limits;
        if limits.active_unique_ceiling == 0
            || limits.resident_cap_ceiling == 0
            || limits.tmp_budget_ceiling == 0
        {
            return Err(CoreError::Config {
                reason: "hard-limit ceilings must be non-zero".to_owned(),
            });
        }
        Budget::new(self.budget.active_unique_limit, self.budget.mode).map_err(|err| {
            CoreError::Config {
                reason: err.to_string(),
            }
        })?;
        if self.budget.active_unique_limit > limits.active_unique_ceiling {
            return Err(CoreError::Config {
                reason: format!(
                    "budget limit {} exceeds hard ceiling {}",
                    self.budget.active_unique_limit, limits.active_unique_ceiling
                ),
            });
        }
        if self.memory.resident_cap_bytes == 0
            || self.memory.resident_cap_bytes > limits.resident_cap_ceiling
        {
            return Err(CoreError::Config {
                reason: "resident cap is zero or above the hard ceiling".to_owned(),
            });
        }
        if self.memory.tmp_budget_bytes == 0
            || self.memory.tmp_budget_bytes > limits.tmp_budget_ceiling
        {
            return Err(CoreError::Config {
                reason: "temporary budget is zero or above the hard ceiling".to_owned(),
            });
        }
        let alignment = self.storage.alignment;
        if !alignment.is_power_of_two()
            || alignment < limits.min_alignment
            || alignment > limits.max_alignment
        {
            return Err(CoreError::Config {
                reason: format!(
                    "alignment {alignment} must be a power of two in {}..={}",
                    limits.min_alignment, limits.max_alignment
                ),
            });
        }
        if self.storage.max_manifest_bytes == 0 {
            return Err(CoreError::Config {
                reason: "max manifest bytes must be non-zero".to_owned(),
            });
        }
        if self.storage.max_bundle_bytes == 0 {
            return Err(CoreError::Config {
                reason: "max bundle bytes must be non-zero".to_owned(),
            });
        }
        if self.storage.max_manifest_bytes > self.storage.max_bundle_bytes {
            return Err(CoreError::Config {
                reason: "max manifest bytes exceeds max bundle bytes".to_owned(),
            });
        }
        if self.doctor.min_free_bytes == 0 {
            return Err(CoreError::Config {
                reason: "doctor minimum free bytes must be non-zero".to_owned(),
            });
        }
        Ok(())
    }

    /// Budget value for the ledger.
    ///
    /// # Errors
    ///
    /// Rejects a zero limit.
    pub fn budget(&self) -> Result<Budget> {
        Budget::new(self.budget.active_unique_limit, self.budget.mode)
    }

    /// Coverage of the configured alignment as a multiple check.
    ///
    /// # Errors
    ///
    /// Rejects a zero alignment or overflow.
    pub fn round_up_alignment(&self, value: u64) -> Result<u64> {
        align_up(value, self.storage.alignment)
    }

    /// Canonical serialization: compact JSON with sorted keys.
    ///
    /// # Errors
    ///
    /// Propagates serialization failure.
    pub fn canonical_json(&self) -> Result<String> {
        crate::canonical::canonical_json(self)
    }

    /// SHA-256 of the canonical JSON, used to pin benchmark provenance.
    ///
    /// # Errors
    ///
    /// Propagates serialization failure.
    pub fn config_hash(&self) -> Result<String> {
        Ok(canonical_json_sha256(self)?.to_hex())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid_and_has_stable_hash() {
        let config = Config::default();
        config.validate().unwrap();
        let hash = config.config_hash().unwrap();
        assert_eq!(hash.len(), 64);
        assert_eq!(hash, Config::default().config_hash().unwrap());
    }

    #[test]
    fn toml_round_trip_preserves_hash() {
        let text = r#"
schema_version = 1

[budget]
active_unique_limit = 1000000000
mode = "compliance"

[memory]
resident_cap_bytes = 4294967296
tmp_budget_bytes = 2147483648

[storage]
alignment = 4096
max_manifest_bytes = 8388608
max_bundle_bytes = 8589934592
verify_sha256 = true

[doctor]
min_free_bytes = 67108864

[hard_limits]
active_unique_ceiling = 1000000000
resident_cap_ceiling = 274877906944
tmp_budget_ceiling = 274877906944
min_alignment = 512
max_alignment = 1048576
"#;
        let config = Config::from_toml_str(text).unwrap();
        assert_eq!(config, Config::default());
        assert_eq!(
            config.config_hash().unwrap(),
            Config::default().config_hash().unwrap()
        );
    }

    #[test]
    fn hard_limits_are_enforced() {
        let mut config = Config::default();
        config.budget.active_unique_limit = 1_000_000_001;
        assert!(matches!(config.validate(), Err(CoreError::Config { .. })));

        let mut config = Config::default();
        config.storage.alignment = 3000;
        assert!(config.validate().is_err());

        let mut config = Config::default();
        config.memory.resident_cap_bytes = 0;
        assert!(config.validate().is_err());

        let mut config = Config::default();
        config.budget.mode = ComplianceMode::ResearchNoncompliantTeacher;
        config.validate().unwrap();
    }

    #[test]
    fn unknown_fields_are_rejected() {
        assert!(Config::from_toml_str("schema_version = 1\nnonsense = true\n").is_err());
    }
}
