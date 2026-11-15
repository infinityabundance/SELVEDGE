//! Read-only host and device facts used by the I/O measurements.
//!
//! Nothing here is inferred from an earlier session: every value is read from
//! `/proc` or `/sys` inside the court at measurement time, and the receipt names
//! the source file. Procfs memory values are host-wide (not namespaced); the
//! cgroup limit is read separately so a reader can tell the two apart.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use rustix::fs::statvfs;
use serde::Serialize;

use crate::error::{Result, StoreError};

/// Host memory facts.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryInfo {
    /// `MemTotal` from `/proc/meminfo`.
    pub total_bytes: u64,
    /// `MemAvailable` from `/proc/meminfo`.
    pub available_bytes: u64,
    /// `SwapTotal`.
    pub swap_total_bytes: u64,
    /// `SwapFree`.
    pub swap_free_bytes: u64,
    /// `memory.max` from the cgroup, when readable.
    pub cgroup_limit_bytes: Option<u64>,
    /// `memory.current` from the cgroup, when readable.
    pub cgroup_current_bytes: Option<u64>,
    /// Source description recorded in receipts.
    pub source: String,
}

/// Read memory facts from `/proc/meminfo` and the cgroup.
///
/// # Errors
///
/// Rejects an unreadable `/proc/meminfo`.
pub fn memory_info() -> Result<MemoryInfo> {
    let text = std::fs::read_to_string("/proc/meminfo")?;
    let mut fields: BTreeMap<String, u64> = BTreeMap::new();
    for line in text.lines() {
        let Some((key, rest)) = line.split_once(':') else {
            continue;
        };
        let value = rest
            .split_whitespace()
            .next()
            .and_then(|token| token.parse::<u64>().ok())
            .unwrap_or(0);
        fields.insert(key.trim().to_owned(), value.saturating_mul(1024));
    }
    let get = |name: &str| fields.get(name).copied().unwrap_or(0);
    Ok(MemoryInfo {
        total_bytes: get("MemTotal"),
        available_bytes: get("MemAvailable"),
        swap_total_bytes: get("SwapTotal"),
        swap_free_bytes: get("SwapFree"),
        cgroup_limit_bytes: cgroup_value("/sys/fs/cgroup/memory.max"),
        cgroup_current_bytes: cgroup_value("/sys/fs/cgroup/memory.current"),
        source: "/proc/meminfo and /sys/fs/cgroup/memory.{max,current}".to_owned(),
    })
}

fn cgroup_value(path: &str) -> Option<u64> {
    let text = std::fs::read_to_string(path).ok()?;
    let trimmed = text.trim();
    if trimmed.is_empty() || trimmed == "max" {
        return None;
    }
    trimmed.parse::<u64>().ok()
}

/// Filesystem and mount facts for a path.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MountInfo {
    /// Path that was inspected.
    pub path: PathBuf,
    /// Block device or pseudo device backing the mount.
    pub source: String,
    /// Mount point.
    pub target: String,
    /// Filesystem type.
    pub fstype: String,
    /// Mount options, verbatim.
    pub options: String,
    /// Block device name (for example `nvme1n1`), when the source is a device.
    pub device_name: Option<String>,
    /// Total bytes from `statvfs`.
    pub total_bytes: u64,
    /// Free bytes available to an unprivileged writer.
    pub free_bytes: u64,
    /// Block size reported by `statvfs`.
    pub block_size: u64,
}

/// Resolve the mount that contains `path` and measure its free space.
///
/// # Errors
///
/// Rejects an unreadable `/proc/mounts`, an unmounted path, or a failing
/// `statvfs`.
pub fn mount_for(path: &Path) -> Result<MountInfo> {
    let canonical = path
        .canonicalize()
        .map_err(|err| StoreError::precondition(format!("cannot canonicalize {path:?}: {err}")))?;
    let mounts = std::fs::read_to_string("/proc/mounts")?;
    let mut best: Option<(usize, String, String, String, String)> = None;
    for line in mounts.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 4 {
            continue;
        }
        let source = unescape(fields[0]);
        let target = unescape(fields[1]);
        let fstype = fields[2].to_owned();
        let options = fields[3].to_owned();
        let target_path = Path::new(&target);
        if !canonical.starts_with(target_path) {
            continue;
        }
        // Overlayfs layers can share a prefix; the longest match wins.
        let depth = target.len();
        if best.as_ref().is_none_or(|(current, ..)| depth > *current) {
            best = Some((depth, source, target, fstype, options));
        }
    }
    let Some((_, source, target, fstype, options)) = best else {
        return Err(StoreError::precondition(format!(
            "no mount contains {canonical:?}"
        )));
    };
    let stats = statvfs(&canonical)?;
    let fragment = stats.f_frsize.max(1);
    Ok(MountInfo {
        path: canonical,
        device_name: device_name(&source),
        source,
        target,
        fstype,
        options,
        total_bytes: stats.f_blocks.saturating_mul(fragment),
        free_bytes: stats.f_bavail.saturating_mul(fragment),
        block_size: stats.f_bsize,
    })
}

fn unescape(field: &str) -> String {
    // /proc/mounts escapes space, tab, newline and backslash as \OOO.
    let mut out = String::with_capacity(field.len());
    let bytes = field.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'\\' && index + 3 < bytes.len() {
            let digits = &field[index + 1..index + 4];
            if let Ok(value) = u8::from_str_radix(digits, 8) {
                out.push(value as char);
                index += 4;
                continue;
            }
        }
        out.push(bytes[index] as char);
        index += 1;
    }
    out
}

fn device_name(source: &str) -> Option<String> {
    let base = source.rsplit('/').next()?;
    if !source.starts_with("/dev/") || base.is_empty() {
        return None;
    }
    // nvme0n1p1 -> nvme0n1; sda1 -> sda; vdb2 -> vdb.
    if let Some(stripped) = base.strip_suffix(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']) {
        if let Some(name) = stripped.strip_suffix('p') {
            return Some(name.to_owned());
        }
        if stripped.starts_with("sd") || stripped.starts_with("vd") || stripped.starts_with("hd") {
            return Some(stripped.to_owned());
        }
    }
    Some(base.to_owned())
}

/// Block-device counters from `/sys/block/<name>/stat`.
///
/// The kernel reports the two time fields in milliseconds; they are kept in
/// that unit and named accordingly rather than rescaled into nanoseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceCounters {
    /// Completed read requests.
    pub reads_completed: u64,
    /// Merged read requests.
    pub reads_merged: u64,
    /// Sectors read (512-byte units, as the kernel reports them).
    pub sectors_read: u64,
    /// Kernel-reported time spent reading, in milliseconds.
    pub read_time_ms: u64,
    /// Completed write requests.
    pub writes_completed: u64,
    /// Sectors written.
    pub sectors_written: u64,
    /// Kernel-reported time spent writing, in milliseconds.
    pub write_time_ms: u64,
}

impl DeviceCounters {
    /// Bytes read, assuming the kernel's 512-byte sector convention.
    #[must_use]
    pub const fn read_bytes(&self) -> u64 {
        self.sectors_read.saturating_mul(512)
    }

    /// Bytes written.
    #[must_use]
    pub const fn written_bytes(&self) -> u64 {
        self.sectors_written.saturating_mul(512)
    }

    /// Difference from an earlier snapshot, saturating at zero.
    #[must_use]
    pub fn delta(&self, earlier: &Self) -> Self {
        Self {
            reads_completed: self.reads_completed.saturating_sub(earlier.reads_completed),
            reads_merged: self.reads_merged.saturating_sub(earlier.reads_merged),
            sectors_read: self.sectors_read.saturating_sub(earlier.sectors_read),
            read_time_ms: self.read_time_ms.saturating_sub(earlier.read_time_ms),
            writes_completed: self
                .writes_completed
                .saturating_sub(earlier.writes_completed),
            sectors_written: self.sectors_written.saturating_sub(earlier.sectors_written),
            write_time_ms: self.write_time_ms.saturating_sub(earlier.write_time_ms),
        }
    }
}

/// Read the kernel's counters for a block device.
///
/// # Errors
///
/// Rejects an unreadable or malformed `/sys/block/<name>/stat`.
pub fn device_counters(name: &str) -> Result<DeviceCounters> {
    let path = format!("/sys/block/{name}/stat");
    let text = std::fs::read_to_string(&path)
        .map_err(|err| StoreError::precondition(format!("cannot read {path}: {err}")))?;
    let fields: Vec<&str> = text.split_whitespace().collect();
    if fields.len() < 11 {
        return Err(StoreError::precondition(format!(
            "{path} has {} fields, expected at least 11",
            fields.len()
        )));
    }
    let parse = |index: usize| -> u64 { fields[index].parse::<u64>().unwrap_or(0) };
    Ok(DeviceCounters {
        reads_completed: parse(0),
        reads_merged: parse(1),
        sectors_read: parse(2),
        read_time_ms: parse(3),
        writes_completed: parse(4),
        sectors_written: parse(6),
        write_time_ms: parse(7),
    })
}

/// Hardware and environment block recorded in every measurement receipt.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Environment {
    /// Hostname.
    pub hostname: String,
    /// Kernel release.
    pub kernel: String,
    /// Architecture.
    pub arch: String,
    /// CPU model string.
    pub cpu_model: String,
    /// Logical CPUs visible to the process.
    pub logical_cpus: u64,
    /// Memory facts.
    pub memory: MemoryInfo,
    /// Court name from the environment, when present.
    pub court: Option<String>,
    /// Court memory limit in bytes, when present.
    pub court_memory_bytes: Option<u64>,
    /// Git revision from the court environment, when present.
    pub revision: Option<String>,
}

/// Collect the environment block.
///
/// # Errors
///
/// Propagates memory and `/proc` read failures.
pub fn environment() -> Result<Environment> {
    let kernel = std::fs::read_to_string("/proc/sys/kernel/osrelease")
        .map(|text| text.trim().to_owned())
        .unwrap_or_else(|_| "unknown".to_owned());
    let cpu_model = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|line| line.starts_with("model name"))
                .and_then(|line| line.split_once(':'))
                .map(|(_, value)| value.trim().to_owned())
        })
        .unwrap_or_else(|| "unknown".to_owned());
    let hostname = std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|text| text.trim().to_owned())
        .unwrap_or_else(|_| "unknown".to_owned());
    let logical_cpus = std::thread::available_parallelism()
        .map(|value| u64::try_from(value.get()).unwrap_or(1))
        .unwrap_or(1);
    Ok(Environment {
        hostname,
        kernel,
        arch: std::env::consts::ARCH.to_owned(),
        cpu_model,
        logical_cpus,
        memory: memory_info()?,
        court: std::env::var("SELVEDGE_COURT").ok(),
        court_memory_bytes: std::env::var("SELVEDGE_COURT_MEM_BYTES")
            .ok()
            .and_then(|value| value.parse::<u64>().ok()),
        revision: std::env::var("SELVEDGE_REVISION").ok(),
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn partition_names_map_to_block_devices() {
        assert_eq!(device_name("/dev/nvme0n1p1"), Some("nvme0n1".to_owned()));
        assert_eq!(device_name("/dev/nvme1n1"), Some("nvme1n1".to_owned()));
        assert_eq!(device_name("/dev/sda1"), Some("sda".to_owned()));
        assert_eq!(device_name("tmpfs"), None);
    }

    #[test]
    fn deltas_saturate_at_zero() {
        let earlier = DeviceCounters {
            reads_completed: 10,
            sectors_read: 100,
            ..DeviceCounters::default()
        };
        let later = DeviceCounters {
            reads_completed: 12,
            sectors_read: 80,
            ..DeviceCounters::default()
        };
        let delta = later.delta(&earlier);
        assert_eq!(delta.reads_completed, 2);
        assert_eq!(delta.sectors_read, 0);
    }

    #[test]
    fn mount_lookup_finds_the_repository_filesystem() {
        let info = mount_for(Path::new(".")).unwrap();
        assert!(info.total_bytes > 0);
        assert!(info.free_bytes <= info.total_bytes);
    }
}
