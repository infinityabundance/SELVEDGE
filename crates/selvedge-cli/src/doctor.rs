//! `selvedge doctor`: environment, schema and configuration probe.
//!
//! Nothing is invented: unreadable optional inputs are reported as skipped, a
//! failed check makes the command exit non-zero, and GPU facts are never
//! fabricated. CUDA is not compiled into this build, which is stated directly.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Args;
use selvedge_core::FORMAT_VERSION;
use selvedge_format::{MANIFEST_SCHEMA, MANIFEST_SCHEMA_VERSION};
use serde::Serialize;
use serde_json::{Value, json};

use crate::CliResult;

#[derive(Debug, Args)]
pub(crate) struct DoctorArgs {
    /// Emit machine-readable JSON.
    #[arg(long)]
    pub(crate) json: bool,
    /// Read host CPU/RAM/NVMe facts read-only from `/proc` and `/sys`
    /// (or `/hostroot` when the court exposes it).
    #[arg(long)]
    pub(crate) probe_host: bool,
    /// Optional TOML configuration.
    #[arg(long)]
    pub(crate) config: Option<PathBuf>,
    /// Repository root to inspect; defaults to the nearest ancestor with
    /// `docker/IMAGES.lock` or `Cargo.toml`/`crates`.
    #[arg(long)]
    pub(crate) root: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum CheckStatus {
    Ok,
    Skipped,
    Failed,
}

#[derive(Debug, Serialize)]
struct Check {
    name: &'static str,
    status: CheckStatus,
    detail: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

#[derive(Debug, Serialize)]
struct DoctorReport {
    schema: &'static str,
    ok: bool,
    checks: Vec<Check>,
}

impl Check {
    fn ok(name: &'static str, detail: impl Into<String>, data: Option<Value>) -> Self {
        Self {
            name,
            status: CheckStatus::Ok,
            detail: detail.into(),
            data,
        }
    }

    fn skipped(name: &'static str, detail: impl Into<String>, data: Option<Value>) -> Self {
        Self {
            name,
            status: CheckStatus::Skipped,
            detail: detail.into(),
            data,
        }
    }

    fn failed(name: &'static str, detail: impl Into<String>, data: Option<Value>) -> Self {
        Self {
            name,
            status: CheckStatus::Failed,
            detail: detail.into(),
            data,
        }
    }
}

/// Run the doctor command.
///
/// # Errors
///
/// Propagates serialization failures only; environment problems are reported as
/// failed checks and produce a non-zero exit code.
pub(crate) fn run(args: &DoctorArgs) -> CliResult {
    let mut checks = Vec::new();
    let config = match crate::config::load(&args.config) {
        Ok(config) => {
            let hash = config.config_hash()?;
            checks.push(Check::ok(
                "config",
                format!("configuration valid; hash {hash}"),
                Some(json!({
                    "config_hash": hash,
                    "budget": {
                        "active_unique_limit": config.budget.active_unique_limit,
                        "mode": config.budget.mode,
                    },
                    "memory": {
                        "resident_cap_bytes": config.memory.resident_cap_bytes,
                        "tmp_budget_bytes": config.memory.tmp_budget_bytes,
                    },
                    "storage": {
                        "alignment": config.storage.alignment,
                        "max_manifest_bytes": config.storage.max_manifest_bytes,
                        "max_bundle_bytes": config.storage.max_bundle_bytes,
                        "verify_sha256": config.storage.verify_sha256,
                    },
                })),
            ));
            Some(config)
        }
        Err(error) => {
            checks.push(Check::failed("config", error.to_string(), None));
            None
        }
    };
    let effective = config.unwrap_or_default();

    checks.push(Check::ok(
        "format_schema",
        format!(
            "container version {FORMAT_VERSION}; manifest {MANIFEST_SCHEMA} v{MANIFEST_SCHEMA_VERSION}"
        ),
        Some(json!({
            "container_version": FORMAT_VERSION,
            "manifest_schema": MANIFEST_SCHEMA,
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        })),
    ));

    let root = args.root.clone().unwrap_or_else(|| {
        find_repo_root(&std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
    });
    checks.push(court_images_check(&root));
    checks.push(gpu_check());

    let target_dir = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("target"));
    match probe_writable(&target_dir) {
        Ok(()) => checks.push(Check::ok(
            "target_dir",
            format!("writable: {}", target_dir.display()),
            Some(json!({ "path": target_dir.display().to_string() })),
        )),
        Err(error) => checks.push(Check::failed(
            "target_dir",
            error,
            Some(json!({ "path": target_dir.display().to_string() })),
        )),
    }

    let min_free = effective.doctor.min_free_bytes;
    match available_bytes(&target_dir) {
        Ok(available) => {
            let detail = format!(
                "{available} bytes available at {} (minimum {min_free})",
                target_dir.display()
            );
            let data = Some(json!({
                "path": target_dir.display().to_string(),
                "available_bytes": available,
                "min_free_bytes": min_free,
            }));
            if available < min_free {
                checks.push(Check::failed("disk", detail, data));
            } else {
                checks.push(Check::ok("disk", detail, data));
            }
        }
        Err(error) => checks.push(Check::failed(
            "disk",
            format!("could not determine free space: {error}"),
            None,
        )),
    }

    if args.probe_host {
        checks.extend(probe_host_checks());
    }

    let ok = !checks
        .iter()
        .any(|check| check.status == CheckStatus::Failed);
    let report = DoctorReport {
        schema: "selvedge/doctor/v1",
        ok,
        checks,
    };
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("selvedge doctor");
        for check in &report.checks {
            let label = match check.status {
                CheckStatus::Ok => "OK  ",
                CheckStatus::Skipped => "SKIP",
                CheckStatus::Failed => "FAIL",
            };
            println!("{label} {:<16} {}", check.name, check.detail);
        }
        println!("overall: {}", if ok { "ok" } else { "failed" });
    }
    Ok(if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

fn find_repo_root(start: &Path) -> PathBuf {
    let start = match std::fs::canonicalize(start) {
        Ok(path) => path,
        Err(_) => start.to_path_buf(),
    };
    for candidate in start.ancestors() {
        if candidate.join("docker/IMAGES.lock").is_file()
            || (candidate.join("Cargo.toml").is_file() && candidate.join("crates").is_dir())
        {
            return candidate.to_path_buf();
        }
    }
    start
}

fn court_images_check(root: &Path) -> Check {
    let path = root.join("docker/IMAGES.lock");
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Check::skipped(
                "court_images",
                format!("{} not found; image identity unknown", path.display()),
                None,
            );
        }
        Err(error) => {
            return Check::failed(
                "court_images",
                format!("cannot read {}: {error}", path.display()),
                None,
            );
        }
    };
    let value: Value = match serde_json::from_str(&text) {
        Ok(value) => value,
        Err(error) => {
            return Check::failed(
                "court_images",
                format!("{} is not valid JSON: {error}", path.display()),
                None,
            );
        }
    };
    let rust = value.pointer("/images/rust");
    let reference = rust
        .and_then(|image| image.get("ref"))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let image_id = rust
        .and_then(|image| image.get("id"))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let rustc = rust
        .and_then(|image| image.pointer("/toolchain/rustc"))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let updated = value
        .get("updated_utc")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    Check::ok(
        "court_images",
        format!("rust court image {reference} ({image_id}), rustc {rustc}, lock updated {updated}"),
        Some(json!({
            "path": path.display().to_string(),
            "rust_ref": reference,
            "rust_image_id": image_id,
            "rustc": rustc,
            "lock_updated_utc": updated,
        })),
    )
}

fn gpu_check() -> Check {
    Check::ok(
        "gpu",
        "CUDA is not compiled into this build; no GPU facts are reported",
        Some(json!({
            "cuda_compiled_in": false,
            "gpu_facts": "not_probed",
        })),
    )
}

fn probe_writable(dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir)
        .map_err(|error| format!("cannot create {}: {error}", dir.display()))?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    let probe = dir.join(format!(".selvedge-doctor-{}-{stamp}", std::process::id()));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&probe)
        .map_err(|error| format!("cannot create {}: {error}", probe.display()))?;
    if let Err(error) = file.write_all(b"selvedge doctor write probe\n") {
        let _ = std::fs::remove_file(&probe);
        return Err(format!("cannot write {}: {error}", probe.display()));
    }
    if let Err(error) = file.sync_all() {
        let _ = std::fs::remove_file(&probe);
        return Err(format!("cannot fsync {}: {error}", probe.display()));
    }
    drop(file);
    std::fs::remove_file(&probe)
        .map_err(|error| format!("cannot remove {}: {error}", probe.display()))?;
    Ok(())
}

fn available_bytes(path: &Path) -> Result<u64, String> {
    let output = std::process::Command::new("df")
        .arg("-P")
        .arg("-B1")
        .arg(path)
        .output()
        .map_err(|error| format!("cannot run df: {error}"))?;
    if !output.status.success() {
        return Err(format!("df exited with {}", output.status));
    }
    let text = String::from_utf8(output.stdout).map_err(|error| format!("df output: {error}"))?;
    let line = text
        .lines()
        .last()
        .ok_or_else(|| "df produced no output".to_owned())?;
    let fields: Vec<&str> = line.split_whitespace().collect();
    let available = fields
        .get(3)
        .ok_or_else(|| format!("df output has no available field: {line:?}"))?;
    available
        .parse::<u64>()
        .map_err(|error| format!("df available value {available:?}: {error}"))
}

fn probe_host_checks() -> Vec<Check> {
    let host_mount = Path::new("/hostroot");
    let (base, source) = if host_mount.join("proc/cpuinfo").exists() {
        ("/hostroot".to_owned(), "read-only host facts via /hostroot")
    } else {
        (String::new(), "read-only container view of /proc and /sys")
    };
    let mut checks = Vec::with_capacity(3);

    let cpu_path = format!("{base}/proc/cpuinfo");
    match read_cpu(&cpu_path) {
        Ok((model, logical)) => checks.push(Check::ok(
            "host_cpu",
            format!("{model}, {logical} logical processors ({source})"),
            Some(json!({
                "model": model,
                "logical_processors": logical,
                "source": cpu_path,
            })),
        )),
        Err(error) => checks.push(Check::failed(
            "host_cpu",
            format!("{error} ({source})"),
            Some(json!({ "source": cpu_path })),
        )),
    }

    let mem_path = format!("{base}/proc/meminfo");
    match read_mem_total(&mem_path) {
        Ok(bytes) => checks.push(Check::ok(
            "host_memory",
            format!("{bytes} bytes total ({source})"),
            Some(json!({ "total_bytes": bytes, "source": mem_path })),
        )),
        Err(error) => checks.push(Check::failed(
            "host_memory",
            format!("{error} ({source})"),
            Some(json!({ "source": mem_path })),
        )),
    }

    let block_path = format!("{base}/sys/block");
    match read_nvme(&block_path) {
        Ok(devices) => {
            let detail = format!("{} NVMe device(s) visible ({source})", devices.len());
            checks.push(Check::ok(
                "host_nvme",
                detail,
                Some(json!({ "devices": devices, "source": block_path })),
            ));
        }
        Err(error) => checks.push(Check::skipped(
            "host_nvme",
            format!("{error} ({source})"),
            Some(json!({ "source": block_path })),
        )),
    }

    checks
}

fn read_cpu(path: &str) -> Result<(String, u64), String> {
    let text =
        std::fs::read_to_string(path).map_err(|error| format!("cannot read {path}: {error}"))?;
    let mut model = None;
    let mut logical = 0_u64;
    for line in text.lines() {
        if let Some((key, value)) = line.split_once(':') {
            match key.trim() {
                "model name" if model.is_none() => {
                    model = Some(value.trim().to_owned());
                }
                "processor" => logical += 1,
                _ => {}
            }
        }
    }
    let model = model.ok_or_else(|| format!("{path} has no model name"))?;
    if logical == 0 {
        return Err(format!("{path} has no processor entries"));
    }
    Ok((model, logical))
}

fn read_mem_total(path: &str) -> Result<u64, String> {
    let text =
        std::fs::read_to_string(path).map_err(|error| format!("cannot read {path}: {error}"))?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let value = rest
                .trim()
                .strip_suffix("kB")
                .ok_or_else(|| format!("unexpected MemTotal line: {line:?}"))?
                .trim();
            let kib: u64 = value
                .parse()
                .map_err(|error| format!("MemTotal value {value:?}: {error}"))?;
            return kib
                .checked_mul(1024)
                .ok_or_else(|| "MemTotal overflows u64".to_owned());
        }
    }
    Err(format!("{path} has no MemTotal"))
}

fn read_nvme(path: &str) -> Result<Vec<Value>, String> {
    let entries =
        std::fs::read_dir(path).map_err(|error| format!("cannot read {path}: {error}"))?;
    let mut devices = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| format!("{path}: {error}"))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.starts_with("nvme") {
            continue;
        }
        let base = format!("{path}/{name}");
        let size_sectors = std::fs::read_to_string(format!("{base}/size"))
            .ok()
            .and_then(|text| text.trim().parse::<u64>().ok());
        let rotational = std::fs::read_to_string(format!("{base}/queue/rotational"))
            .ok()
            .and_then(|text| text.trim().parse::<u8>().ok());
        let model = std::fs::read_to_string(format!("{base}/device/model"))
            .ok()
            .map(|text| text.trim().to_owned());
        devices.push(json!({
            "name": name,
            "size_bytes": size_sectors.and_then(|sectors| sectors.checked_mul(512)),
            "rotational": rotational,
            "model": model,
        }));
    }
    devices.sort_by_key(|device| {
        device
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned()
    });
    Ok(devices)
}
