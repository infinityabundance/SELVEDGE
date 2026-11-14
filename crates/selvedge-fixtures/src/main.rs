//! `selvedge-fixtures` — regenerate the tracked synthetic bundle.
//!
//! Usage:
//!   cargo run -p selvedge-fixtures -- [--out-dir DIR] [--force|--check]
//!
//! `--check` verifies that the committed artifact matches the generator and
//! writes nothing.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use selvedge_fixtures::{FIXTURE_BUNDLE_ID, tiny_bundle};
use selvedge_format::sidecar::hash_file;
use selvedge_format::{FormatError, WriteOptions, publish_bundle, validate_sidecar};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("selvedge-fixtures: {error}");
            ExitCode::FAILURE
        }
    }
}

#[derive(Debug)]
struct Options {
    out_dir: PathBuf,
    force: bool,
    check: bool,
}

fn parse_args() -> Result<Option<Options>, Box<dyn std::error::Error>> {
    let mut out_dir = PathBuf::from("fixtures");
    let mut positional = false;
    let mut force = false;
    let mut check = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out-dir" => {
                let value = args
                    .next()
                    .ok_or_else(|| usage_error("--out-dir needs a value"))?;
                out_dir = PathBuf::from(value);
            }
            "--force" => force = true,
            "--check" => check = true,
            "-h" | "--help" => {
                print_help();
                return Ok(None);
            }
            other if !other.starts_with('-') => {
                if positional {
                    return Err(usage_error("only one output directory may be given"));
                }
                out_dir = PathBuf::from(other);
                positional = true;
            }
            other => return Err(usage_error(format!("unknown argument {other:?}"))),
        }
    }
    Ok(Some(Options {
        out_dir,
        force,
        check,
    }))
}

fn usage_error(message: impl Into<String>) -> Box<dyn std::error::Error> {
    Box::new(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        message.into(),
    ))
}

fn print_help() {
    println!("usage: selvedge-fixtures [OUT_DIR] [--out-dir DIR] [--force|--check]");
    println!();
    println!("  OUT_DIR        destination directory (default: fixtures)");
    println!("  --force        replace an existing artifact with different bytes");
    println!("  --check        verify the committed artifact; write nothing");
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let Some(options) = parse_args()? else {
        return Ok(());
    };
    let fixture = tiny_bundle()?;
    let encoded = fixture.encoded()?;
    let expected_sha256 = selvedge_core::Sha256Digest::of(&encoded);
    std::fs::create_dir_all(&options.out_dir)?;
    let bundle_path = options.out_dir.join(format!("{FIXTURE_BUNDLE_ID}.slv"));
    let sidecar_path = options
        .out_dir
        .join(format!("{FIXTURE_BUNDLE_ID}.manifest.json"));
    let sums_path = options.out_dir.join("SHA256SUMS");

    if options.check {
        let committed = std::fs::read(&bundle_path)
            .map_err(|err| usage_error(format!("cannot read {}: {err}", bundle_path.display())))?;
        if committed != encoded {
            return Err(usage_error(
                "committed fixture bytes differ from the generator; run without --check to regenerate",
            ));
        }
        validate_sidecar(&bundle_path, &sidecar_path)?;
        println!(
            "check bundle={} sha256={} shards={} payload_bytes={}",
            bundle_path.display(),
            expected_sha256.to_hex(),
            fixture.manifest.shards.len(),
            fixture.manifest.payload_len
        );
        return Ok(());
    }

    let existing = std::fs::read(&bundle_path).ok();
    if let Some(existing) = &existing {
        if existing == &encoded {
            validate_sidecar(&bundle_path, &sidecar_path)?;
            write_sums(&sums_path, &[&bundle_path, &sidecar_path])?;
            println!(
                "unchanged bundle={} sha256={} shards={} payload_bytes={}",
                bundle_path.display(),
                expected_sha256.to_hex(),
                fixture.manifest.shards.len(),
                fixture.manifest.payload_len
            );
            return Ok(());
        }
        if !options.force {
            return Err(usage_error(
                "existing fixture differs from the generator; pass --force to replace it",
            ));
        }
    }

    let options_write = WriteOptions {
        force: true,
        verify_before_publish: true,
        publish_sidecar: true,
        ..WriteOptions::default()
    };
    let published = publish_bundle(
        &options.out_dir,
        &fixture.manifest,
        &fixture.payload,
        &options_write,
    )?;
    write_sums(
        &sums_path,
        &[published.path.as_path(), sidecar_path.as_path()],
    )?;
    println!(
        "wrote bundle={} sha256={} shards={} payload_bytes={} sidecar={}",
        published.path.display(),
        published.bundle_sha256.to_hex(),
        fixture.manifest.shards.len(),
        fixture.manifest.payload_len,
        sidecar_path.display()
    );
    Ok(())
}

fn write_sums(path: &Path, files: &[&Path]) -> Result<(), FormatError> {
    let mut lines = BTreeMap::new();
    for file in files {
        let name = file
            .file_name()
            .ok_or_else(|| FormatError::ManifestInvalid {
                reason: format!("{} has no file name", file.display()),
            })?
            .to_string_lossy()
            .into_owned();
        let digest = hash_file(file)?;
        lines.insert(name, digest.to_hex());
    }
    let mut text = String::new();
    for (name, digest) in lines {
        text.push_str(&format!("{digest}  {name}\n"));
    }
    std::fs::write(path, text)?;
    Ok(())
}
