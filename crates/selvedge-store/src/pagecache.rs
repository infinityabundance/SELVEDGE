//! Page-cache control.
//!
//! This is the only module in the crate that talks to the kernel about caching,
//! and it does so through `rustix`, which exposes `posix_fadvise` as a safe
//! wrapper. There is no `unsafe` code and no `O_DIRECT` path: a correct direct
//! read needs a page-aligned buffer, which would require an aligned allocator
//! and therefore unsafe code or a new dependency. The cold control is
//! `POSIX_FADV_DONTNEED` plus a device-counter check, and it is reported as an
//! advisory action, not as proof that a page was dropped.
//!
//! `FADV_DONTNEED` drops clean, unreferenced pages of the named file. A page
//! still mapped by another process, or dirtied and not yet written back, is not
//! dropped. The measurement commands therefore also record the device's sector
//! counter before and after each cold run, which shows whether the reads came
//! from the device.

use std::fs::File;
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rustix::fs::{fadvise, Advice};
use serde::Serialize;

use crate::error::{Result, StoreError};

/// Outcome of an advisory action.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AdviceReport {
    /// Path the advice applies to.
    pub path: PathBuf,
    /// Advice name.
    pub advice: &'static str,
    /// Bytes covered.
    pub bytes: u64,
    /// Time spent issuing the advice.
    pub elapsed_ns: u64,
    /// Always `advisory`; the kernel is not obliged to drop a page.
    pub strength: &'static str,
}

/// Page-cache advice helpers.
#[derive(Debug)]
pub struct PageCache;

impl PageCache {
    /// Ask the kernel to drop this file's clean cached pages.
    ///
    /// # Errors
    ///
    /// Rejects a file that cannot be opened, or a failing `posix_fadvise`.
    pub fn drop_file(path: &Path) -> Result<AdviceReport> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        let start = Instant::now();
        fadvise(&file, 0, None, Advice::DontNeed)?;
        let elapsed_ns = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        Ok(AdviceReport {
            path: path.to_path_buf(),
            advice: "posix_fadvise_dontneed",
            bytes: len,
            elapsed_ns,
            strength: "advisory",
        })
    }

    /// Ask the kernel to drop cached pages of a byte range.
    ///
    /// # Errors
    ///
    /// Rejects a zero length or a failing `posix_fadvise`.
    pub fn drop_range(file: &File, offset: u64, len: u64) -> Result<()> {
        let Some(len) = NonZeroU64::new(len) else {
            return Err(StoreError::config("fadvise range length is zero"));
        };
        fadvise(file, offset, Some(len), Advice::DontNeed)?;
        Ok(())
    }

    /// Tell the kernel a range will be read soon.
    ///
    /// # Errors
    ///
    /// Rejects a zero length or a failing `posix_fadvise`.
    pub fn will_need_range(file: &File, offset: u64, len: u64) -> Result<()> {
        let Some(len) = NonZeroU64::new(len) else {
            return Err(StoreError::config("fadvise range length is zero"));
        };
        fadvise(file, offset, Some(len), Advice::WillNeed)?;
        Ok(())
    }

    /// Advise sequential access for the remainder of the file.
    ///
    /// # Errors
    ///
    /// Propagates a failing `posix_fadvise`.
    pub fn sequential(file: &File, offset: u64) -> Result<()> {
        fadvise(file, offset, None, Advice::Sequential)?;
        Ok(())
    }

    /// Advise random access for the remainder of the file.
    ///
    /// # Errors
    ///
    /// Propagates a failing `posix_fadvise`.
    pub fn random(file: &File, offset: u64) -> Result<()> {
        fadvise(file, offset, None, Advice::Random)?;
        Ok(())
    }
}
