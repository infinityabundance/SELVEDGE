//! A one-shot claim.

use std::sync::atomic::{AtomicBool, Ordering};

pub fn claim(flag: &AtomicBool) -> bool {
    flag.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok()
}
