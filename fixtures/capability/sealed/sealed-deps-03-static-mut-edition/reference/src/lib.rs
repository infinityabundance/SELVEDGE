//! An atomic counter.

use std::sync::atomic::{AtomicI64, Ordering};

static COUNTER: AtomicI64 = AtomicI64::new(0);

pub fn bump() -> i64 {
    COUNTER.fetch_add(1, Ordering::SeqCst) + 1
}
