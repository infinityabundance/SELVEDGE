//! A one-shot claim. The visible test fails: the compare-exchange
//! compares against the claimed value, not the free one.

use std::sync::atomic::{AtomicBool, Ordering};

pub fn claim(flag: &AtomicBool) -> bool {
    flag.compare_exchange(true, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok()
}

#[cfg(test)]
mod tests {
    use super::claim;
    use std::sync::atomic::AtomicBool;

    #[test]
    fn visible_first_claim() {
        let flag = AtomicBool::new(false);
        assert!(claim(&flag));
    }
}
