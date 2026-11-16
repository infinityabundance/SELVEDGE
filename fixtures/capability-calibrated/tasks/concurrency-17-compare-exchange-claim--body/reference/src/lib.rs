//! A one-shot claim.

use std::sync::atomic::{AtomicBool, Ordering};

pub fn claim(flag: &AtomicBool) -> bool {
    
    flag.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok()

}

#[cfg(test)]
mod visible {
    use super::*;

    use std::sync::atomic::{AtomicBool, Ordering};
    
    #[test]
    fn the_first_claim_wins() {
        let flag = AtomicBool::new(false);
        assert!(claim(&flag));
    }
}
