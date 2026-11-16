//! A one-shot claim.

use std::sync::atomic::{AtomicBool, Ordering};

// TODO(FUNCTION)

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
