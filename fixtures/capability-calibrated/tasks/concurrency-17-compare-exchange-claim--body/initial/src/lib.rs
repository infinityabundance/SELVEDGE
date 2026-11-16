//! A one-shot claim.

use std::sync::atomic::{AtomicBool, Ordering};

pub fn claim(flag: &AtomicBool) -> bool {
    todo!()
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
