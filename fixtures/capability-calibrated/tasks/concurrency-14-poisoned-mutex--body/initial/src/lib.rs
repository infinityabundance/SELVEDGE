//! Record bytes and survive a worker panic.

use std::sync::Mutex;

pub fn record_with_panic(shared: &Mutex<Vec<u8>>) -> usize {
    todo!()
}

#[cfg(test)]
mod visible {
    use super::*;

    use std::sync::Mutex;
    
    #[test]
    fn the_recorded_bytes_survive_the_panic() {
        let shared = Mutex::new(Vec::new());
        assert_eq!(record_with_panic(&shared), 1);
    }
}
