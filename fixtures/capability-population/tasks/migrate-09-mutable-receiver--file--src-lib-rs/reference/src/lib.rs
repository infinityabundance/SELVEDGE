//! Count through the counter crate.

pub fn count_to(limit: u32) -> u32 {
    let mut counter = counter::Counter::new();
    for _ in 0..limit {
        counter.tick();
    }
    counter.value()
}
