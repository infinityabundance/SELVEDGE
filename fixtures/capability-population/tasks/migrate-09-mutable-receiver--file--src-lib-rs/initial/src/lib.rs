//! Count through the counter crate. This does not build: version 2
//! requires a mutable receiver for `tick`.

pub fn count_to(limit: u32) -> u32 {
    let counter = counter::Counter::new();
    for _ in 0..limit {
        counter.tick();
    }
    counter.value()
}
