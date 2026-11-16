//! Count from several threads. The crate does not compile: `Cell` is
//! not `Sync`.

use std::cell::Cell;

pub fn count_with(counter: &Cell<u64>, threads: usize) -> u64 {
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                counter.set(counter.get() + 1);
            });
        }
    });
    counter.get()
}
