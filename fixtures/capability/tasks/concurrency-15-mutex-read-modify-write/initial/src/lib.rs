//! A mutex-protected counter. The visible test fails: the read and
//! the write are in separate critical sections.

use std::sync::{Barrier, Mutex};

pub fn bump_pair(counter: &Mutex<u64>) {
    let barrier = Barrier::new(2);
    std::thread::scope(|scope| {
        for _ in 0..2 {
            scope.spawn(|| {
                let value = { *counter.lock().unwrap() };
                barrier.wait();
                let mut guard = counter.lock().unwrap();
                *guard = value + 1;
            });
        }
    });
}

pub fn bump_many(counter: &Mutex<u64>, threads: usize, per_thread: usize) {
    let barrier = Barrier::new(threads);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                for _ in 0..per_thread {
                    let value = { *counter.lock().unwrap() };
                    barrier.wait();
                    let mut guard = counter.lock().unwrap();
                    *guard = value + 1;
                }
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::bump_pair;
    use std::sync::Mutex;

    #[test]
    fn visible_pair() {
        let counter = Mutex::new(0);
        bump_pair(&counter);
        assert_eq!(*counter.lock().unwrap(), 2);
    }
}
