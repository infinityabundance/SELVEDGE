//! Sum a vector across four worker threads.

use std::rc::Rc;

pub fn total_in_threads(data: Vec<i64>) -> i64 {
    if data.is_empty() {
        return 0;
    }
    let shared = Rc::new(data);
    let workers = 4.min(shared.len());
    let chunk = shared.len().div_ceil(workers);
    let handles: Vec<_> = (0..workers)
        .map(|index| {
            let part = Rc::clone(&shared);
            std::thread::spawn(move || {
                let start = (index * chunk).min(part.len());
                let end = ((index + 1) * chunk).min(part.len());
                part[start..end].iter().copied().sum::<i64>()
            })
        })
        .collect();
    handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .sum()
}
