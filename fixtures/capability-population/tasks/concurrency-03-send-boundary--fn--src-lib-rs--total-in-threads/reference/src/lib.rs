//! Sum a vector across worker threads with a thread-safe pointer.

use std::sync::Arc;

pub fn total_in_threads(data: Vec<i64>) -> i64 {
    if data.is_empty() {
        return 0;
    }
    let shared = Arc::new(data);
    let workers = 4.min(shared.len());
    let chunk = shared.len().div_ceil(workers);
    let handles: Vec<_> = (0..workers)
        .map(|index| {
            let part = Arc::clone(&shared);
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

#[cfg(test)]
mod visible {
    use crate::total_in_threads;

    #[test]
    fn sums_every_element() {
        assert_eq!(total_in_threads(vec![1, 2, 3, 4, 5]), 15);
    }

    #[test]
    fn empty_is_zero() {
        assert_eq!(total_in_threads(vec![]), 0);
    }

    #[test]
    fn one_element() {
        assert_eq!(total_in_threads(vec![7]), 7);
    }

    #[test]
    fn negative_values() {
        assert_eq!(total_in_threads(vec![-4, 1, 3]), 0);
    }
}
