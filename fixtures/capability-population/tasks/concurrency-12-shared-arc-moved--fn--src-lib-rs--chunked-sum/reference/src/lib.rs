//! A chunked sum.

use std::sync::Arc;

pub fn chunked_sum(values: Arc<Vec<u64>>, threads: usize) -> u64 {
    let mut handles = Vec::new();
    for index in 0..threads {
        let values = Arc::clone(&values);
        handles.push(std::thread::spawn(move || {
            values.iter().skip(index).step_by(threads).sum::<u64>()
        }));
    }
    handles.into_iter().map(|handle| handle.join().unwrap()).sum()
}

#[cfg(test)]
mod visible {
    use std::sync::Arc;

    use crate::chunked_sum;

    #[test]
    fn two_workers_cover_every_index() {
        let values = Arc::new(vec![1, 2, 3, 4]);
        assert_eq!(chunked_sum(values, 2), 10);
    }

    #[test]
    fn one_worker_sums_everything() {
        let values = Arc::new(vec![5, 6, 7]);
        assert_eq!(chunked_sum(values, 1), 18);
    }

    #[test]
    fn an_empty_vector_sums_to_zero() {
        let values = Arc::new(Vec::new());
        assert_eq!(chunked_sum(values, 3), 0);
    }
}
