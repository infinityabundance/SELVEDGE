//! Sum two-element chunks in parallel.

use std::thread;

pub fn sum_parallel(values: &[i32]) -> i32 {
    let handles: Vec<_> = values
        .chunks(2)
        .map(|chunk| thread::spawn(move || chunk.iter().sum::<i32>()))
        .collect();
    handles.into_iter().map(|handle| handle.join().unwrap()).sum()
}
