//! Sum of squares, computed twice by two workers.

use std::sync::Arc;
use std::thread;

pub fn square_sum(values: Vec<i32>) -> i32 {
    let values = Arc::new(values);
    let first = {
        let values = Arc::clone(&values);
        thread::spawn(move || {
            values.iter().map(|value| value * value).sum::<i32>()
        })
    };
    let second = {
        let values = Arc::clone(&values);
        thread::spawn(move || {
            values.iter().map(|value| value * value).sum::<i32>()
        })
    };
    let left = first.join().unwrap();
    let right = second.join().unwrap();
    assert_eq!(left, right);
    left
}
