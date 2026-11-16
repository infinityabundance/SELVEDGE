//! Sum of squares, computed twice by two workers.

use std::thread;

pub fn square_sum(values: Vec<i32>) -> i32 {
    let first = thread::spawn(move || {
        values.iter().map(|value| value * value).sum::<i32>()
    });
    let second = thread::spawn(move || {
        values.iter().map(|value| value * value).sum::<i32>()
    });
    let left = first.join().unwrap();
    let right = second.join().unwrap();
    assert_eq!(left, right);
    left
}
