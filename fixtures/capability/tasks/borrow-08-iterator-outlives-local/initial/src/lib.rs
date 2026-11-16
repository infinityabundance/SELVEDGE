//! Even numbers below a limit.

pub fn evens(limit: i32) -> impl Iterator<Item = i32> {
    let values: Vec<i32> = (0..limit).collect();
    values.iter().copied().filter(|value| *value % 2 == 0)
}
