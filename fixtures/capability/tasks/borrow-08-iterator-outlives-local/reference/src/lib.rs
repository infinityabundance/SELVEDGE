//! Even numbers below a limit.

pub fn evens(limit: i32) -> impl Iterator<Item = i32> {
    (0..limit).filter(|value| value % 2 == 0)
}
