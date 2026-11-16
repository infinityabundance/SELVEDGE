//! Flooring mean.

pub fn mean_floor<T>(items: &[T]) -> i64 {
    if items.is_empty() {
        return 0;
    }
    let total: i64 = items.iter().map(|value| value.as_i64()).sum();
    total / items.len() as i64
}
