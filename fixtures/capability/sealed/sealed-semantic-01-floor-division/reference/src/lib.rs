//! Integer average with floor division.

pub fn mean_floor(values: &[i64]) -> i64 {
    if values.is_empty() {
        return 0;
    }
    let sum: i64 = values.iter().sum();
    sum.div_euclid(values.len() as i64)
}
