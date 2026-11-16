//! Integer average with the standard division.

pub fn mean_floor(values: &[i64]) -> i64 {
    if values.is_empty() {
        return 0;
    }
    let sum: i64 = values.iter().sum();
    sum / values.len() as i64
}

#[cfg(test)]
mod tests {
    use super::mean_floor;

    #[test]
    fn visible_positive_average() {
        assert_eq!(mean_floor(&[2, 4]), 3);
    }
}
