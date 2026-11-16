//! Integer averages, currently truncated.

pub fn average(values: &[u32]) -> u32 {
    if values.is_empty() {
        return 0;
    }
    values.iter().sum::<u32>() / values.len() as u32
}

#[cfg(test)]
mod tests {
    use super::average;

    #[test]
    fn exact_mean_is_unchanged() {
        assert_eq!(average(&[2, 4]), 3);
    }
}
