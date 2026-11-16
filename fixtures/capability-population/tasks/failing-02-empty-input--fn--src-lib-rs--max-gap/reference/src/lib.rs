//! Largest gap between consecutive values.

pub fn max_gap(values: &[i64]) -> i64 {
    values
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod visible {
    use crate::max_gap;

    #[test]
    fn empty_slice() {
        assert_eq!(max_gap(&[]), 0);
    }

    #[test]
    fn single_value() {
        assert_eq!(max_gap(&[5]), 0);
    }

    #[test]
    fn largest_gap_wins() {
        assert_eq!(max_gap(&[1, 4, 2, 9]), 7);
    }

    #[test]
    fn negative_gaps_are_reported() {
        assert_eq!(max_gap(&[9, 1]), -8);
    }
}
