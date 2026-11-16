//! Integer averages with half-up rounding.

pub fn average(values: &[u32]) -> u32 {
    if values.is_empty() {
        return 0;
    }
    let len = values.len() as u32;
    (values.iter().sum::<u32>() + len / 2) / len
}

#[cfg(test)]
mod visible {
    use crate::average;

    #[test]
    fn rounds_half_up() {
        assert_eq!(average(&[1, 2]), 2);
    }

    #[test]
    fn exact_means_are_unchanged() {
        assert_eq!(average(&[2, 4]), 3);
    }

    #[test]
    fn rounds_down_below_half() {
        assert_eq!(average(&[1, 2, 3]), 2);
    }

    #[test]
    fn empty_is_zero() {
        assert_eq!(average(&[]), 0);
    }

    #[test]
    fn single_value_is_itself() {
        assert_eq!(average(&[7]), 7);
    }
}
