//! Tie-to-even rounding.

pub fn round_half_even(value: f64) -> f64 {
    let floor = value.floor();
    let fraction = value - floor;
    if (fraction - 0.5).abs() < 1e-12 {
        if (floor as i64) % 2 == 0 {
            floor
        } else {
            floor + 1.0
        }
    } else {
        value.round()
    }
}

#[cfg(test)]
mod visible {
    use crate::round_half_even;

    #[test]
    fn positive_ties_go_to_even() {
        assert_eq!(round_half_even(2.5), 2.0);
        assert_eq!(round_half_even(3.5), 4.0);
    }

    #[test]
    fn negative_ties_go_to_even() {
        assert_eq!(round_half_even(-2.5), -2.0);
        assert_eq!(round_half_even(-3.5), -4.0);
    }

    #[test]
    fn non_ties_round_normally() {
        assert_eq!(round_half_even(2.4), 2.0);
        assert_eq!(round_half_even(2.6), 3.0);
        assert_eq!(round_half_even(-2.6), -3.0);
    }

    #[test]
    fn integers_are_unchanged() {
        assert_eq!(round_half_even(7.0), 7.0);
    }
}
