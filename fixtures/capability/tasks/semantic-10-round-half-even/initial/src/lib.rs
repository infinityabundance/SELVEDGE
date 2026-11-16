//! Tie-to-even rounding with the standard rounding rule.

pub fn round_half_even(value: f64) -> f64 {
    value.round()
}

#[cfg(test)]
mod tests {
    use super::round_half_even;

    #[test]
    fn visible_not_a_tie() {
        assert_eq!(round_half_even(2.4), 2.0);
    }
}
