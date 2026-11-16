//! Doubling through the units crate.

pub fn double(input: &str) -> Option<f64> {
    units::parse_value(input).map(|value| value * 2.0)
}

#[cfg(test)]
mod visible {
    use crate::double;

    #[test]
    fn doubles_a_number() {
        assert_eq!(double("2.5"), Some(5.0));
    }

    #[test]
    fn ignores_surrounding_space() {
        assert_eq!(double(" 3 "), Some(6.0));
    }

    #[test]
    fn rejects_text() {
        assert_eq!(double("x"), None);
    }
}
