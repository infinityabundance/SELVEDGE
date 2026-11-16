//! Percentage scaling through the vendored crate.

pub fn scale_percent(value: f64, percent: i64) -> f64 {
    scaling::scale(value, percent as f64 / 100.0)
}

pub fn dependency_label() -> &'static str {
    scaling::label()
}

#[cfg(test)]
mod visible {
    use crate::{dependency_label, scale_percent};

    #[test]
    fn scales_by_percentage() {
        assert_eq!(scale_percent(200.0, 50), 100.0);
    }

    #[test]
    fn zero_percent_is_zero() {
        assert_eq!(scale_percent(10.0, 0), 0.0);
    }

    #[test]
    fn fractional_percent() {
        assert!((scale_percent(-5.0, 10) - (-0.5)).abs() < 1e-12);
    }

    #[test]
    fn the_vendored_version_is_the_one_in_use() {
        assert_eq!(dependency_label(), "scaling 2.0.0");
    }
}
