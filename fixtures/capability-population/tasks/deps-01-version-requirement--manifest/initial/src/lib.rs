//! Percentage scaling through the vendored crate.

pub fn scale_percent(value: f64, percent: i64) -> f64 {
    scaling::scale(value, percent as f64 / 100.0)
}

pub fn dependency_label() -> &'static str {
    scaling::label()
}
