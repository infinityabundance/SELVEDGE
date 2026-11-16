//! Scaling, version 2. `scale` takes an `f64` factor; version 1 took an
//! `i64` percentage.

pub fn scale(value: f64, factor: f64) -> f64 {
    value * factor
}

pub fn label() -> &'static str {
    "scaling 2.0.0"
}
