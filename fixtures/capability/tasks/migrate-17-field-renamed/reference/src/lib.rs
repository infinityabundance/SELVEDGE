//! Doubling a measurement.

pub fn doubled(text: &str) -> Option<f64> {
    measure::load(text).map(|measure| measure.amount * 2.0)
}
