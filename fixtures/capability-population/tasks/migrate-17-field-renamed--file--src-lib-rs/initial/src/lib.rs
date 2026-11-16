//! Doubling a measurement. This does not build: version 2 renamed the
//! field.

pub fn doubled(text: &str) -> Option<f64> {
    measure::load(text).map(|measure| measure.value * 2.0)
}
