//! Scale, version 2. The input is generic over `Into<f64>`.

pub fn scale<T: Into<f64>>(value: T) -> f64 {
    value.into()
}
