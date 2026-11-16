//! Temperature conversion.

#[derive(Debug, PartialEq)]
pub struct Celsius(pub f64);

impl From<f64> for Celsius {
    fn from(value: f64) -> Self {
        Celsius(value)
    }
}

impl From<i32> for Celsius {
    fn from(value: i32) -> Self {
        Celsius(value as f64)
    }
}
