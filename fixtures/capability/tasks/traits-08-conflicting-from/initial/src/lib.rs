//! Temperature conversion with an overlapping blanket impl.

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

impl<T: Into<f64>> From<T> for Celsius {
    fn from(value: T) -> Self {
        Celsius(value.into())
    }
}
