//! Units, version 2. `parse` is now `parse_value`.

pub fn parse_value(input: &str) -> Option<f64> {
    input.trim().parse().ok()
}
