//! Clamping with the standard two-step form.

pub fn clamp(value: i64, low: i64, high: i64) -> i64 {
    value.max(low).min(high)
}

#[cfg(test)]
mod tests {
    use super::clamp;

    #[test]
    fn visible_inside_the_bounds() {
        assert_eq!(clamp(5, 0, 10), 5);
    }
}
