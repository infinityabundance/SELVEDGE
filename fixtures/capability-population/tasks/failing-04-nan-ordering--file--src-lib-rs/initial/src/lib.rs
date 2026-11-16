//! Sort floats ascending with NaN ordered last.

pub fn sort_ascending(values: &mut [f64]) {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
}

#[cfg(test)]
mod tests {
    use super::sort_ascending;

    #[test]
    fn nan_sorts_last() {
        let mut values = [3.5, f64::NAN, 1.5, 2.5];
        sort_ascending(&mut values);
        assert_eq!(&values[..3], &[1.5, 2.5, 3.5]);
        assert!(values[3].is_nan());
    }
}
