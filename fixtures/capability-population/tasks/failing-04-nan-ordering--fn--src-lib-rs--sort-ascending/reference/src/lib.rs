//! Sort floats ascending with NaN ordered last.

pub fn sort_ascending(values: &mut [f64]) {
    values.sort_by(|a, b| match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => a.partial_cmp(b).expect("finite values compare"),
    });
}

#[cfg(test)]
mod visible {
    use crate::sort_ascending;

    #[test]
    fn sorts_finite_values() {
        let mut values = [3.5, 1.5, 2.5];
        sort_ascending(&mut values);
        assert_eq!(values, [1.5, 2.5, 3.5]);
    }

    #[test]
    fn nan_sorts_after_finite_values() {
        let mut values = [3.5, f64::NAN, 1.5, 2.5];
        sort_ascending(&mut values);
        assert_eq!(&values[..3], &[1.5, 2.5, 3.5]);
        assert!(values[3].is_nan());
    }

    #[test]
    fn many_nans_stay_last() {
        let mut values = [f64::NAN, 2.5, f64::NAN, 1.5];
        sort_ascending(&mut values);
        assert_eq!(&values[..2], &[1.5, 2.5]);
        assert!(values[2].is_nan() && values[3].is_nan());
    }

    #[test]
    fn all_nan_is_stable_in_content() {
        let mut values = [f64::NAN, f64::NAN];
        sort_ascending(&mut values);
        assert!(values.iter().all(|value| value.is_nan()));
    }
}
