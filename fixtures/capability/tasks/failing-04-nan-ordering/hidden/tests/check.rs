use failing_04_nan_ordering::sort_ascending;

#[test]
fn sorts_finite_values() {
    let mut values = [3.0, 1.0, 2.0];
    sort_ascending(&mut values);
    assert_eq!(values, [1.0, 2.0, 3.0]);
}

#[test]
fn nan_sorts_after_finite_values() {
    let mut values = [3.0, f64::NAN, 1.0, 2.0];
    sort_ascending(&mut values);
    assert_eq!(&values[..3], &[1.0, 2.0, 3.0]);
    assert!(values[3].is_nan());
}

#[test]
fn many_nans_stay_last() {
    let mut values = [f64::NAN, 2.0, f64::NAN, 1.0];
    sort_ascending(&mut values);
    assert_eq!(&values[..2], &[1.0, 2.0]);
    assert!(values[2].is_nan() && values[3].is_nan());
}

#[test]
fn all_nan_is_stable_in_content() {
    let mut values = [f64::NAN, f64::NAN];
    sort_ascending(&mut values);
    assert!(values.iter().all(|value| value.is_nan()));
}
