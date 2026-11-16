//! Sort floats ascending with NaN ordered last.

pub fn sort_ascending(values: &mut [f64]) {
    values.sort_by(|a, b| match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => a.partial_cmp(b).expect("finite values compare"),
    });
}
