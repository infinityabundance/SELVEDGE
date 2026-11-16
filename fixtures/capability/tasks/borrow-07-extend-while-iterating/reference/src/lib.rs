//! Double every even value, appending the doubles.

pub fn double_evens(values: &mut Vec<i32>) {
    let additions: Vec<i32> = values
        .iter()
        .filter(|value| *value % 2 == 0)
        .map(|value| value * 2)
        .collect();
    values.extend(additions);
}
