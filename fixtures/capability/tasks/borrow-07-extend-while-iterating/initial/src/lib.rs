//! Double every even value, appending the doubles.

pub fn double_evens(values: &mut Vec<i32>) {
    for value in values.iter() {
        if value % 2 == 0 {
            values.push(*value * 2);
        }
    }
}
