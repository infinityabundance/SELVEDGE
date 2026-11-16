//! Lower-bound search.

pub fn lower_bound(values: &[i32], target: i32) -> usize {
    let mut low = 0;
    let mut high = values.len();
    while low < high {
        let mid = (low + high) / 2;
        if values[mid] < target {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    low
}
