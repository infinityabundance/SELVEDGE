//! Last index of a value.

pub fn last_index(values: &[i32], target: i32) -> Option<usize> {
    let mut low = 0usize;
    let mut high = values.len();
    while low < high {
        let middle = low + (high - low) / 2;
        if values[middle] <= target {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    if low > 0 && values[low - 1] == target {
        Some(low - 1)
    } else {
        None
    }
}
