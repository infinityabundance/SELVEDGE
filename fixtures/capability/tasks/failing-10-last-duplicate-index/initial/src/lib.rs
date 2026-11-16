//! Last index of a value. The visible test fails: the search stops at
//! the first match.

pub fn last_index(values: &[i32], target: i32) -> Option<usize> {
    let mut low = 0usize;
    let mut high = values.len();
    while low < high {
        let middle = low + (high - low) / 2;
        if values[middle] == target {
            return Some(middle);
        }
        if values[middle] < target {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::last_index;

    #[test]
    fn visible_last_position() {
        assert_eq!(last_index(&[1, 2, 2], 2), Some(2));
    }
}
