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

#[cfg(test)]
mod visible {
    use crate::last_index;

    #[test]
    fn finds_the_last_of_several_duplicates() {
        assert_eq!(last_index(&[1, 2, 2, 2, 3], 2), Some(3));
    }

    #[test]
    fn finds_a_unique_value() {
        assert_eq!(last_index(&[1, 3, 5], 5), Some(2));
    }

    #[test]
    fn a_missing_value_is_none() {
        assert_eq!(last_index(&[1, 2, 3], 4), None);
    }

    #[test]
    fn an_empty_slice_is_none() {
        assert_eq!(last_index(&[], 0), None);
    }
}
