//! Lower-bound search. The visible test fails: a present value is
//! reported one index too far.

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
    if low < values.len() && values[low] == target {
        low + 1
    } else {
        low
    }
}

#[cfg(test)]
mod tests {
    use super::lower_bound;

    #[test]
    fn finds_a_present_value() {
        assert_eq!(lower_bound(&[1, 3, 5], 3), 1);
    }
}
