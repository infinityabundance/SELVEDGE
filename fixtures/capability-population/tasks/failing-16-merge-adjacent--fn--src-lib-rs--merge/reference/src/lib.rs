//! Interval merging.

pub fn merge(ranges: &[(i32, i32)]) -> Vec<(i32, i32)> {
    let mut sorted = ranges.to_vec();
    sorted.sort_by_key(|range| range.0);
    let mut out: Vec<(i32, i32)> = Vec::new();
    for range in sorted {
        match out.last_mut() {
            Some(current) if range.0 <= current.1 => {
                if range.1 > current.1 {
                    current.1 = range.1;
                }
            }
            _ => out.push(range),
        }
    }
    out
}

#[cfg(test)]
mod visible {
    use crate::merge;

    #[test]
    fn touching_intervals_merge() {
        assert_eq!(merge(&[(1, 2), (2, 3)]), vec![(1, 3)]);
    }

    #[test]
    fn overlapping_intervals_merge() {
        assert_eq!(merge(&[(1, 5), (2, 3)]), vec![(1, 5)]);
    }

    #[test]
    fn disjoint_intervals_are_sorted_and_kept() {
        assert_eq!(merge(&[(5, 6), (1, 2)]), vec![(1, 2), (5, 6)]);
    }

    #[test]
    fn empty_input_is_empty() {
        assert!(merge(&[]).is_empty());
    }
}
