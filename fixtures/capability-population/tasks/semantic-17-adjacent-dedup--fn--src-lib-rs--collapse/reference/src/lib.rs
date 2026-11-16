//! Collapsing adjacent duplicates.

pub fn collapse(values: &[u8]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    for value in values {
        if out.last() != Some(value) {
            out.push(*value);
        }
    }
    out
}

#[cfg(test)]
mod visible {
    use crate::collapse;

    #[test]
    fn a_later_repeat_is_kept() {
        assert_eq!(collapse(&[1, 2, 1]), vec![1, 2, 1]);
    }

    #[test]
    fn adjacent_duplicates_are_removed() {
        assert_eq!(collapse(&[1, 1, 2]), vec![1, 2]);
    }

    #[test]
    fn long_runs_collapse_to_one() {
        assert_eq!(collapse(&[3, 3, 3, 3]), vec![3]);
    }

    #[test]
    fn empty_input_is_empty() {
        assert!(collapse(&[]).is_empty());
    }
}
