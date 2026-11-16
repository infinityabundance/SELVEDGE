//! Increment every entry in place.

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use super::*;

    #[test]
    fn increments_every_entry_and_sums() {
        let mut entries = vec![1, 2];
        assert_eq!(stamp(&mut entries), 5);
        assert_eq!(entries, vec![2, 3]);
    }
}
