//! Last index of a value.

pub fn last_index(values: &[i32], target: i32) -> Option<usize> {
    todo!()
}

#[cfg(test)]
mod visible {
    use super::*;

    #[test]
    fn finds_the_last_of_several_duplicates() {
        assert_eq!(last_index(&[1, 2, 2, 2, 3], 2), Some(3));
    }
}
