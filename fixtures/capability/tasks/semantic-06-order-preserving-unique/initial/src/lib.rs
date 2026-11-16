//! De-duplication by sorting, which loses first-occurrence order.

pub fn unique(values: &[i32]) -> Vec<i32> {
    let mut sorted = values.to_vec();
    sorted.sort();
    sorted.dedup();
    sorted
}

#[cfg(test)]
mod tests {
    use super::unique;

    #[test]
    fn removes_duplicates_from_sorted_input() {
        assert_eq!(unique(&[1, 1, 2]), vec![1, 2]);
    }
}
