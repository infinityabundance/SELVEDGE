//! Sorting through decimal text.

pub fn sorted(values: &[u32]) -> Vec<u32> {
    let mut text: Vec<String> =
        values.iter().map(|value| value.to_string()).collect();
    text.sort();
    text.iter().filter_map(|value| value.parse().ok()).collect()
}

#[cfg(test)]
mod tests {
    use super::sorted;

    #[test]
    fn visible_single_digits() {
        assert_eq!(sorted(&[3, 1, 2]), vec![1, 2, 3]);
    }
}
