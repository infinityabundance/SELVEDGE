//! Join a list and measure it.

pub fn join_then_measure(parts: Vec<String>) -> (String, usize) {
    let joined = parts.join("-");
    let count = parts.len();
    let first = parts.into_iter().next().unwrap_or_default();
    (format!("{joined}:{first}"), count)
}

#[cfg(test)]
mod visible {
    use crate::join_then_measure;

    #[test]
    fn joins_and_counts() {
        let parts = vec!["a".to_string(), "b".to_string()];
        assert_eq!(join_then_measure(parts),
                   ("a-b:a".to_string(), 2));
    }

    #[test]
    fn an_empty_list_has_an_empty_first_part() {
        assert_eq!(join_then_measure(Vec::new()), (":".to_string(), 0));
    }

    #[test]
    fn a_single_part_is_joined_with_itself() {
        let parts = vec!["x".to_string()];
        assert_eq!(join_then_measure(parts), ("x:x".to_string(), 1));
    }
}
