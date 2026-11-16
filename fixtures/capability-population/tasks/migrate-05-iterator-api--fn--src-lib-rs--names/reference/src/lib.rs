//! Collect item names through the item crate.

pub fn names(input: &str) -> Vec<String> {
    let mut names = Vec::new();
    items::scan(input, &mut |item| names.push(item.name.to_string()));
    names
}

#[cfg(test)]
mod visible {
    use crate::names;

    #[test]
    fn visits_every_segment() {
        assert_eq!(names("a,b"), vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn empty_input_is_empty() {
        assert!(names("").is_empty());
    }

    #[test]
    fn empty_segments_are_skipped() {
        assert_eq!(names("a,,b"), vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn segments_are_not_trimmed() {
        assert_eq!(names("a, b"), vec!["a".to_string(), " b".to_string()]);
    }
}
