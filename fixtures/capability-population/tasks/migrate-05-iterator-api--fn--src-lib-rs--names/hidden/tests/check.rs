use migrate_05_iterator_api::names;

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
