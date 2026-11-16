use semantic_12_split_once::split_key;

#[test]
fn the_value_keeps_later_separators() {
    assert_eq!(split_key("a:b:c"),
               Some(("a".to_string(), "b:c".to_string())));
}

#[test]
fn empty_parts_are_allowed() {
    assert_eq!(split_key("a:"), Some(("a".to_string(), String::new())));
    assert_eq!(split_key(":b"), Some((String::new(), "b".to_string())));
}

#[test]
fn no_separator_is_none() {
    assert_eq!(split_key("abc"), None);
}
