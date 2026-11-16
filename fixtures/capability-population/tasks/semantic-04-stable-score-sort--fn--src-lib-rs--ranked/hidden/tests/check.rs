use semantic_04_stable_score_sort::ranked;

#[test]
fn orders_by_score_descending() {
    let entries = vec![("a".to_string(), 1), ("b".to_string(), 3)];
    assert_eq!(ranked(&entries), vec!["b", "a"]);
}

#[test]
fn ties_break_by_name_ascending() {
    let entries = vec![("z".to_string(), 5), ("a".to_string(), 5)];
    assert_eq!(ranked(&entries), vec!["a", "z"]);
}

#[test]
fn ties_with_many_entries_are_deterministic() {
    let entries = vec![
        ("c".to_string(), 2),
        ("a".to_string(), 2),
        ("b".to_string(), 2),
    ];
    assert_eq!(ranked(&entries), vec!["a", "b", "c"]);
}

#[test]
fn empty_is_empty() {
    assert!(ranked(&[]).is_empty());
}
