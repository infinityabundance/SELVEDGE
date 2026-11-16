use borrow_14_borrow_after_move::join_then_measure;

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
