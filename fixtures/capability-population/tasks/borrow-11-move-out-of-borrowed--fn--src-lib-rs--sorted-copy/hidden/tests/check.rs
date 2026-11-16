use borrow_11_move_out_of_borrowed::sorted_copy;

#[test]
fn returns_the_elements_in_order() {
    let values = vec!["beta".to_string(), "alpha".to_string()];
    assert_eq!(sorted_copy(&values), vec!["alpha".to_string(),
                                         "beta".to_string()]);
}

#[test]
fn the_caller_keeps_every_element() {
    let values = vec!["b".to_string(), "a".to_string()];
    let _copy = sorted_copy(&values);
    assert_eq!(values, vec!["b".to_string(), "a".to_string()]);
}

#[test]
fn duplicates_are_kept() {
    let values = vec!["x".to_string(), "x".to_string()];
    assert_eq!(sorted_copy(&values).len(), 2);
}
