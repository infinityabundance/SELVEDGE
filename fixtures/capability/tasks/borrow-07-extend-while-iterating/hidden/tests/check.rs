use borrow_07_extend_while_iterating::double_evens;

#[test]
fn appends_doubles_in_order() {
    let mut values = vec![1, 2, 3, 4];
    double_evens(&mut values);
    assert_eq!(values, vec![1, 2, 3, 4, 4, 8]);
}

#[test]
fn appends_nothing_without_evens() {
    let mut values = vec![1, 3, 5];
    double_evens(&mut values);
    assert_eq!(values, vec![1, 3, 5]);
}

#[test]
fn negative_evens_count() {
    let mut values = vec![-2, 1];
    double_evens(&mut values);
    assert_eq!(values, vec![-2, 1, -4]);
}

#[test]
fn empty_stays_empty() {
    let mut values: Vec<i32> = Vec::new();
    double_evens(&mut values);
    assert!(values.is_empty());
}
