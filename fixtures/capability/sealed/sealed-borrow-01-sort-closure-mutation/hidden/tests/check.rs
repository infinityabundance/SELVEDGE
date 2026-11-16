use sealed_borrow_01_sort_closure_mutation::sort_and_count_duplicates;

#[test]
fn counts_duplicates_and_sorts() {
    let mut values = vec![4, 1, 4, 2, 4];
    assert_eq!(sort_and_count_duplicates(&mut values), 2);
    assert_eq!(values, vec![1, 2, 4, 4, 4]);
}

#[test]
fn a_distinct_vector_has_no_duplicates() {
    let mut values = vec![5, 2, 9];
    assert_eq!(sort_and_count_duplicates(&mut values), 0);
    assert_eq!(values, vec![2, 5, 9]);
}

#[test]
fn an_empty_vector_is_unchanged() {
    let mut values: Vec<i64> = Vec::new();
    assert_eq!(sort_and_count_duplicates(&mut values), 0);
    assert!(values.is_empty());
}
