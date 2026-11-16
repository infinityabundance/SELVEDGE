use sealed_borrow_02_swap_two_elements::swap_ends;

#[test]
fn swaps_the_ends() {
    let mut values = vec![1, 2, 3, 4];
    swap_ends(&mut values);
    assert_eq!(values, vec![4, 2, 3, 1]);
}

#[test]
fn a_single_element_is_unchanged() {
    let mut values = vec![7];
    swap_ends(&mut values);
    assert_eq!(values, vec![7]);
}

#[test]
fn an_empty_vector_is_unchanged() {
    let mut values: Vec<i64> = Vec::new();
    swap_ends(&mut values);
    assert!(values.is_empty());
}
