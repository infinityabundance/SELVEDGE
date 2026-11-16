use borrow_13_mutate_through_shared::stamp;

#[test]
fn increments_every_entry_and_sums() {
    let mut entries = vec![1, 2];
    assert_eq!(stamp(&mut entries), 5);
    assert_eq!(entries, vec![2, 3]);
}

#[test]
fn an_empty_vector_sums_to_zero() {
    let mut entries: Vec<u32> = Vec::new();
    assert_eq!(stamp(&mut entries), 0);
    assert!(entries.is_empty());
}

#[test]
fn a_single_entry_is_incremented() {
    let mut entries = vec![41];
    assert_eq!(stamp(&mut entries), 42);
    assert_eq!(entries, vec![42]);
}
