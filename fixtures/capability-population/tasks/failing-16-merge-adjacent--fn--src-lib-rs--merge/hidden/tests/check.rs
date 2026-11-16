use failing_16_merge_adjacent::merge;

#[test]
fn touching_intervals_merge() {
    assert_eq!(merge(&[(1, 2), (2, 3)]), vec![(1, 3)]);
}

#[test]
fn overlapping_intervals_merge() {
    assert_eq!(merge(&[(1, 5), (2, 3)]), vec![(1, 5)]);
}

#[test]
fn disjoint_intervals_are_sorted_and_kept() {
    assert_eq!(merge(&[(5, 6), (1, 2)]), vec![(1, 2), (5, 6)]);
}

#[test]
fn empty_input_is_empty() {
    assert!(merge(&[]).is_empty());
}
