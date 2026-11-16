use sealed_borrow_03_temporary_split_iterator::first_segment_length;

#[test]
fn measures_the_first_segment() {
    assert_eq!(first_segment_length("alpha,beta"), 5);
}

#[test]
fn a_single_segment_is_measured() {
    assert_eq!(first_segment_length("one"), 3);
}

#[test]
fn an_empty_segment_is_zero() {
    assert_eq!(first_segment_length(""), 0);
    assert_eq!(first_segment_length(",tail"), 0);
}
