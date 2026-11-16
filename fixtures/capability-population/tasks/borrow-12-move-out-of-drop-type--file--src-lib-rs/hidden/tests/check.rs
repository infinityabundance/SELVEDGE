use borrow_12_move_out_of_drop_type::{into_data, Buffer};

#[test]
fn returns_the_buffered_bytes() {
    assert_eq!(into_data(Buffer::new(vec![1, 2, 3])), vec![1, 2, 3]);
}

#[test]
fn an_empty_buffer_returns_no_bytes() {
    assert!(into_data(Buffer::new(Vec::new())).is_empty());
}

#[test]
fn repeated_hand_offs_are_independent() {
    let first = into_data(Buffer::new(vec![9]));
    let second = into_data(Buffer::new(vec![4, 5]));
    assert_eq!(first, vec![9]);
    assert_eq!(second, vec![4, 5]);
}
