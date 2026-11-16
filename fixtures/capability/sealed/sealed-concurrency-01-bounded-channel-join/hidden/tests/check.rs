use sealed_concurrency_01_bounded_channel_join::collect_after_join;

#[test]
fn three_messages_are_collected_in_order() {
    assert_eq!(collect_after_join(3), vec![0, 1, 2]);
}

#[test]
fn one_message_is_collected() {
    assert_eq!(collect_after_join(1), vec![0]);
}

#[test]
fn no_messages_is_empty() {
    assert!(collect_after_join(0).is_empty());
}
