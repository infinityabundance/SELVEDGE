use migrate_13_const_generic::make_counter;

#[test]
fn a_fresh_frame_has_no_used_cells() {
    assert_eq!(make_counter().used(), 0);
}

#[test]
fn a_value_can_be_stored() {
    let mut frame = make_counter();
    frame.set(1, 7);
    assert_eq!(frame.used(), 1);
}

#[test]
fn the_frame_has_four_cells() {
    let mut frame = make_counter();
    frame.set(3, 1);
    frame.set(4, 1);
    assert_eq!(frame.used(), 1);
}

#[test]
fn frames_are_independent() {
    let mut first = make_counter();
    let second = make_counter();
    first.set(0, 9);
    assert_eq!(first.used(), 1);
    assert_eq!(second.used(), 0);
}
