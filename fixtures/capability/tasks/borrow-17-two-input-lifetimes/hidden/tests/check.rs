use borrow_17_two_input_lifetimes::longer;

#[test]
fn returns_the_longer_text() {
    assert_eq!(longer("ab", "cde"), "cde");
}

#[test]
fn a_tie_returns_the_first_argument() {
    assert_eq!(longer("ab", "cd"), "ab");
}

#[test]
fn an_empty_first_argument_yields_the_second() {
    assert_eq!(longer("", "x"), "x");
}

#[test]
fn the_result_can_be_used_after_both_inputs_are_read() {
    let left = String::from("left");
    let right = String::from("longer");
    let chosen = longer(&left, &right);
    assert_eq!(chosen.len(), 6);
}
