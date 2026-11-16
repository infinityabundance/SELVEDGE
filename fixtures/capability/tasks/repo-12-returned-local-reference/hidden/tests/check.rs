use repo_12_returned_local_reference::cached;

#[test]
fn trims_and_uppercases() {
    assert_eq!(cached(" ab "), "AB".to_string());
}

#[test]
fn empty_input_stays_empty() {
    assert_eq!(cached(""), String::new());
}

#[test]
fn successive_calls_are_independent() {
    let first = cached("x");
    let second = cached("y");
    assert_eq!(first, "X");
    assert_eq!(second, "Y");
}
