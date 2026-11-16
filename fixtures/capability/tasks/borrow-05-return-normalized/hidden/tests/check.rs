use borrow_05_return_normalized::normalize;

#[test]
fn collapses_internal_runs() {
    assert_eq!(normalize("  alpha   beta  "), "alpha beta");
}

#[test]
fn handles_tabs_and_newlines() {
    assert_eq!(normalize("a\tb\nc"), "a b c");
}

#[test]
fn empty_and_blank_inputs_are_empty() {
    assert_eq!(normalize(""), "");
    assert_eq!(normalize("   \t\n"), "");
}

#[test]
fn successive_calls_are_independent() {
    let first = normalize("  one  ");
    let second = normalize("two");
    assert_eq!(first, "one");
    assert_eq!(second, "two");
}
