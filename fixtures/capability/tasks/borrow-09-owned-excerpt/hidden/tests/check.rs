use borrow_09_owned_excerpt::excerpt;

#[test]
fn trims_and_upper_cases() {
    assert_eq!(excerpt("  hello  ").text(), "HELLO");
}

#[test]
fn empty_source_is_empty() {
    assert_eq!(excerpt("   ").text(), "");
}

#[test]
fn excerpts_are_independent() {
    let first = excerpt("one");
    let second = excerpt("two");
    assert_eq!(first.text(), "ONE");
    assert_eq!(second.text(), "TWO");
}

#[test]
fn the_field_is_owned_and_readable() {
    let value = excerpt("mixed Case");
    assert_eq!(value.text, "MIXED CASE");
}
