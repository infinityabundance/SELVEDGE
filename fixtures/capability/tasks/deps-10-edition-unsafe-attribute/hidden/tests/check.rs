use deps_10_edition_unsafe_attribute::answer;

#[test]
fn the_exported_function_answers() {
    assert_eq!(answer(), 42);
}
