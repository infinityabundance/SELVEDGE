use repo_13_undeclared_module_path::slugify;

#[test]
fn words_are_joined_with_hyphens() {
    assert_eq!(slugify("A B"), "a-b");
}

#[test]
fn surrounding_space_is_trimmed() {
    assert_eq!(slugify("  x  "), "x");
}

#[test]
fn empty_input_is_empty() {
    assert_eq!(slugify(""), "");
}
