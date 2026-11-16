use deps_03_optional_feature::{backend, fold};

#[test]
fn folds_whitespace_and_case() {
    assert_eq!(fold("  A\tB  "), "a b");
}

#[test]
fn the_fast_backend_is_enabled() {
    assert_eq!(backend(), "fast");
}
