use deps_15_missing_package_name::answer;

#[test]
fn the_package_is_named() {
    assert_eq!(answer(), 42);
}
