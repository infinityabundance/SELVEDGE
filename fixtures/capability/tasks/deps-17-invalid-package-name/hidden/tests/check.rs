use deps_17_invalid_package_name::answer;

#[test]
fn the_package_name_is_a_crate_identifier() {
    assert_eq!(answer(), 42);
}
