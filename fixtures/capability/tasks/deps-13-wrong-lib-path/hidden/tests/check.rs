use deps_13_wrong_lib_path::answer;

#[test]
fn the_library_is_found() {
    assert_eq!(answer(), 42);
}
