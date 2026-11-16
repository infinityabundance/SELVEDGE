use repo_18_moved_across_modules::{register_and_count, Registry};

#[test]
fn two_names_are_counted() {
    assert_eq!(register_and_count(Registry::new(), &["a", "b"]), 2);
}

#[test]
fn no_names_are_counted() {
    assert_eq!(register_and_count(Registry::new(), &[]), 0);
}

#[test]
fn a_single_name_is_counted() {
    assert_eq!(register_and_count(Registry::new(), &["a"]), 1);
}
