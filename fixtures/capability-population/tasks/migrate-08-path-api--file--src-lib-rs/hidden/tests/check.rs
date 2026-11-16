use migrate_08_path_api::stem_of;

#[test]
fn takes_the_stem_of_a_path() {
    assert_eq!(stem_of("a/b.txt"), Some("b".to_string()));
}

#[test]
fn a_bare_name_is_its_own_stem() {
    assert_eq!(stem_of("file"), Some("file".to_string()));
}

#[test]
fn only_the_last_extension_is_removed() {
    assert_eq!(stem_of("a/b.tar.gz"), Some("b.tar".to_string()));
}

#[test]
fn a_root_has_no_stem() {
    assert_eq!(stem_of("/"), None);
}
