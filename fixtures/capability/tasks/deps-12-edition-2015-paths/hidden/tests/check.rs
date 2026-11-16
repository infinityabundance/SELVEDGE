use deps_12_edition_2015_paths::label_of;

#[test]
fn the_dependency_is_reachable() {
    assert_eq!(label_of(), "textkit 2.0.0");
}
