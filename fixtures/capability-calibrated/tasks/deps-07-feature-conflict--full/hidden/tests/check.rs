use deps_07_feature_conflict::selected;

#[test]
fn selects_the_safe_mode() {
    assert_eq!(selected(), "safe");
}

#[test]
fn the_selection_is_stable() {
    assert_eq!(selected(), selected());
}

#[test]
fn the_fast_mode_is_not_selected() {
    assert_ne!(selected(), "fast");
}
