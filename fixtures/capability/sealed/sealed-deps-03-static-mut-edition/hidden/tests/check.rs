use sealed_deps_03_static_mut_edition::bump;

#[test]
fn a_bump_returns_a_positive_value() {
    assert!(bump() >= 1);
}

#[test]
fn successive_bumps_increase() {
    let before = bump();
    let after = bump();
    assert!(after > before);
}

#[test]
fn bumps_are_unique() {
    assert_ne!(bump(), bump());
}
