use deps_09_disable_default_features::render;

#[test]
fn trims_and_upper_cases() {
    assert_eq!(render("  alpha  "), "ALPHA");
}

#[test]
fn empty_stays_empty() {
    assert_eq!(render("   "), "");
}

#[test]
fn mixed_case_is_upper_cased() {
    assert_eq!(render("Mixed"), "MIXED");
}
