use deps_08_renamed_package::normalise;

#[test]
fn trims_and_upper_cases() {
    assert_eq!(normalise("  alpha  "), "ALPHA");
}

#[test]
fn empty_stays_empty() {
    assert_eq!(normalise("   "), "");
}

#[test]
fn mixed_case_is_upper_cased() {
    assert_eq!(normalise("Mixed"), "MIXED");
}
