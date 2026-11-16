use deps_04_path_version_requirement::{dependency_label, twice};

#[test]
fn doubles_values() {
    assert_eq!(twice(21), 42);
}

#[test]
fn doubles_negatives() {
    assert_eq!(twice(-4), -8);
}

#[test]
fn doubles_zero() {
    assert_eq!(twice(0), 0);
}

#[test]
fn uses_the_vendored_version() {
    assert_eq!(dependency_label(), "mathx 2.0.0");
}
