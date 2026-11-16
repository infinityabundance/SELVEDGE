use sealed_repo_01_private_method::{describe, Config};

#[test]
fn the_report_reads_the_value() {
    assert_eq!(describe(&Config::new(7)), 7);
}

#[test]
fn the_constructor_still_works() {
    assert_eq!(Config::new(3).value, 3);
}
