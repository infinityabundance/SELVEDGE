use migrate_02_builder::connect;

#[test]
fn builds_an_endpoint() {
    assert_eq!(connect("example.org", 8080, "/x"), Ok("example.org:8080/x".to_string()));
}

#[test]
fn rejects_an_empty_host() {
    assert_eq!(connect("", 80, "/"), Err("empty host".to_string()));
}
