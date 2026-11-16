use repo_07_field_rename::{read, write, Config};

#[test]
fn exposes_the_new_field() {
    assert_eq!(Config::new("alpha", 2).path, "alpha");
}

#[test]
fn keeps_retries() {
    assert_eq!(Config::new("alpha", 2).retries, 2);
}

#[test]
fn reader_uses_the_field() {
    assert_eq!(read(&Config::new("alpha", 2)), "alpha:2");
}

#[test]
fn writer_uses_the_field() {
    assert_eq!(write(&Config::new("beta", 0)), "write beta");
}
