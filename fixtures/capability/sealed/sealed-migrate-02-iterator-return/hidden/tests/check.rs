use sealed_migrate_02_iterator_return::{summarise, Catalog};

fn catalog(names: &[&str]) -> Catalog {
    Catalog::new(names.iter().map(|name| name.to_string()).collect())
}

#[test]
fn summarises_a_non_empty_catalog() {
    assert_eq!(summarise(&catalog(&["a", "b"])),
               (2, Some("a".to_string())));
}

#[test]
fn summarises_an_empty_catalog() {
    assert_eq!(summarise(&catalog(&[])), (0, None));
}

#[test]
fn a_single_entry_is_summarised() {
    assert_eq!(summarise(&catalog(&["only"])),
               (1, Some("only".to_string())));
}
