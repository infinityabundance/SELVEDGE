use trio_01_lifetime_chain::{chain, lookup, resolve};

fn table() -> Vec<(&'static str, &'static str)> {
    vec![("alpha", "one"), ("beta", "two")]
}

#[test]
fn direct_lookup_borrows_the_table() {
    let table = table();
    assert_eq!(lookup(&table, "alpha"), Some("one"));
}

#[test]
fn resolution_follows_one_alias() {
    let table = table();
    let aliases = [("a", "alpha")];
    assert_eq!(resolve(&table, &aliases, "a"), Some("one"));
}

#[test]
fn the_chain_result_outlives_the_alias_slice() {
    let table = table();
    let value = {
        let aliases = [("a", "alpha")];
        chain(&table, &aliases, "a")
    };
    assert_eq!(value, Some("one"));
}

#[test]
fn an_unknown_key_is_none() {
    let table = table();
    let aliases = [("a", "alpha")];
    assert_eq!(chain(&table, &aliases, "missing"), None);
}

#[test]
fn a_direct_hit_skips_the_alias_table() {
    let table = table();
    let aliases = [("alpha", "beta")];
    assert_eq!(chain(&table, &aliases, "alpha"), Some("one"));
}
