use trio_07_scope_resolver::{resolve, Scope};

#[test]
fn the_innermost_binding_wins() {
    let outer = Scope::root().push("x");
    let inner = outer.push("x");
    assert_eq!(resolve(&inner, "x"), Some(0));
}

#[test]
fn an_outer_binding_is_found_at_its_distance() {
    let outer = Scope::root().push("x");
    let inner = outer.push("y");
    assert_eq!(resolve(&inner, "x"), Some(1));
}

#[test]
fn a_free_name_is_none() {
    let scope = Scope::root().push("y");
    assert_eq!(resolve(&scope, "x"), None);
}

#[test]
fn shadowing_across_three_levels_resolves_innermost() {
    let outer = Scope::root().push("x");
    let middle = outer.push("y");
    let inner = middle.push("x");
    assert_eq!(resolve(&inner, "x"), Some(0));
    assert_eq!(resolve(&inner, "y"), Some(1));
}
