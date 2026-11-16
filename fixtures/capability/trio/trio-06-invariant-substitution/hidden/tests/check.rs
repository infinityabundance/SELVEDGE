use trio_06_invariant_substitution::{free_vars, substitute, Term};

fn var(name: &str) -> Term {
    Term::Var(name.to_string())
}

#[test]
fn replaces_a_free_variable() {
    let term = Term::App("f".to_string(), vec![var("x")]);
    let expected = Term::App("f".to_string(), vec![var("y")]);
    assert_eq!(substitute(&term, "x", &var("y")), Some(expected));
}

#[test]
fn a_shadowed_binder_is_unchanged() {
    let term = Term::Lam("x".to_string(), Box::new(var("x")));
    assert_eq!(substitute(&term, "x", &var("y")), Some(term));
}

#[test]
fn a_capturing_substitution_is_rejected() {
    let term = Term::Lam("y".to_string(), Box::new(var("x")));
    assert_eq!(substitute(&term, "x", &var("y")), None);
}

#[test]
fn substitution_descends_under_an_unrelated_binder() {
    let term = Term::Lam("z".to_string(), Box::new(var("x")));
    let expected = Term::Lam("z".to_string(), Box::new(var("y")));
    assert_eq!(substitute(&term, "x", &var("y")), Some(expected));
}

#[test]
fn the_free_variable_relation_is_preserved() {
    let term = Term::Lam("z".to_string(), Box::new(var("x")));
    let result = substitute(&term, "x", &var("y")).unwrap();
    assert_eq!(free_vars(&result), vec!["y".to_string()]);
    assert_eq!(free_vars(&term), vec!["x".to_string()]);
}
