use trio_09_quantifier_occurrences::{occurrences, Formula, Occurrence};

fn var(name: &str) -> Formula {
    Formula::Var(name.to_string())
}

#[test]
fn a_bound_occurrence_is_bound_at_depth_zero() {
    let formula = Formula::Forall("x".to_string(), Box::new(var("x")));
    assert_eq!(occurrences(&formula, "x"), vec![Occurrence::Bound(0)]);
}

#[test]
fn free_and_bound_occurrences_are_distinguished() {
    let formula = Formula::And(
        Box::new(var("x")),
        Box::new(Formula::Forall("x".to_string(), Box::new(var("x")))),
    );
    assert_eq!(
        occurrences(&formula, "x"),
        vec![Occurrence::Free, Occurrence::Bound(0)]
    );
}

#[test]
fn a_variable_under_an_unrelated_binder_is_free() {
    let formula = Formula::Exists("y".to_string(), Box::new(var("x")));
    assert_eq!(occurrences(&formula, "x"), vec![Occurrence::Free]);
}

#[test]
fn an_outer_variable_is_bound_at_the_nearest_same_name_binder() {
    let formula = Formula::Forall(
        "x".to_string(),
        Box::new(Formula::Exists("y".to_string(), Box::new(var("x")))),
    );
    assert_eq!(occurrences(&formula, "x"), vec![Occurrence::Bound(0)]);
}

#[test]
fn nested_same_name_binders_increase_the_binding_depth() {
    let formula = Formula::Forall(
        "x".to_string(),
        Box::new(Formula::Forall("x".to_string(), Box::new(var("x")))),
    );
    assert_eq!(occurrences(&formula, "x"), vec![Occurrence::Bound(1)]);
}

#[test]
fn a_formula_without_the_name_has_no_occurrences() {
    let formula = Formula::And(
        Box::new(Formula::Atom("p".to_string())),
        Box::new(Formula::Atom("q".to_string())),
    );
    assert!(occurrences(&formula, "x").is_empty());
}
