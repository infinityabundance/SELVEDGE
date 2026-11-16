use trio_02_math_dependency::{minimal_recompute, PlanError, Rule};

const RULES: &[Rule] = &[
    Rule { output: "b", inputs: &["a"] },
    Rule { output: "c", inputs: &["b"] },
];

fn plan<'a>(
    rules: &'a [Rule],
    known: &[&str],
    target: &'static str,
) -> Result<Vec<&'static str>, PlanError> {
    minimal_recompute(rules, known, target)
}

#[test]
fn known_inputs_are_pruned() {
    assert_eq!(plan(RULES, &["a"], "c"), Ok(vec!["b", "c"]));
}

#[test]
fn a_known_target_needs_nothing() {
    assert_eq!(plan(RULES, &["a", "b", "c"], "c"), Ok(vec![]));
}

#[test]
fn intermediate_known_values_are_skipped() {
    assert_eq!(plan(RULES, &["a", "b"], "c"), Ok(vec!["c"]));
}

#[test]
fn a_diamond_orders_dependencies_first() {
    let rules = [
        Rule { output: "b", inputs: &["a"] },
        Rule { output: "c", inputs: &["a"] },
        Rule { output: "d", inputs: &["b", "c"] },
    ];
    assert_eq!(plan(&rules, &["a"], "d"), Ok(vec!["b", "c", "d"]));
}

#[test]
fn an_unknown_target_is_an_error() {
    assert_eq!(plan(RULES, &["a"], "zz"), Err(PlanError::Unknown("zz")));
}

#[test]
fn a_cycle_is_an_error() {
    let rules = [
        Rule { output: "a", inputs: &["b"] },
        Rule { output: "b", inputs: &["a"] },
    ];
    assert_eq!(plan(&rules, &[], "a"), Err(PlanError::Cycle("a")));
}
