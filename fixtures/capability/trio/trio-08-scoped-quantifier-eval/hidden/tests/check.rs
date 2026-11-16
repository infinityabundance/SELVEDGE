use trio_08_scoped_quantifier_eval::{eval, Expr};

fn var(name: &str) -> Expr {
    Expr::Var(name.to_string())
}

fn add(left: Expr, right: Expr) -> Expr {
    Expr::Add(Box::new(left), Box::new(right))
}

fn mul(left: Expr, right: Expr) -> Expr {
    Expr::Mul(Box::new(left), Box::new(right))
}

fn sum(variable: &str, low: i64, high: i64, body: Expr) -> Expr {
    Expr::Sum {
        variable: variable.to_string(),
        low,
        high,
        body: Box::new(body),
    }
}

#[test]
fn a_sum_evaluates_its_body() {
    let expr = sum("x", 1, 4, mul(var("x"), var("x")));
    assert_eq!(eval(&expr, &[]), Ok(30));
}

#[test]
fn an_inner_binder_shadows_an_outer_one() {
    let outer = vec![("x".to_string(), 100)];
    let expr = sum("x", 1, 2, add(var("x"), Expr::Const(1)));
    assert_eq!(eval(&expr, &outer), Ok(5));
}

#[test]
fn an_unbound_variable_is_an_error() {
    assert_eq!(eval(&var("z"), &[]), Err("unbound z".to_string()));
}

#[test]
fn an_outer_binding_remains_visible_under_an_unrelated_binder() {
    let outer = vec![("y".to_string(), 7)];
    let expr = sum("x", 1, 2, var("y"));
    assert_eq!(eval(&expr, &outer), Ok(14));
}

#[test]
fn arithmetic_nesting_is_respected() {
    let expr = sum("x", 1, 3, add(mul(Expr::Const(2), var("x")), Expr::Const(1)));
    assert_eq!(eval(&expr, &[]), Ok(15));
}
