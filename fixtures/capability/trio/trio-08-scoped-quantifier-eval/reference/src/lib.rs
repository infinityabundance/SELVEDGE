//! Scoped arithmetic evaluation with nearest-binding resolution.

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Const(i64),
    Var(String),
    Add(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Sum { variable: String, low: i64, high: i64, body: Box<Expr> },
}

pub fn eval(expr: &Expr, environment: &[(String, i64)]) -> Result<i64, String> {
    match expr {
        Expr::Const(value) => Ok(*value),
        Expr::Var(name) => environment
            .iter()
            .rev()
            .find(|(binding, _value)| binding == name)
            .map(|(_binding, value)| *value)
            .ok_or_else(|| format!("unbound {name}")),
        Expr::Add(left, right) => Ok(eval(left, environment)? + eval(right, environment)?),
        Expr::Mul(left, right) => Ok(eval(left, environment)? * eval(right, environment)?),
        Expr::Sum { variable, low, high, body } => {
            let mut total = 0;
            let mut inner = environment.to_vec();
            for value in *low..=*high {
                inner.push((variable.clone(), value));
                total += eval(body, &inner)?;
            }
            Ok(total)
        }
    }
}
