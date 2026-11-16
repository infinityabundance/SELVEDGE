//! Occurrence classification, currently counting every binder.

#[derive(Clone, Debug, PartialEq)]
pub enum Occurrence {
    Free,
    Bound(u32),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Formula {
    Var(String),
    Atom(String),
    Not(Box<Formula>),
    And(Box<Formula>, Box<Formula>),
    Forall(String, Box<Formula>),
    Exists(String, Box<Formula>),
}

pub fn occurrences(formula: &Formula, name: &str) -> Vec<Occurrence> {
    fn walk(formula: &Formula, name: &str, depth: u32, out: &mut Vec<Occurrence>) {
        match formula {
            Formula::Var(variable) => {
                if variable == name {
                    out.push(if depth == 0 {
                        Occurrence::Free
                    } else {
                        Occurrence::Bound(depth - 1)
                    });
                }
            }
            Formula::Atom(_) => {}
            Formula::Not(inner) => walk(inner, name, depth, out),
            Formula::And(left, right) => {
                walk(left, name, depth, out);
                walk(right, name, depth, out);
            }
            Formula::Forall(_binder, body) => walk(body, name, depth + 1, out),
            Formula::Exists(_binder, body) => walk(body, name, depth + 1, out),
        }
    }
    let mut out = Vec::new();
    walk(formula, name, 0, &mut out);
    out
}
