//! Naive substitution that ignores binding structure.

#[derive(Clone, Debug, PartialEq)]
pub enum Term {
    Var(String),
    App(String, Vec<Term>),
    Lam(String, Box<Term>),
}

pub fn free_vars(term: &Term) -> Vec<String> {
    fn collect(term: &Term, bound: &mut Vec<String>, out: &mut Vec<String>) {
        match term {
            Term::Var(name) => {
                if !bound.contains(name) {
                    out.push(name.clone());
                }
            }
            Term::App(_function, arguments) => {
                for argument in arguments {
                    collect(argument, bound, out);
                }
            }
            Term::Lam(binder, body) => {
                bound.push(binder.clone());
                collect(body, bound, out);
                bound.pop();
            }
        }
    }
    let mut out = Vec::new();
    collect(term, &mut Vec::new(), &mut out);
    out.sort();
    out.dedup();
    out
}

pub fn substitute(term: &Term, name: &str, replacement: &Term) -> Option<Term> {
    Some(match term {
        Term::Var(variable) if variable == name => replacement.clone(),
        Term::Var(variable) => Term::Var(variable.clone()),
        Term::App(function, arguments) => Term::App(
            function.clone(),
            arguments
                .iter()
                .map(|argument| substitute(argument, name, replacement).unwrap())
                .collect(),
        ),
        Term::Lam(binder, body) => Term::Lam(
            binder.clone(),
            Box::new(substitute(body, name, replacement).unwrap()),
        ),
    })
}
