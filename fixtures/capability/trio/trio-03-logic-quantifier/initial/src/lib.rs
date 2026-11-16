//! Quantified premise chains.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantifier {
    All,
    Some,
}

#[derive(Debug, Clone, Copy)]
pub struct Premise {
    pub subject: &'static str,
    pub quantifier: Quantifier,
    pub object: &'static str,
}

#[derive(Debug, PartialEq)]
pub struct Repair {
    pub path: Vec<usize>,
    pub strengthened: Vec<usize>,
}

pub fn repair_path(
    premises: &[Premise],
    from: &'static str,
    to: &'static str,
) -> Option<Repair> {
    if from == to {
        return Some(Repair { path: Vec::new(), strengthened: Vec::new() });
    }
    premises
        .iter()
        .position(|premise| premise.subject == from && premise.object == to)
        .map(|index| Repair { path: vec![index], strengthened: Vec::new() })
}
