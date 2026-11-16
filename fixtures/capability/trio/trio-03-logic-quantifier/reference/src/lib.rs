//! Quantified premise chains.

use std::collections::VecDeque;

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
    let mut queue: VecDeque<(&'static str, Vec<usize>)> = VecDeque::new();
    let mut seen: Vec<&'static str> = vec![from];
    queue.push_back((from, Vec::new()));
    while let Some((node, path)) = queue.pop_front() {
        for (index, premise) in premises.iter().enumerate() {
            if premise.subject != node || seen.contains(&premise.object) {
                continue;
            }
            let mut next_path = path.clone();
            next_path.push(index);
            if premise.object == to {
                let strengthened = next_path
                    .iter()
                    .copied()
                    .filter(|index| premises[*index].quantifier == Quantifier::Some)
                    .collect();
                return Some(Repair { path: next_path, strengthened });
            }
            seen.push(premise.object);
            queue.push_back((premise.object, next_path));
        }
    }
    None
}
