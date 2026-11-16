//! Recompute planning over declared rules.

#[derive(Debug, Clone, Copy)]
pub struct Rule {
    pub output: &'static str,
    pub inputs: &'static [&'static str],
}

#[derive(Debug, PartialEq)]
pub enum PlanError {
    Unknown(&'static str),
    Cycle(&'static str),
}

pub fn minimal_recompute(
    rules: &[Rule],
    known: &[&str],
    target: &'static str,
) -> Result<Vec<&'static str>, PlanError> {
    if known.contains(&target) {
        return Ok(Vec::new());
    }
    let mut plan = Vec::new();
    let mut stack = vec![target];
    let mut visited: Vec<&'static str> = Vec::new();
    while let Some(node) = stack.pop() {
        if visited.contains(&node) {
            continue;
        }
        visited.push(node);
        let rule = rules
            .iter()
            .find(|rule| rule.output == node)
            .ok_or(PlanError::Unknown(node))?;
        plan.push(node);
        for input in rule.inputs {
            if !known.contains(input) {
                stack.push(input);
            }
        }
    }
    Ok(plan)
}
