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

fn visit(
    rules: &[Rule],
    known: &[&str],
    node: &'static str,
    done: &mut Vec<&'static str>,
    visiting: &mut Vec<&'static str>,
    plan: &mut Vec<&'static str>,
) -> Result<(), PlanError> {
    if known.contains(&node) || done.contains(&node) {
        return Ok(());
    }
    if visiting.contains(&node) {
        return Err(PlanError::Cycle(node));
    }
    let rule = rules
        .iter()
        .find(|rule| rule.output == node)
        .ok_or(PlanError::Unknown(node))?;
    visiting.push(node);
    for input in rule.inputs {
        visit(rules, known, input, done, visiting, plan)?;
    }
    visiting.pop();
    done.push(node);
    plan.push(node);
    Ok(())
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
    let mut done = Vec::new();
    let mut visiting = Vec::new();
    visit(rules, known, target, &mut done, &mut visiting, &mut plan)?;
    Ok(plan)
}
