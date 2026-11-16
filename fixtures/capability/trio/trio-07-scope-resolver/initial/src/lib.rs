//! Scope resolution, currently returning the outermost binding.

#[derive(Clone, Debug, PartialEq)]
pub struct Scope {
    pub bindings: Vec<String>,
    pub parent: Option<Box<Scope>>,
}

impl Scope {
    pub fn root() -> Self {
        Scope { bindings: Vec::new(), parent: None }
    }

    pub fn push(&self, binding: &str) -> Self {
        Scope {
            bindings: vec![binding.to_string()],
            parent: Some(Box::new(self.clone())),
        }
    }
}

pub fn resolve(scope: &Scope, name: &str) -> Option<usize> {
    let mut current = Some(scope);
    let mut depth = 0;
    let mut found = None;
    while let Some(node) = current {
        if node.bindings.iter().any(|binding| binding == name) {
            found = Some(depth);
        }
        depth += 1;
        current = node.parent.as_deref();
    }
    found
}
