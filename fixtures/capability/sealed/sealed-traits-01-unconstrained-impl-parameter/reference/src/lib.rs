//! A registry.

#[derive(Default)]
pub struct Registry {
    pub entries: Vec<String>,
}

impl Registry {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }
}
