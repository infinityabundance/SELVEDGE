//! The name registry.

#[derive(Default)]
pub struct Registry {
    entries: Vec<String>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, name: &str) {
        self.entries.push(name.to_string());
    }

    pub fn count(&self) -> usize {
        self.entries.len()
    }
}
