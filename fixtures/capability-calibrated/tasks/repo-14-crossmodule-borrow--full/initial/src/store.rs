//! The item store.

pub struct Store {
    pub items: Vec<String>,
}

impl Store {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }
}
