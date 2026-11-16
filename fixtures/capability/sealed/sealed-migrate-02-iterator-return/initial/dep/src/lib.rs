//! Catalog, version 2. `items` owns its result instead of borrowing the
//! catalog.

pub struct Catalog {
    entries: Vec<String>,
}

impl Catalog {
    pub fn new(entries: Vec<String>) -> Self {
        Self { entries }
    }

    pub fn items(&self) -> impl Iterator<Item = String> + '_ {
        self.entries.iter().cloned()
    }
}
