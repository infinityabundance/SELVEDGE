//! Tally, version 2. Version 1 exposed `Tally::sum`; version 2 calls
//! the same method `Tally::total`.

pub struct Tally {
    values: Vec<i64>,
}

impl Tally {
    pub fn new() -> Self {
        Self { values: Vec::new() }
    }

    pub fn push(&mut self, value: i64) {
        self.values.push(value);
    }

    pub fn total(&self) -> i64 {
        self.values.iter().sum()
    }
}
