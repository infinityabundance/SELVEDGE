//! A ledger behind an interior-mutability cell.

use std::cell::RefCell;

pub struct Ledger {
    entries: RefCell<Vec<i64>>,
}

impl Ledger {
    pub fn new() -> Self {
        Self { entries: RefCell::new(Vec::new()) }
    }

    pub fn record(&self, value: i64) {
        let mut entries = self.entries.borrow_mut();
        let total: i64 = entries.iter().sum();
        entries.push(total + value);
    }

    pub fn entries(&self) -> Vec<i64> {
        self.entries.borrow().clone()
    }
}
