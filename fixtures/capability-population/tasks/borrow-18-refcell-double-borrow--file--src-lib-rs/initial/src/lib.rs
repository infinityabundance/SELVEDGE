//! A ledger behind an interior-mutability cell. The visible test
//! fails at run time: `record` takes two live mutable borrows.

use std::cell::RefCell;

pub struct Ledger {
    entries: RefCell<Vec<i64>>,
}

impl Ledger {
    pub fn new() -> Self {
        Self { entries: RefCell::new(Vec::new()) }
    }

    pub fn record(&self, value: i64) {
        let entries = self.entries.borrow_mut();
        let total: i64 = entries.iter().sum();
        self.entries.borrow_mut().push(total + value);
    }

    pub fn entries(&self) -> Vec<i64> {
        self.entries.borrow().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::Ledger;

    #[test]
    fn visible_record_panics() {
        let ledger = Ledger::new();
        ledger.record(1);
        assert_eq!(ledger.entries(), vec![1]);
    }
}
