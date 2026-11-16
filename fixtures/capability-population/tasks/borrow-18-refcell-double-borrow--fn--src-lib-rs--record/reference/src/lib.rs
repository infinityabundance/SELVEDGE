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

#[cfg(test)]
mod visible {
    use crate::Ledger;

    #[test]
    fn the_first_record_is_its_own_total() {
        let ledger = Ledger::new();
        ledger.record(1);
        assert_eq!(ledger.entries(), vec![1]);
    }

    #[test]
    fn each_record_runs_the_total_forward() {
        let ledger = Ledger::new();
        ledger.record(1);
        ledger.record(2);
        ledger.record(5);
        assert_eq!(ledger.entries(), vec![1, 3, 9]);
    }

    #[test]
    fn zero_is_recorded() {
        let ledger = Ledger::new();
        ledger.record(0);
        assert_eq!(ledger.entries(), vec![0]);
    }

    #[test]
    fn reads_do_not_change_the_ledger() {
        let ledger = Ledger::new();
        ledger.record(4);
        assert_eq!(ledger.entries(), vec![4]);
        assert_eq!(ledger.entries(), vec![4]);
    }
}
