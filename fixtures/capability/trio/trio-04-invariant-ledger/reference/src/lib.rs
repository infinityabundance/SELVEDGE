//! A ledger that preserves its invariant.

#[derive(Debug)]
pub struct Ledger {
    entries: Vec<i64>,
    total: i64,
}

impl Ledger {
    pub fn new() -> Self {
        Ledger { entries: Vec::new(), total: 0 }
    }

    pub fn deposit(&mut self, amount: i64) {
        self.entries.push(amount);
        self.total += amount;
    }

    pub fn withdraw(&mut self, amount: i64) -> Result<(), String> {
        if amount <= 0 {
            return Err("invalid amount".to_string());
        }
        if amount > self.balance() {
            return Err("insufficient funds".to_string());
        }
        self.entries.push(-amount);
        self.total -= amount;
        Ok(())
    }

    pub fn total(&self) -> i64 {
        self.total
    }

    pub fn balance(&self) -> i64 {
        self.entries.iter().sum()
    }

    pub fn entries(&self) -> &[i64] {
        &self.entries
    }
}
