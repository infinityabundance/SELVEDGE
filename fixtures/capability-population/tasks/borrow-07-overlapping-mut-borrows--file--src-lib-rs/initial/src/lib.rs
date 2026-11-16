//! A counter with two overlapping mutable borrows.

pub struct Counter {
    value: i32,
}

impl Counter {
    pub fn new(value: i32) -> Self {
        Counter { value }
    }

    pub fn value(&self) -> i32 {
        self.value
    }

    pub fn value_mut(&mut self) -> &mut i32 {
        &mut self.value
    }

    pub fn bump_twice(&mut self) {
        let first = self.value_mut();
        let second = self.value_mut();
        *first += 1;
        *second += 1;
    }
}
