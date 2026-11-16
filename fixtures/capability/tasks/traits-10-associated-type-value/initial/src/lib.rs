//! A sequence with an associated item type. The crate does not build:
//! the associated type and the produced value disagree.

pub trait Sequence {
    type Item;
    fn next_item(&mut self) -> Option<Self::Item>;
}

pub struct Counter {
    pub current: u32,
}

impl Sequence for Counter {
    type Item = u32;

    fn next_item(&mut self) -> Option<Self::Item> {
        self.current += 1;
        Some(self.current as u64)
    }
}

pub fn drain<S: Sequence<Item = u64>>(sequence: &mut S, limit: usize) -> Vec<u64> {
    let mut out = Vec::new();
    for _ in 0..limit {
        match sequence.next_item() {
            Some(item) => out.push(item),
            None => break,
        }
    }
    out
}
