//! A repository trait with an associated identifier type.

use std::collections::BTreeMap;

pub trait Repo {
    type Id;
    fn get(&self, id: Self::Id) -> Option<i64>;
    fn put(&mut self, id: Self::Id, value: i64);
}

#[derive(Default)]
pub struct MemoryRepo {
    items: BTreeMap<u32, i64>,
}

impl Repo for MemoryRepo {
    type Id = u32;

    fn get(&self, id: u32) -> Option<i64> {
        self.items.get(&id).copied()
    }

    fn put(&mut self, id: u32, value: i64) {
        self.items.insert(id, value);
    }
}

pub fn first_value<R: Repo<Id = u32>>(repo: &R) -> i64 {
    repo.get(0).unwrap_or(0)
}
