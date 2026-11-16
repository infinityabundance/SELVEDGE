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

#[cfg(test)]
mod visible {
    use crate::{first_value, MemoryRepo, Repo};

    #[test]
    fn put_then_get() {
        let mut repo = MemoryRepo::default();
        repo.put(7, 11);
        assert_eq!(repo.get(7), Some(11));
    }

    #[test]
    fn the_generic_helper_uses_the_associated_type() {
        let mut repo = MemoryRepo::default();
        repo.put(0, 5);
        assert_eq!(first_value(&repo), 5);
    }

    #[test]
    fn a_missing_id_is_none() {
        let repo = MemoryRepo::default();
        assert_eq!(repo.get(3), None);
    }
}
