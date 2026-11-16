//! Two traits with the same method.

pub trait Left {
    fn value(&self) -> u32;
}

pub trait Right {
    fn value(&self) -> u32;
}

pub struct Item(pub u32);

impl Left for Item {
    fn value(&self) -> u32 {
        self.0
    }
}

impl Right for Item {
    fn value(&self) -> u32 {
        self.0 + 100
    }
}

pub fn left(item: &Item) -> u32 {
    Left::value(item)
}

pub fn right(item: &Item) -> u32 {
    Right::value(item)
}

#[cfg(test)]
mod visible {
    use crate::{left, right, Item};

    #[test]
    fn left_uses_the_left_trait() {
        assert_eq!(left(&Item(1)), 1);
    }

    #[test]
    fn right_uses_the_right_trait() {
        assert_eq!(right(&Item(1)), 101);
    }

    #[test]
    fn the_traits_are_independent() {
        let item = Item(7);
        assert_eq!(left(&item), 7);
        assert_eq!(right(&item), 107);
    }
}
