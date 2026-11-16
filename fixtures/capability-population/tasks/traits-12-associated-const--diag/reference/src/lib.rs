//! A trait with an associated constant.

pub trait Limits {
    const MAX: usize;
    fn capped(&self, value: usize) -> usize;
}

pub struct Small;

impl Limits for Small {
    const MAX: usize = 8;

    fn capped(&self, value: usize) -> usize {
        value.min(Self::MAX)
    }
}
