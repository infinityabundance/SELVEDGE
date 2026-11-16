//! A trait with an associated constant. The crate does not build: the
//! impl omits `MAX`.

pub trait Limits {
    const MAX: usize;
    fn capped(&self, value: usize) -> usize;
}

pub struct Small;

impl Limits for Small {
    fn capped(&self, value: usize) -> usize {
        value.min(8)
    }
}
