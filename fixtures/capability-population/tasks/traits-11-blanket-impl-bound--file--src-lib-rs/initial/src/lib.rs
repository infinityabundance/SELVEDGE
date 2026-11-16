//! A blanket summary. The crate does not build: the impl formats `T`
//! without requiring `Display`.

pub trait Summary {
    fn summary(&self) -> String;
}

impl<T> Summary for T {
    fn summary(&self) -> String {
        format!("{}", self)
    }
}
