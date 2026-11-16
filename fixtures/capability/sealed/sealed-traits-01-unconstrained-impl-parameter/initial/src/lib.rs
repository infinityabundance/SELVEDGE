//! A registry with an unconstrained impl parameter. The crate does not
//! build.

#[derive(Default)]
pub struct Registry {
    pub entries: Vec<String>,
}

impl<T> Registry {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }
}
