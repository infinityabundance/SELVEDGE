//! A named tag. The crate does not build: the impl defines a method the
//! trait does not declare.

pub trait Named {
    fn name(&self) -> &str;
}

pub struct Tag {
    pub name: String,
}

impl Named for Tag {
    fn name(&self) -> &str {
        &self.name
    }

    fn rename(&mut self, new: &str) {
        self.name = new.to_string();
    }
}
