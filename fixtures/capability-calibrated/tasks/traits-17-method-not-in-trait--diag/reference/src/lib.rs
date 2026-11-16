//! A named tag.

pub trait Named {
    fn name(&self) -> &str;
}

pub struct Tag {
    pub name: String,
}

impl Tag {
    pub fn rename(&mut self, new: &str) {
        self.name = new.to_string();
    }
}

impl Named for Tag {
    fn name(&self) -> &str {
        &self.name
    }
}
