//! A named tag.

pub trait Named {
    fn name(&self) -> &str;
}

pub struct Tag {
    pub name: String,
}

impl Tag {
// TODO(FUNCTION)
}

impl Named for Tag {
    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod visible {
    use super::*;

    #[test]
    fn the_tag_can_be_renamed() {
        let mut tag = Tag { name: "old".to_string() };
        tag.rename("new");
        assert_eq!(tag.name(), "new");
    }
}
