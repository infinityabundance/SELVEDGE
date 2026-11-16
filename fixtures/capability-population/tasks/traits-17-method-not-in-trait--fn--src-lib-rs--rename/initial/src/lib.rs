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
    use crate::{Named, Tag};

    #[test]
    fn the_tag_can_be_renamed() {
        let mut tag = Tag { name: "old".to_string() };
        tag.rename("new");
        assert_eq!(tag.name(), "new");
    }

    #[test]
    fn the_trait_method_sees_the_new_name() {
        let mut tag = Tag { name: "old".to_string() };
        tag.rename("alpha");
        assert_eq!(Named::name(&tag), "alpha");
    }

    #[test]
    fn an_empty_name_is_allowed() {
        let mut tag = Tag { name: "old".to_string() };
        tag.rename("");
        assert_eq!(tag.name(), "");
    }
}
