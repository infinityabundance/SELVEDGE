//! A report trait with a default method.

pub trait Report {
    fn title(&self) -> String;

// TODO(FUNCTION)
}

pub struct Doc {
    pub text: String,
}

impl Report for Doc {
    fn title(&self) -> String {
        self.text.clone()
    }
}

#[cfg(test)]
mod visible {
    use crate::{Doc, Report};

    #[test]
    fn the_default_render_wraps_the_title() {
        let doc = Doc { text: "x".to_string() };
        assert_eq!(doc.render(), "[x]");
    }

    #[test]
    fn the_title_is_available_directly() {
        let doc = Doc { text: "alpha".to_string() };
        assert_eq!(doc.title(), "alpha");
    }

    #[test]
    fn an_empty_title_renders_empty_brackets() {
        let doc = Doc { text: String::new() };
        assert_eq!(doc.render(), "[]");
    }
}
