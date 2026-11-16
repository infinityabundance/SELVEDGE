//! Documents, version 2: rendering is an extension trait.

pub struct Doc {
    pub title: String,
}

pub trait Render {
    fn render(&self) -> String;
}

impl Render for Doc {
    fn render(&self) -> String {
        format!("<h1>{}</h1>", self.title)
    }
}

pub fn doc(title: &str) -> Doc {
    Doc { title: title.to_string() }
}
