//! A report trait with a default method. The crate does not build: the
//! default method calls an item the trait does not declare.

pub trait Report {
    fn title(&self) -> String;

    fn render(&self) -> String {
        format!("[{}]", self.heading())
    }
}

pub struct Doc {
    pub text: String,
}

impl Report for Doc {
    fn title(&self) -> String {
        self.text.clone()
    }
}
