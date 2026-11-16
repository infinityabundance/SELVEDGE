//! A report trait with a default method.

pub trait Report {
    fn title(&self) -> String;

    fn render(&self) -> String {
        format!("[{}]", self.title())
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
