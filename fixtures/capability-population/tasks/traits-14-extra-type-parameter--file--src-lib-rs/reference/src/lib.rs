//! An encoder trait.

pub trait Encode {
    fn encode(&self) -> String;
}

pub struct Doc {
    pub text: String,
}

impl Encode for Doc {
    fn encode(&self) -> String {
        self.text.clone()
    }
}
