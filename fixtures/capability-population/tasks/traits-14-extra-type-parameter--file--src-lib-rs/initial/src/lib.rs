//! An encoder trait. The crate does not build: the impl declares a type
//! parameter the trait does not have.

pub trait Encode {
    fn encode(&self) -> String;
}

pub struct Doc {
    pub text: String,
}

impl Encode for Doc {
    fn encode<T>(&self) -> String {
        self.text.clone()
    }
}
