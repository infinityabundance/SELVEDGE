//! An in-memory sink. This does not compile: the impl omits `finish`.

use crate::traits::Sink;

#[derive(Default)]
pub struct Memory {
    pub bytes: Vec<u8>,
}

impl Sink for Memory {
    type Error = ();

    fn accept(&mut self, value: u8) -> Result<(), Self::Error> {
        self.bytes.push(value);
        Ok(())
    }
}
