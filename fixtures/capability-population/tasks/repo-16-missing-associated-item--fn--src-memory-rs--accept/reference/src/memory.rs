//! An in-memory sink.

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

    fn finish(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}

#[cfg(test)]
mod visible {
    use crate::{Memory, Sink};

    #[test]
    fn accepted_bytes_are_recorded() {
        let mut memory = Memory::default();
        memory.accept(1).unwrap();
        memory.accept(2).unwrap();
        memory.finish().unwrap();
        assert_eq!(memory.bytes, vec![1, 2]);
    }

    #[test]
    fn finish_on_an_empty_sink_succeeds() {
        let mut memory = Memory::default();
        assert!(memory.finish().is_ok());
        assert!(memory.bytes.is_empty());
    }

    #[test]
    fn the_error_type_is_the_unit_type() {
        let mut memory = Memory::default();
        let result: Result<(), ()> = memory.finish();
        assert!(result.is_ok());
    }
}
