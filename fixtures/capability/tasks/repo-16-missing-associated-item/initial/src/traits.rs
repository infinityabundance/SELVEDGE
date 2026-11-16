//! The sink trait.

pub trait Sink {
    type Error;
    fn accept(&mut self, value: u8) -> Result<(), Self::Error>;
    fn finish(&mut self) -> Result<(), Self::Error>;
}
