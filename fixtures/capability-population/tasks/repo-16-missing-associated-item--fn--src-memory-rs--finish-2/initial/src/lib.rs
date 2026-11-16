//! A sink and its in-memory implementation.

pub mod memory;
pub mod traits;

pub use memory::Memory;
pub use traits::Sink;
