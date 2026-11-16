//! Configuration shared by a reader and a writer.

pub mod config;
pub mod reader;
pub mod writer;

pub use config::Config;
pub use reader::read;
pub use writer::write;
