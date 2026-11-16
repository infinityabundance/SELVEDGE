//! A store and its synchroniser.

pub mod store;
pub mod sync;

pub use store::Store;
pub use sync::append_all;
