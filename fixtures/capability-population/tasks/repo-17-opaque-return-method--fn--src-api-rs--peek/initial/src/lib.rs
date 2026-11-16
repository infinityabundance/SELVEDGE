//! A keyed store and its API.

pub mod api;
pub mod store;

pub use api::{open, peek};
pub use store::Store;
