//! A registry and its registration flow.

pub mod flow;
pub mod registry;

pub use flow::register_and_count;
pub use registry::Registry;
