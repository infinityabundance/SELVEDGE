//! Two item types re-exported at the root. This does not compile: both
//! re-exports use the same name.

pub mod left;
pub mod right;

pub use left::Item;
pub use right::Item;
