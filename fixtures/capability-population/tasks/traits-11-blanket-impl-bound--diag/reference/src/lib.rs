//! A blanket summary.

use std::fmt::Display;

pub trait Summary {
    fn summary(&self) -> String;
}

impl<T: Display> Summary for T {
    fn summary(&self) -> String {
        format!("{}", self)
    }
}
