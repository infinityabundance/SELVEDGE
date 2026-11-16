//! Double through the vendored crate.

pub fn twice(value: i64) -> i64 {
    mathx::twice(value)
}

pub fn dependency_label() -> &'static str {
    mathx::label()
}
