//! Text folding through the vendored crate.

pub fn fold(input: &str) -> String {
    textkit::normalise_fast(input)
}

pub fn backend() -> &'static str {
    textkit::backend()
}
