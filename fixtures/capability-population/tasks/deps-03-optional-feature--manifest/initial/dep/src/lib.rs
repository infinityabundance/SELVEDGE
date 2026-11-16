//! Text handling, version 2: the fast path is an optional feature.

#[cfg(feature = "fast")]
pub fn normalise_fast(input: &str) -> String {
    input
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

#[cfg(feature = "fast")]
pub fn backend() -> &'static str {
    "fast"
}
