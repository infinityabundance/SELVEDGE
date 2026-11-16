//! Join a list and measure it.

pub fn join_then_measure(parts: Vec<String>) -> (String, usize) {
    let joined = parts.join("-");
    let count = parts.len();
    let first = parts.into_iter().next().unwrap_or_default();
    (format!("{joined}:{first}"), count)
}
