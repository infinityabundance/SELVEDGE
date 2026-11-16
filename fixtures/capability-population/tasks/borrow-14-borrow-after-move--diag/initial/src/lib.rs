//! Join a list and measure it. The crate does not build: the list is
//! used after it is moved.

pub fn join_then_measure(parts: Vec<String>) -> (String, usize) {
    let joined = parts.join("-");
    let first = parts.into_iter().next().unwrap_or_default();
    let count = parts.len();
    (format!("{joined}:{first}"), count)
}
