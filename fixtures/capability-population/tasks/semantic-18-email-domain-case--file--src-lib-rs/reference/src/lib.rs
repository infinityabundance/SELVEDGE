//! Address normalisation: only the domain is lowercased.

pub fn normalise(text: &str) -> String {
    match text.split_once('@') {
        Some((local, domain)) => {
            format!("{}@{}", local, domain.to_lowercase())
        }
        None => text.to_string(),
    }
}
