//! Text normalisation.

pub fn normalise(text: &str) -> String {
    text.trim().to_uppercase()
}
