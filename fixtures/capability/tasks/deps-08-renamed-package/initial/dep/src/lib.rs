//! Text normalisation, version 2. The package is `textkit-impl`.

pub fn normalise(value: &str) -> String {
    value.trim().to_uppercase()
}
