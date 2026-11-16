//! Render a plate. This does not build: version 2 takes the text by
//! value.

pub fn plate(text: &str) -> String {
    nameplate::render(text)
}
