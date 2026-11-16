//! Nameplate, version 2. `render` takes the text by value.

pub fn render(text: String) -> String {
    format!("[{text}]")
}
