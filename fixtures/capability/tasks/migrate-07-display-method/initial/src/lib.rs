//! Render a label. This does not build: version 2 removed the
//! `text` accessor and renders through `Display`.

pub fn render(value: &str) -> String {
    let label = labels::Label::new(value);
    label.text().to_string()
}
