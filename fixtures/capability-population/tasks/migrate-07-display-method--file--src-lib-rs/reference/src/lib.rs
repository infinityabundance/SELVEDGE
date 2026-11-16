//! Render a label.

pub fn render(value: &str) -> String {
    let label = labels::Label::new(value);
    label.to_string()
}
