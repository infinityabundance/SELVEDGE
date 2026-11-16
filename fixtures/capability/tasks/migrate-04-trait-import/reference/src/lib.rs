//! Render a document via the extension trait.

use documents::Render;

pub fn render_doc(title: &str) -> String {
    documents::doc(title).render()
}
