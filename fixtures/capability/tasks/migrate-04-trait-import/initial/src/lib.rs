//! Render a document. This does not build: `render` moved into a trait
//! that this module does not import.

pub fn render_doc(title: &str) -> String {
    documents::doc(title).render()
}
