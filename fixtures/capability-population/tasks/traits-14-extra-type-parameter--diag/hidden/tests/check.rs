use traits_14_extra_type_parameter::{Doc, Encode};

#[test]
fn encodes_the_text() {
    let doc = Doc { text: "abc".to_string() };
    assert_eq!(doc.encode(), "abc");
}

#[test]
fn an_empty_document_encodes_to_empty() {
    let doc = Doc { text: String::new() };
    assert_eq!(doc.encode(), "");
}

#[test]
fn the_generic_caller_works() {
    fn render<E: Encode>(value: &E) -> String {
        value.encode()
    }
    let doc = Doc { text: "x".to_string() };
    assert_eq!(render(&doc), "x");
}
