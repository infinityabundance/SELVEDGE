use migrate_04_trait_import::render_doc;

#[test]
fn renders_a_heading() {
    assert_eq!(render_doc("t"), "<h1>t</h1>");
}

#[test]
fn renders_an_empty_title() {
    assert_eq!(render_doc(""), "<h1></h1>");
}
