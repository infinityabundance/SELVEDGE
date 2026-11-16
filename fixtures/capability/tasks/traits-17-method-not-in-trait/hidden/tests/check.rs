use traits_17_method_not_in_trait::{Named, Tag};

#[test]
fn the_tag_can_be_renamed() {
    let mut tag = Tag { name: "old".to_string() };
    tag.rename("new");
    assert_eq!(tag.name(), "new");
}

#[test]
fn the_trait_method_sees_the_new_name() {
    let mut tag = Tag { name: "old".to_string() };
    tag.rename("alpha");
    assert_eq!(Named::name(&tag), "alpha");
}

#[test]
fn an_empty_name_is_allowed() {
    let mut tag = Tag { name: "old".to_string() };
    tag.rename("");
    assert_eq!(tag.name(), "");
}
