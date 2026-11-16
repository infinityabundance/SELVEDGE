use sealed_repo_02_duplicate_reexport::{Item, RightItem};

#[test]
fn the_left_item_keeps_its_name() {
    let item = Item { value: 4 };
    assert_eq!(item.left_value(), 4);
}

#[test]
fn the_right_item_is_aliased() {
    let item = RightItem { value: 4 };
    assert_eq!(item.right_value(), 8);
}
