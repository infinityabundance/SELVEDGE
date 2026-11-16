use borrow_01_local_reference::label;

#[test]
fn label_owns_its_result() {
    assert_eq!(label("job"), "job:ready".to_string());
}

#[test]
fn successive_labels_are_independent() {
    let first = label("a");
    let second = label("b");
    assert_eq!(first, "a:ready");
    assert_eq!(second, "b:ready");
}
