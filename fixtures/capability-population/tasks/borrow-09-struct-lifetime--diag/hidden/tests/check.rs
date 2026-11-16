use borrow_09_struct_lifetime::View;

#[test]
fn borrows_a_static_literal() {
    assert_eq!(View::new("alpha").text(), "alpha");
}

#[test]
fn borrows_an_owned_string() {
    let owned = String::from("beta");
    let view = View::new(&owned);
    assert_eq!(view.text(), "beta");
}

#[test]
fn empty_text_is_empty() {
    assert_eq!(View::new("").text(), "");
}

#[test]
fn two_views_are_independent() {
    let first = String::from("one");
    let second = String::from("two");
    assert_eq!(View::new(&first).text(), "one");
    assert_eq!(View::new(&second).text(), "two");
}
