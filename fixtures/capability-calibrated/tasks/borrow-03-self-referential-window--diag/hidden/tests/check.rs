use borrow_03_self_referential_window::Window;

#[test]
fn first_word() {
    assert_eq!(Window::new("alpha beta".to_string()).first(), "alpha");
}

#[test]
fn empty_text_has_no_first_word() {
    assert_eq!(Window::new(String::new()).first(), "");
}

#[test]
fn first_word_survives_replace() {
    let mut window = Window::new("alpha beta".to_string());
    window.replace("gamma delta".to_string());
    assert_eq!(window.first(), "gamma");
}
