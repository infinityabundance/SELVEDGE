use traits_11_blanket_impl_bound::Summary;

#[test]
fn integers_are_summarised() {
    assert_eq!(12u8.summary(), "12");
}

#[test]
fn strings_are_summarised() {
    assert_eq!("alpha".summary(), "alpha");
}

#[test]
fn the_display_form_is_used() {
    assert_eq!(format!("{:.2}", 1.5f64).summary(), "1.50");
}
