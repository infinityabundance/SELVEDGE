use deps_17_invalid_package_name::median;

#[test]
fn an_odd_length_takes_the_middle() {
    assert_eq!(median(&mut [3, 1, 2]), Some(2));
}

#[test]
fn an_even_length_takes_the_upper_middle() {
    assert_eq!(median(&mut [4, 1, 3, 2]), Some(3));
}

#[test]
fn an_empty_slice_is_none() {
    assert_eq!(median(&mut []), None);
}
