use failing_18_negative_caesar_shift::shift;

#[test]
fn shifts_backwards_across_the_alphabet() {
    assert_eq!(shift("abc", -1), "zab");
}

#[test]
fn preserves_case_and_punctuation() {
    assert_eq!(shift("Hello, World!", 3), "Khoor, Zruog!");
}

#[test]
fn wraps_forwards_and_backwards() {
    assert_eq!(shift("xyz", 3), "abc");
    assert_eq!(shift("ABC", -3), "XYZ");
}

#[test]
fn a_zero_shift_changes_nothing() {
    assert_eq!(shift("a1!", 0), "a1!");
}
