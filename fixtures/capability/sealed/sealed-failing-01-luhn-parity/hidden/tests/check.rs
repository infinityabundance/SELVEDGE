use sealed_failing_01_luhn_parity::luhn_valid;

#[test]
fn accepts_a_known_valid_number() {
    assert!(luhn_valid("49927398716"));
}

#[test]
fn accepts_another_known_valid_number() {
    assert!(luhn_valid("79927398713"));
}

#[test]
fn rejects_a_single_digit_change() {
    assert!(!luhn_valid("49927398717"));
}

#[test]
fn rejects_non_digits_and_short_input() {
    assert!(!luhn_valid("4992739871x"));
    assert!(!luhn_valid("7"));
}
