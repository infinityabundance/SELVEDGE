use sealed_semantic_03_percent_rounding::percent;

#[test]
fn two_thirds_rounds_up() {
    assert_eq!(percent(2, 3), 67);
}

#[test]
fn one_third_rounds_down() {
    assert_eq!(percent(1, 3), 33);
}

#[test]
fn an_exact_half_rounds_up() {
    assert_eq!(percent(1, 8), 13);
}

#[test]
fn zero_total_and_zero_part() {
    assert_eq!(percent(0, 5), 0);
    assert_eq!(percent(5, 0), 0);
}
