use semantic_05_saturating_multiply::scale;

#[test]
fn small_products_are_exact_by_multiplication() {
    assert_eq!(scale(2, 3), 6);
}

#[test]
fn zero_absorbs() {
    assert_eq!(scale(u32::MAX, 0), 0);
}

#[test]
fn products_past_the_bound_saturate() {
    assert_eq!(scale(u32::MAX / 2 + 1, 2), u32::MAX);
}

#[test]
fn the_bound_times_one_is_the_bound() {
    assert_eq!(scale(u32::MAX, 1), u32::MAX);
}
