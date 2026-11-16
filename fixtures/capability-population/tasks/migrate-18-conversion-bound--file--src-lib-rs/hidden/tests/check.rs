use migrate_18_conversion_bound::scale_units;

#[test]
fn a_small_value_scales() {
    assert_eq!(scale_units(3), 3.0);
}

#[test]
fn zero_scales_to_zero() {
    assert_eq!(scale_units(0), 0.0);
}

#[test]
fn a_large_value_keeps_precision_to_2_pow_40() {
    assert_eq!(scale_units(1 << 40), 1_099_511_627_776.0);
}
