use deps_13_wrong_lib_path::clamp_percent;

#[test]
fn half_is_fifty() {
    assert_eq!(clamp_percent(1, 2), 50);
}

#[test]
fn an_overshoot_is_clamped() {
    assert_eq!(clamp_percent(3, 2), 100);
}

#[test]
fn a_zero_total_is_zero() {
    assert_eq!(clamp_percent(0, 0), 0);
}
