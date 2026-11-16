use traits_06_missing_generic_bound::{total, Measure, Reading};

#[test]
fn sums_measurements() {
    assert_eq!(total(&[Reading(1), Reading(2), Reading(3)]), 6);
}

#[test]
fn empty_is_zero() {
    assert_eq!(total::<Reading>(&[]), 0);
}

#[test]
fn works_with_another_measure() {
    struct Weight(u32);
    impl Measure for Weight {
        fn value(&self) -> u32 { self.0 }
    }
    assert_eq!(total(&[Weight(5), Weight(7)]), 12);
}
