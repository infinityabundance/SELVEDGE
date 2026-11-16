use failing_14_century_leap_year::days_in_month;

#[test]
fn centuries_that_are_not_leap_years() {
    assert_eq!(days_in_month(1900, 2), 28);
    assert_eq!(days_in_month(2100, 2), 28);
}

#[test]
fn the_four_hundred_year_rule() {
    assert_eq!(days_in_month(2000, 2), 29);
    assert_eq!(days_in_month(1600, 2), 29);
}

#[test]
fn ordinary_years() {
    assert_eq!(days_in_month(2023, 2), 28);
    assert_eq!(days_in_month(2024, 2), 29);
}

#[test]
fn month_lengths_and_invalid_months() {
    assert_eq!(days_in_month(2023, 4), 30);
    assert_eq!(days_in_month(2023, 12), 31);
    assert_eq!(days_in_month(2023, 0), 0);
    assert_eq!(days_in_month(2023, 13), 0);
}
