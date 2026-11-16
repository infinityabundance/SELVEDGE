//! Days in a month.

pub fn days_in_month(year: u32, month: u32) -> u32 {
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => 0,
    }
}

#[cfg(test)]
mod visible {
    use crate::days_in_month;

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
}
