//! Days in a month. The visible test fails: the century rule is
//! missing from the leap-year test.

pub fn days_in_month(year: u32, month: u32) -> u32 {
    let leap = year % 4 == 0;
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::days_in_month;

    #[test]
    fn visible_nineteen_hundred() {
        assert_eq!(days_in_month(1900, 2), 28);
    }
}
