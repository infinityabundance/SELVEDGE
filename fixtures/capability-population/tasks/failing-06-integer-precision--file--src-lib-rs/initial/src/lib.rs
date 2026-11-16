//! Percentage arithmetic. The visible test fails because the division
//! happens before the multiplication.

pub fn percent_of(cents: u64, percent: u64) -> u64 {
    cents / 100 * percent
}

#[cfg(test)]
mod tests {
    use super::percent_of;

    #[test]
    fn keeps_the_remainder() {
        assert_eq!(percent_of(199, 50), 99);
    }
}
