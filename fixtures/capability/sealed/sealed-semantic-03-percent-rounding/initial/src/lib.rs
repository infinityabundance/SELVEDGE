//! Whole-number percentages with truncating division.

pub fn percent(part: u64, total: u64) -> u64 {
    if total == 0 {
        return 0;
    }
    part * 100 / total
}

#[cfg(test)]
mod tests {
    use super::percent;

    #[test]
    fn visible_exact_half() {
        assert_eq!(percent(1, 2), 50);
    }
}
