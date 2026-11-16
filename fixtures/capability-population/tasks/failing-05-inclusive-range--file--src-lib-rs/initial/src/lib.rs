//! Inclusive range sums. The visible test fails: the upper bound is
//! excluded.

pub fn range_sum(start: i32, end: i32) -> i32 {
    (start..end).sum()
}

#[cfg(test)]
mod tests {
    use super::range_sum;

    #[test]
    fn sums_inclusively() {
        assert_eq!(range_sum(1, 4), 10);
    }
}
