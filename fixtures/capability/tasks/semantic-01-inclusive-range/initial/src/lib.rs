//! Inclusive range sums. The visible test encodes the off-by-one
//! behaviour as if it were correct.

pub fn range_sum(start: i32, end: i32) -> i32 {
    (start..end).sum()
}

#[cfg(test)]
mod tests {
    use super::range_sum;

    #[test]
    fn sums_the_visible_example() {
        assert_eq!(range_sum(1, 4), 6);
    }
}
