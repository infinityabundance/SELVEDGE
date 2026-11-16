//! Even numbers below a limit.

pub fn evens(limit: i32) -> impl Iterator<Item = i32> {
    (0..limit).filter(|value| value % 2 == 0)
}

#[cfg(test)]
mod visible {
    use crate::evens;

    #[test]
    fn yields_the_even_numbers() {
        assert_eq!(evens(10).collect::<Vec<_>>(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn zero_limit_is_empty() {
        assert_eq!(evens(0).collect::<Vec<_>>(), Vec::<i32>::new());
    }

    #[test]
    fn one_limit_is_zero() {
        assert_eq!(evens(1).collect::<Vec<_>>(), vec![0]);
    }

    #[test]
    fn negative_limit_is_empty() {
        assert_eq!(evens(-5).collect::<Vec<_>>(), Vec::<i32>::new());
    }
}
