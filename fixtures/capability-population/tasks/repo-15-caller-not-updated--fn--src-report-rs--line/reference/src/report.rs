//! A report line.

pub fn line(part: u32) -> String {
    crate::format::ratio(part, 10)
}

#[cfg(test)]
mod visible {
    use crate::line;

    #[test]
    fn half_of_ten_is_fifty_percent() {
        assert_eq!(line(5), "50.0%");
    }

    #[test]
    fn zero_is_zero_percent() {
        assert_eq!(line(0), "0.0%");
    }

    #[test]
    fn the_whole_is_one_hundred_percent() {
        assert_eq!(line(10), "100.0%");
    }
}
