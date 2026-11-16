//! Page bounds.

pub fn page_bounds(total: usize, page_size: usize, page: usize) -> (usize, usize) {
    if page_size == 0 || page == 0 {
        return (0, 0);
    }
    let start = (page - 1) * page_size;
    if start >= total {
        return (total, total);
    }
    (start, (start + page_size).min(total))
}

#[cfg(test)]
mod visible {
    use crate::page_bounds;

    #[test]
    fn full_pages() {
        assert_eq!(page_bounds(5, 2, 1), (0, 2));
        assert_eq!(page_bounds(5, 2, 2), (2, 4));
    }

    #[test]
    fn the_partial_last_page_is_clamped() {
        assert_eq!(page_bounds(5, 2, 3), (4, 5));
    }

    #[test]
    fn pages_outside_the_range_are_empty() {
        assert_eq!(page_bounds(5, 2, 4), (5, 5));
        assert_eq!(page_bounds(0, 3, 1), (0, 0));
        assert_eq!(page_bounds(5, 0, 1), (0, 0));
    }
}
