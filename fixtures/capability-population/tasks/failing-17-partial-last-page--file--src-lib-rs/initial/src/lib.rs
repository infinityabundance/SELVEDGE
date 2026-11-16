//! Page bounds. The visible test fails: the last page is not clamped.

pub fn page_bounds(total: usize, page_size: usize, page: usize) -> (usize, usize) {
    if page_size == 0 || page == 0 {
        return (0, 0);
    }
    let start = (page - 1) * page_size;
    if start >= total {
        return (total, total);
    }
    (start, start + page_size)
}

#[cfg(test)]
mod tests {
    use super::page_bounds;

    #[test]
    fn visible_partial_page() {
        assert_eq!(page_bounds(5, 2, 3), (4, 5));
    }
}
