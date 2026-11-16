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
