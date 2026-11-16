//! Chunk arithmetic. The visible test fails: the count is short by one
//! whenever the length is not an exact multiple of the chunk size.

pub fn chunk_count(len: usize, size: usize) -> usize {
    if size == 0 {
        return 0;
    }
    len / size
}

#[cfg(test)]
mod tests {
    use super::chunk_count;

    #[test]
    fn five_items_in_chunks_of_two_need_three_chunks() {
        assert_eq!(chunk_count(5, 2), 3);
    }
}
