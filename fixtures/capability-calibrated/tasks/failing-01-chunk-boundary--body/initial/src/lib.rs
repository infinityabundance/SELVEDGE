//! Chunk arithmetic.

pub fn chunk_count(len: usize, size: usize) -> usize {
    todo!()
}

#[cfg(test)]
mod visible {
    use super::*;

    #[test]
    fn exact_multiple() {
        assert_eq!(chunk_count(6, 3), 2);
    }
}
