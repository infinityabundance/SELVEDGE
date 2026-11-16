//! Chunk arithmetic.

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use super::*;

    #[test]
    fn exact_multiple() {
        assert_eq!(chunk_count(6, 3), 2);
    }
}
