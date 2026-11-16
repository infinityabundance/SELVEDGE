//! Chunk arithmetic.

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use crate::chunk_count;

    #[test]
    fn exact_multiple() {
        assert_eq!(chunk_count(6, 3), 2);
    }

    #[test]
    fn empty_input() {
        assert_eq!(chunk_count(0, 4), 0);
    }

    #[test]
    fn zero_size() {
        assert_eq!(chunk_count(9, 0), 0);
    }

    #[test]
    fn partial_chunk() {
        assert_eq!(chunk_count(5, 2), 3);
    }

    #[test]
    fn does_not_overflow() {
        assert_eq!(chunk_count(usize::MAX, 1), usize::MAX);
    }
}
