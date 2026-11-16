//! Buffer hand-off.

pub struct Buffer {
    pub data: Vec<u8>,
}

impl Buffer {
    pub fn new(data: Vec<u8>) -> Self {
        Self { data }
    }
}

impl Drop for Buffer {
fn drop(&mut self) {
        self.data.clear();
    }
}

pub fn into_data(mut buffer: Buffer) -> Vec<u8> {
    std::mem::take(&mut buffer.data)
}

#[cfg(test)]
mod visible {
    use crate::{into_data, Buffer};

    #[test]
    fn returns_the_buffered_bytes() {
        assert_eq!(into_data(Buffer::new(vec![1, 2, 3])), vec![1, 2, 3]);
    }

    #[test]
    fn an_empty_buffer_returns_no_bytes() {
        assert!(into_data(Buffer::new(Vec::new())).is_empty());
    }

    #[test]
    fn repeated_hand_offs_are_independent() {
        let first = into_data(Buffer::new(vec![9]));
        let second = into_data(Buffer::new(vec![4, 5]));
        assert_eq!(first, vec![9]);
        assert_eq!(second, vec![4, 5]);
    }
}
