//! Buffer hand-off. The crate does not build: `Buffer` implements
//! `Drop`, so its field cannot be moved out.

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

pub fn into_data(buffer: Buffer) -> Vec<u8> {
    buffer.data
}
