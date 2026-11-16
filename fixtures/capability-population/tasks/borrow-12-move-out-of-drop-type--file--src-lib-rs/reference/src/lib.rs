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
