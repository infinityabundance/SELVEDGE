//! Frame, version 2. The length became a const generic parameter.

pub struct Frame<const N: usize> {
    pub cells: [u8; N],
}

impl<const N: usize> Frame<N> {
    pub fn new() -> Self {
        Self { cells: [0; N] }
    }

    pub fn set(&mut self, index: usize, value: u8) {
        if index < N {
            self.cells[index] = value;
        }
    }

    pub fn used(&self) -> usize {
        self.cells.iter().filter(|cell| **cell != 0).count()
    }
}
