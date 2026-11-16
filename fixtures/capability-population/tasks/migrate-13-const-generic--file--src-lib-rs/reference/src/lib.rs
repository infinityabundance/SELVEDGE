//! A frame counter.

pub use frame::Frame;

pub fn make_counter() -> Frame<4> {
    Frame::new()
}
