//! A frame counter. This does not build: version 2 requires a const
//! generic length.

pub use frame::Frame;

pub fn make_counter() -> Frame {
    Frame::new()
}
