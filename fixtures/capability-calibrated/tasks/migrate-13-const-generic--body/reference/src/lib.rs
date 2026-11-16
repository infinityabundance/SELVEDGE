//! A frame counter.

pub use frame::Frame;

pub fn make_counter() -> Frame<4> {
    
    Frame::new()

}

#[cfg(test)]
mod visible {
    use super::*;

    #[test]
    fn a_fresh_frame_has_no_used_cells() {
        assert_eq!(make_counter().used(), 0);
    }
}
