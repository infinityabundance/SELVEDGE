//! Numeric state codes.

pub use status::State;

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use crate::{code, State};

    #[test]
    fn live_is_one() {
        assert_eq!(code(State::Live), 1);
    }

    #[test]
    fn paused_is_two() {
        assert_eq!(code(State::Paused), 2);
    }

    #[test]
    fn draft_is_zero() {
        assert_eq!(code(State::Draft), 0);
    }
}
