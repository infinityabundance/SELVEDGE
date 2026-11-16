//! Numeric state codes.

pub use status::State;

pub fn code(state: State) -> u8 {
    match state {
        State::Live => 1,
        State::Paused => 2,
        State::Draft => 0,
    }
}
