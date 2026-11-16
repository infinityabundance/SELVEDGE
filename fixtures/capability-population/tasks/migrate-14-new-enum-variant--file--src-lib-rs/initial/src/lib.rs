//! Numeric state codes. This does not build: version 2 added a
//! variant the match does not cover.

pub use status::State;

pub fn code(state: State) -> u8 {
    match state {
        State::Live => 1,
        State::Paused => 2,
    }
}
