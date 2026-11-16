//! Event kinds.

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Event {
    Start,
    Stop,
    Pause,
}
