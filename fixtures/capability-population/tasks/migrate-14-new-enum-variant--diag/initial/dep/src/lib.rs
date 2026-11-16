//! Status, version 2. Version 2 added the `Draft` state.

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum State {
    Live,
    Paused,
    Draft,
}
