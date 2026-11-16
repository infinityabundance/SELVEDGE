//! State construction.

pub struct State {
    pub count: i32,
}

pub fn initial() -> State {
    State { count: 0 }
}
