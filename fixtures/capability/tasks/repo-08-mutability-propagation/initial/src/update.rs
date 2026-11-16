//! State updates.

use crate::state::State;

pub fn bump(state: &State, by: i32) {
    state.count += by;
}
