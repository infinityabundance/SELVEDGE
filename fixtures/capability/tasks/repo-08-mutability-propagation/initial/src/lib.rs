//! A state pipeline.

pub mod report;
pub mod state;
pub mod update;

pub fn run() -> i32 {
    let state = state::initial();
    update::bump(&state, 1);
    update::bump(&state, 1);
    report::count(&state)
}
