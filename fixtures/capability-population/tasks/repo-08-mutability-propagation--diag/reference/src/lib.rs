//! A state pipeline.

pub mod report;
pub mod state;
pub mod update;

pub fn run() -> i32 {
    let mut state = state::initial();
    update::bump(&mut state, 1);
    update::bump(&mut state, 1);
    report::count(&state)
}
