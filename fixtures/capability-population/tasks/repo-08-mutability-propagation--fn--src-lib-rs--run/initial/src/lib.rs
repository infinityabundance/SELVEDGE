//! A state pipeline.

pub mod report;
pub mod state;
pub mod update;

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use crate::{report::count, state, update::bump, run};

    #[test]
    fn run_bumps_twice() {
        assert_eq!(run(), 2);
    }

    #[test]
    fn initial_state_is_zero() {
        assert_eq!(count(&state::initial()), 0);
    }

    #[test]
    fn bump_accumulates() {
        let mut state = state::initial();
        bump(&mut state, 5);
        bump(&mut state, -2);
        assert_eq!(count(&state), 3);
    }
}
