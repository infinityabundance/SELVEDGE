//! Longest common prefix.

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use crate::common_prefix;

    #[test]
    fn finds_the_shared_prefix() {
        assert_eq!(common_prefix(&["flower", "flow", "flight"]), "fl");
    }

    #[test]
    fn identical_inputs_are_the_prefix() {
        assert_eq!(common_prefix(&["same", "same"]), "same");
    }

    #[test]
    fn no_shared_start_is_empty() {
        assert_eq!(common_prefix(&["alpha", "beta"]), "");
    }

    #[test]
    fn an_empty_slice_is_empty() {
        assert_eq!(common_prefix(&[]), "");
    }

    #[test]
    fn a_single_input_is_its_own_prefix() {
        assert_eq!(common_prefix(&["x"]), "x");
    }
}
