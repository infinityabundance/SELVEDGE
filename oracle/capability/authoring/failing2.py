"""Failing-test debugging, batch two (five tasks).

Each initial state compiles and its own visible unit test fails for the
documented reason. The hidden tests are a superset, so a repair that only makes
the visible assertion pass is still rejected.
"""

from .common import code

TASKS = [
    {
        "id": "failing-05-inclusive-range",
        "category": "failing-test",
        "difficulty": "easy",
        "capability": "Debug an exclusive range used where the contract is inclusive.",
        "prompt_intro": (
            "The visible unit test fails. `range_sum(start, end)` must sum every "
            "integer from `start` through `end`, inclusive. A range whose start is "
            "greater than its end is empty. Keep the signature "
            "`pub fn range_sum(start: i32, end: i32) -> i32`."),
        "observed_failure": (
            "thread 'tests::sums_inclusively' panicked at src/lib.rs:15:9:\n"
            "assertion `left == right` failed\n"
            "  left: 6\n"
            " right: 10\n"
            "\n"
            "test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out"),
        "files": {
            "src/lib.rs": code("""
                //! Inclusive range sums. The visible test fails: the upper bound is
                //! excluded.

                pub fn range_sum(start: i32, end: i32) -> i32 {
                    (start..end).sum()
                }

                #[cfg(test)]
                mod tests {
                    use super::range_sum;

                    #[test]
                    fn sums_inclusively() {
                        assert_eq!(range_sum(1, 4), 10);
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use failing_05_inclusive_range::range_sum;

                #[test]
                fn sums_both_bounds() {
                    assert_eq!(range_sum(1, 4), 10);
                }

                #[test]
                fn a_single_point_is_that_point() {
                    assert_eq!(range_sum(3, 3), 3);
                }

                #[test]
                fn reversed_bounds_are_empty() {
                    assert_eq!(range_sum(4, 1), 0);
                }

                #[test]
                fn negative_ranges_work() {
                    assert_eq!(range_sum(-2, 2), 0);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Inclusive range sums.

                pub fn range_sum(start: i32, end: i32) -> i32 {
                    (start..=end).sum()
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "(start..=end)", "replace": "(start..end)"},
        "initial_failure": {"kind": "test_failure", "must_contain": ["test result: FAILED"]},
    },
    {
        "id": "failing-06-integer-precision",
        "category": "failing-test",
        "difficulty": "medium",
        "capability": "Move an integer division after the multiplication that needs the precision.",
        "prompt_intro": (
            "The visible unit test fails. `percent_of(cents, percent)` must return "
            "`cents * percent / 100`, truncated to whole cents. The current code "
            "divides before multiplying, so it loses every remainder. Keep the "
            "signature `pub fn percent_of(cents: u64, percent: u64) -> u64`."),
        "observed_failure": (
            "thread 'tests::keeps_the_remainder' panicked at src/lib.rs:16:9:\n"
            "assertion `left == right` failed\n"
            "  left: 50\n"
            " right: 99\n"
            "\n"
            "test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out"),
        "files": {
            "src/lib.rs": code("""
                //! Percentage arithmetic. The visible test fails because the division
                //! happens before the multiplication.

                pub fn percent_of(cents: u64, percent: u64) -> u64 {
                    cents / 100 * percent
                }

                #[cfg(test)]
                mod tests {
                    use super::percent_of;

                    #[test]
                    fn keeps_the_remainder() {
                        assert_eq!(percent_of(199, 50), 99);
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use failing_06_integer_precision::percent_of;

                #[test]
                fn keeps_the_remainder_for_an_odd_amount() {
                    assert_eq!(percent_of(199, 50), 99);
                }

                #[test]
                fn exact_hundreds_are_unchanged() {
                    assert_eq!(percent_of(200, 50), 100);
                }

                #[test]
                fn one_percent_of_one_cent_truncates_to_zero() {
                    assert_eq!(percent_of(1, 1), 0);
                }

                #[test]
                fn full_percent_is_the_whole_amount() {
                    assert_eq!(percent_of(1, 100), 1);
                }

                #[test]
                fn zero_percent_is_zero() {
                    assert_eq!(percent_of(10_000, 0), 0);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Percentage arithmetic.

                pub fn percent_of(cents: u64, percent: u64) -> u64 {
                    cents * percent / 100
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "cents * percent / 100",
                     "replace": "cents * percent / 99"},
        "initial_failure": {"kind": "test_failure", "must_contain": ["test result: FAILED"]},
    },
    {
        "id": "failing-07-case-insensitive-word",
        "category": "failing-test",
        "difficulty": "easy",
        "capability": "Match the first word case-insensitively and reject an empty needle.",
        "prompt_intro": (
            "The visible unit test fails. `is_command(text, name)` must report "
            "whether the first whitespace-delimited word of `text` equals `name`, "
            "ignoring ASCII case. An empty `name` is never a command. Keep the "
            "signature `pub fn is_command(text: &str, name: &str) -> bool`."),
        "observed_failure": (
            "thread 'tests::matches_regardless_of_case' panicked at src/lib.rs:17:9:\n"
            "assertion failed: is_command(\"RUN now\", \"run\")\n"
            "\n"
            "test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out"),
        "files": {
            "src/lib.rs": code("""
                //! Command detection. The visible test fails: matching is
                //! case-sensitive and an empty name matches an empty text.

                pub fn is_command(text: &str, name: &str) -> bool {
                    text.split_whitespace().next().map(|word| word == name).unwrap_or(false)
                }

                #[cfg(test)]
                mod tests {
                    use super::is_command;

                    #[test]
                    fn matches_regardless_of_case() {
                        assert!(is_command("RUN now", "run"));
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use failing_07_case_insensitive_word::is_command;

                #[test]
                fn matches_uppercase_input() {
                    assert!(is_command("RUN now", "run"));
                    assert!(is_command("Run now", "RUN"));
                }

                #[test]
                fn only_the_first_word_matches() {
                    assert!(!is_command("please run", "run"));
                }

                #[test]
                fn a_prefix_is_not_a_word_match() {
                    assert!(!is_command("running", "run"));
                }

                #[test]
                fn an_empty_name_is_not_a_command() {
                    assert!(!is_command("", ""));
                    assert!(!is_command("run", ""));
                }

                #[test]
                fn blank_text_is_not_a_command() {
                    assert!(!is_command("   ", "run"));
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Command detection.

                pub fn is_command(text: &str, name: &str) -> bool {
                    if name.is_empty() {
                        return false;
                    }
                    text.split_whitespace()
                        .next()
                        .map(|word| word.eq_ignore_ascii_case(name))
                        .unwrap_or(false)
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "word.eq_ignore_ascii_case(name)",
                     "replace": "word == name"},
        "initial_failure": {"kind": "test_failure", "must_contain": ["test result: FAILED"]},
    },
    {
        "id": "failing-08-lower-bound",
        "category": "failing-test",
        "difficulty": "hard",
        "capability": "Fix a binary-search return value without breaking insertion points.",
        "prompt_intro": (
            "The visible unit test fails. `lower_bound(values, target)` returns the "
            "index of the first element equal to `target`, or the index at which "
            "`target` would be inserted to keep the sorted slice ordered. Return "
            "`values.len()` when every element is smaller. Keep the signature "
            "`pub fn lower_bound(values: &[i32], target: i32) -> usize`."),
        "observed_failure": (
            "thread 'tests::finds_a_present_value' panicked at src/lib.rs:17:9:\n"
            "assertion `left == right` failed\n"
            "  left: 2\n"
            " right: 1\n"
            "\n"
            "test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out"),
        "files": {
            "src/lib.rs": code("""
                //! Lower-bound search. The visible test fails: a present value is
                //! reported one index too far.

                pub fn lower_bound(values: &[i32], target: i32) -> usize {
                    let mut low = 0;
                    let mut high = values.len();
                    while low < high {
                        let mid = (low + high) / 2;
                        if values[mid] < target {
                            low = mid + 1;
                        } else {
                            high = mid;
                        }
                    }
                    if low < values.len() && values[low] == target {
                        low + 1
                    } else {
                        low
                    }
                }

                #[cfg(test)]
                mod tests {
                    use super::lower_bound;

                    #[test]
                    fn finds_a_present_value() {
                        assert_eq!(lower_bound(&[1, 3, 5], 3), 1);
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use failing_08_lower_bound::lower_bound;

                #[test]
                fn finds_a_present_value_at_its_index() {
                    assert_eq!(lower_bound(&[1, 3, 5], 3), 1);
                }

                #[test]
                fn finds_the_first_duplicate() {
                    assert_eq!(lower_bound(&[1, 2, 2, 2, 3], 2), 1);
                }

                #[test]
                fn reports_the_insertion_point_for_an_absent_value() {
                    assert_eq!(lower_bound(&[1, 3, 5], 4), 2);
                }

                #[test]
                fn an_empty_slice_has_index_zero() {
                    assert_eq!(lower_bound(&[], 9), 0);
                }

                #[test]
                fn a_value_before_everything_has_index_zero() {
                    assert_eq!(lower_bound(&[4, 5, 6], 1), 0);
                }

                #[test]
                fn a_value_after_everything_has_length_index() {
                    assert_eq!(lower_bound(&[1, 2], 9), 2);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Lower-bound search.

                pub fn lower_bound(values: &[i32], target: i32) -> usize {
                    let mut low = 0;
                    let mut high = values.len();
                    while low < high {
                        let mid = (low + high) / 2;
                        if values[mid] < target {
                            low = mid + 1;
                        } else {
                            high = mid;
                        }
                    }
                    low
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "values[mid] < target",
                     "replace": "values[mid] <= target"},
        "initial_failure": {"kind": "test_failure", "must_contain": ["test result: FAILED"]},
    },
    {
        "id": "failing-09-state-cycle",
        "category": "failing-test",
        "difficulty": "easy",
        "capability": "Repair a transition table so every state follows the declared cycle.",
        "prompt_intro": (
            "The visible unit test fails. The traffic light cycles "
            "`Red -> Green -> Yellow -> Red`. `next(light)` returns the following "
            "state. Keep the enum and the signature "
            "`pub fn next(light: Light) -> Light`."),
        "observed_failure": (
            "thread 'tests::green_follows_red_then_yellow' panicked at src/lib.rs:21:9:\n"
            "assertion `left == right` failed\n"
            "  left: Green\n"
            " right: Yellow\n"
            "\n"
            "test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out"),
        "files": {
            "src/lib.rs": code("""
                //! Traffic-light transitions. The visible test fails: Green skips
                //! Yellow.

                #[derive(Clone, Copy, Debug, PartialEq, Eq)]
                pub enum Light {
                    Red,
                    Green,
                    Yellow,
                }

                pub fn next(light: Light) -> Light {
                    match light {
                        Light::Red => Light::Green,
                        Light::Green => Light::Red,
                        Light::Yellow => Light::Green,
                    }
                }

                #[cfg(test)]
                mod tests {
                    use super::{next, Light};

                    #[test]
                    fn green_follows_red_then_yellow() {
                        assert_eq!(next(Light::Red), Light::Green);
                        assert_eq!(next(Light::Green), Light::Yellow);
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use failing_09_state_cycle::{next, Light};

                #[test]
                fn red_to_green() {
                    assert_eq!(next(Light::Red), Light::Green);
                }

                #[test]
                fn green_to_yellow() {
                    assert_eq!(next(Light::Green), Light::Yellow);
                }

                #[test]
                fn yellow_to_red() {
                    assert_eq!(next(Light::Yellow), Light::Red);
                }

                #[test]
                fn three_steps_return_home() {
                    let start = Light::Red;
                    assert_eq!(next(next(next(start))), start);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Traffic-light transitions.

                #[derive(Clone, Copy, Debug, PartialEq, Eq)]
                pub enum Light {
                    Red,
                    Green,
                    Yellow,
                }

                pub fn next(light: Light) -> Light {
                    match light {
                        Light::Red => Light::Green,
                        Light::Green => Light::Yellow,
                        Light::Yellow => Light::Red,
                    }
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "Light::Green => Light::Yellow",
                     "replace": "Light::Green => Light::Red"},
        "initial_failure": {"kind": "test_failure", "must_contain": ["test result: FAILED"]},
    },
]
