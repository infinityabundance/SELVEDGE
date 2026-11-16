"""Compile-clean but behaviourally wrong repairs (nine tasks).

The initial state builds and its own visible test passes; the defect is only
visible to the hidden behavioural tests (`initial_failure.kind` is
`hidden_test_failure`). This category is deliberately separate from
`failing-test`, where a visible assertion is already red.
"""

from .common import code

HIDDEN_FAILURE = {"kind": "hidden_test_failure", "must_contain": ["test result: FAILED"]}

TASKS = [
    {
        "id": "semantic-01-inclusive-range",
        "category": "semantic-repair",
        "difficulty": "easy",
        "capability": "Notice an excluded endpoint that the visible test encoded as correct.",
        "prompt_intro": (
            "A user reports that `range_sum(1, 4)` returns 6, but the API contract "
            "says the sum includes both endpoints. The crate builds and its visible "
            "test passes. Fix the behaviour; the visible test is not authoritative. "
            "Keep the signature."),
        "observed_failure": (
            "the crate builds and tests pass; the user reports range_sum(1, 4) == 6\n"
            "but the documented contract requires 10"),
        "files": {
            "src/lib.rs": code("""
                //! Inclusive range sums. The visible test encodes the off-by-one
                //! behaviour as if it were correct.

                pub fn range_sum(start: i32, end: i32) -> i32 {
                    (start..end).sum()
                }

                #[cfg(test)]
                mod tests {
                    use super::range_sum;

                    #[test]
                    fn sums_the_visible_example() {
                        assert_eq!(range_sum(1, 4), 6);
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use semantic_01_inclusive_range::range_sum;

                #[test]
                fn includes_the_upper_bound() {
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
                fn negative_ranges_are_inclusive() {
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
        "initial_failure": dict(HIDDEN_FAILURE),
    },
    {
        "id": "semantic-02-rounded-average",
        "category": "semantic-repair",
        "difficulty": "easy",
        "capability": "Replace truncating division with the documented half-up rounding.",
        "prompt_intro": (
            "The average helper builds and its visible example passes, but the "
            "contract says the mean of integer values is rounded half up. "
            "`average(&[1, 2])` must return 2, not 1. Keep the signature "
            "`pub fn average(values: &[u32]) -> u32` and return 0 for an empty "
            "slice."),
        "observed_failure": (
            "the crate builds and tests pass; average(&[1, 2]) returns 1 where the\n"
            "contract requires half-up rounding to 2"),
        "files": {
            "src/lib.rs": code("""
                //! Integer averages, currently truncated.

                pub fn average(values: &[u32]) -> u32 {
                    if values.is_empty() {
                        return 0;
                    }
                    values.iter().sum::<u32>() / values.len() as u32
                }

                #[cfg(test)]
                mod tests {
                    use super::average;

                    #[test]
                    fn exact_mean_is_unchanged() {
                        assert_eq!(average(&[2, 4]), 3);
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use semantic_02_rounded_average::average;

                #[test]
                fn rounds_half_up() {
                    assert_eq!(average(&[1, 2]), 2);
                }

                #[test]
                fn exact_means_are_unchanged() {
                    assert_eq!(average(&[2, 4]), 3);
                }

                #[test]
                fn rounds_down_below_half() {
                    assert_eq!(average(&[1, 2, 3]), 2);
                }

                #[test]
                fn empty_is_zero() {
                    assert_eq!(average(&[]), 0);
                }

                #[test]
                fn single_value_is_itself() {
                    assert_eq!(average(&[7]), 7);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Integer averages with half-up rounding.

                pub fn average(values: &[u32]) -> u32 {
                    if values.is_empty() {
                        return 0;
                    }
                    let len = values.len() as u32;
                    (values.iter().sum::<u32>() + len / 2) / len
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "len / 2", "replace": "len / 3"},
        "initial_failure": dict(HIDDEN_FAILURE),
    },
    {
        "id": "semantic-03-case-insensitive-first-word",
        "category": "semantic-repair",
        "difficulty": "easy",
        "capability": "Apply case folding that the passing lowercase example does not exercise.",
        "prompt_intro": (
            "The command detector builds and its lowercase example passes, but the "
            "contract is case-insensitive. `is_command(\"RUN now\", \"run\")` must be "
            "true, an empty name is never a command, and only the first word is "
            "compared. Keep the signature."),
        "observed_failure": (
            "the crate builds and tests pass; is_command(\"RUN now\", \"run\") is false\n"
            "although the contract is case-insensitive"),
        "files": {
            "src/lib.rs": code("""
                //! Command detection, currently case-sensitive.

                pub fn is_command(text: &str, name: &str) -> bool {
                    text.split_whitespace().next().map(|word| word == name).unwrap_or(false)
                }

                #[cfg(test)]
                mod tests {
                    use super::is_command;

                    #[test]
                    fn matches_a_lowercase_word() {
                        assert!(is_command("run now", "run"));
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use semantic_03_case_insensitive_first_word::is_command;

                #[test]
                fn matches_regardless_of_case() {
                    assert!(is_command("RUN now", "run"));
                    assert!(is_command("Run now", "RUN"));
                }

                #[test]
                fn only_the_first_word_counts() {
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
                //! Case-insensitive command detection.

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
        "initial_failure": dict(HIDDEN_FAILURE),
    },
    {
        "id": "semantic-04-stable-score-sort",
        "category": "semantic-repair",
        "difficulty": "hard",
        "capability": "Add the documented tie-break that the visible data never triggers.",
        "prompt_intro": (
            "The ranking helper builds and its visible example passes because the "
            "scores are distinct. The contract is: score descending, ties by name "
            "ascending. Fix the tie-break. Keep the signature "
            "`pub fn ranked(entries: &[(String, u32)]) -> Vec<String>`."),
        "observed_failure": (
            "the crate builds and tests pass; equal scores keep input order where the\n"
            "contract requires name-ascending tie-break"),
        "files": {
            "src/lib.rs": code("""
                //! Score ranking without a tie-break.

                pub fn ranked(entries: &[(String, u32)]) -> Vec<String> {
                    let mut sorted = entries.to_vec();
                    sorted.sort_by(|left, right| right.1.cmp(&left.1));
                    sorted.into_iter().map(|(name, _score)| name).collect()
                }

                #[cfg(test)]
                mod tests {
                    use super::ranked;

                    #[test]
                    fn orders_distinct_scores() {
                        let entries = vec![("low".to_string(), 1), ("high".to_string(), 2)];
                        assert_eq!(ranked(&entries), vec!["high", "low"]);
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use semantic_04_stable_score_sort::ranked;

                #[test]
                fn orders_by_score_descending() {
                    let entries = vec![("a".to_string(), 1), ("b".to_string(), 3)];
                    assert_eq!(ranked(&entries), vec!["b", "a"]);
                }

                #[test]
                fn ties_break_by_name_ascending() {
                    let entries = vec![("z".to_string(), 5), ("a".to_string(), 5)];
                    assert_eq!(ranked(&entries), vec!["a", "z"]);
                }

                #[test]
                fn ties_with_many_entries_are_deterministic() {
                    let entries = vec![
                        ("c".to_string(), 2),
                        ("a".to_string(), 2),
                        ("b".to_string(), 2),
                    ];
                    assert_eq!(ranked(&entries), vec!["a", "b", "c"]);
                }

                #[test]
                fn empty_is_empty() {
                    assert!(ranked(&[]).is_empty());
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Score ranking with a deterministic tie-break.

                pub fn ranked(entries: &[(String, u32)]) -> Vec<String> {
                    let mut sorted = entries.to_vec();
                    sorted.sort_by(|left, right| {
                        right.1.cmp(&left.1).then(left.0.cmp(&right.0))
                    });
                    sorted.into_iter().map(|(name, _score)| name).collect()
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "left.0.cmp(&right.0)",
                     "replace": "right.0.cmp(&left.0)"},
        "initial_failure": dict(HIDDEN_FAILURE),
    },
    {
        "id": "semantic-05-saturating-multiply",
        "category": "semantic-repair",
        "difficulty": "medium",
        "capability": "Saturate a multiplication that the small visible example never pushes to the bound.",
        "prompt_intro": (
            "The scaling helper builds and its small example passes. The contract "
            "says the product saturates at `u32::MAX`; it must never panic or wrap. "
            "Fix `scale` while keeping `pub fn scale(value: u32, factor: u32) -> u32`."),
        "observed_failure": (
            "the crate builds and tests pass; scale(u32::MAX / 2 + 1, 2) panics with\n"
            "'attempt to multiply with overflow' where the contract requires saturation"),
        "files": {
            "src/lib.rs": code("""
                //! Scaling, currently unchecked.

                pub fn scale(value: u32, factor: u32) -> u32 {
                    value * factor
                }

                #[cfg(test)]
                mod tests {
                    use super::scale;

                    #[test]
                    fn small_products_are_exact() {
                        assert_eq!(scale(2, 3), 6);
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use semantic_05_saturating_multiply::scale;

                #[test]
                fn small_products_are_exact_by_multiplication() {
                    assert_eq!(scale(2, 3), 6);
                }

                #[test]
                fn zero_absorbs() {
                    assert_eq!(scale(u32::MAX, 0), 0);
                }

                #[test]
                fn products_past_the_bound_saturate() {
                    assert_eq!(scale(u32::MAX / 2 + 1, 2), u32::MAX);
                }

                #[test]
                fn the_bound_times_one_is_the_bound() {
                    assert_eq!(scale(u32::MAX, 1), u32::MAX);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Saturating scaling.

                pub fn scale(value: u32, factor: u32) -> u32 {
                    value.saturating_mul(factor)
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "value.saturating_mul(factor)",
                     "replace": "value.wrapping_mul(factor)"},
        "initial_failure": dict(HIDDEN_FAILURE),
    },
    {
        "id": "semantic-06-order-preserving-unique",
        "category": "semantic-repair",
        "difficulty": "hard",
        "capability": "Preserve first-occurrence order where sorting happens to pass the visible case.",
        "prompt_intro": (
            "The de-duplication helper builds and its sorted example passes. The "
            "contract is first-occurrence order, not sorted order: "
            "`unique(&[3, 1, 3, 2, 1]) == [3, 1, 2]`. Fix the helper. Keep "
            "`pub fn unique(values: &[i32]) -> Vec<i32>`."),
        "observed_failure": (
            "the crate builds and tests pass; unique(&[3, 1, 3, 2, 1]) returns\n"
            "[1, 2, 3] where the contract requires first-occurrence order"),
        "files": {
            "src/lib.rs": code("""
                //! De-duplication by sorting, which loses first-occurrence order.

                pub fn unique(values: &[i32]) -> Vec<i32> {
                    let mut sorted = values.to_vec();
                    sorted.sort();
                    sorted.dedup();
                    sorted
                }

                #[cfg(test)]
                mod tests {
                    use super::unique;

                    #[test]
                    fn removes_duplicates_from_sorted_input() {
                        assert_eq!(unique(&[1, 1, 2]), vec![1, 2]);
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use semantic_06_order_preserving_unique::unique;

                #[test]
                fn keeps_first_occurrence_order() {
                    assert_eq!(unique(&[3, 1, 3, 2, 1]), vec![3, 1, 2]);
                }

                #[test]
                fn sorted_input_is_unchanged() {
                    assert_eq!(unique(&[1, 1, 2]), vec![1, 2]);
                }

                #[test]
                fn all_duplicates_collapse_to_one() {
                    assert_eq!(unique(&[5, 5, 5]), vec![5]);
                }

                #[test]
                fn empty_is_empty() {
                    assert!(unique(&[]).is_empty());
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Order-preserving de-duplication.

                use std::collections::HashSet;

                pub fn unique(values: &[i32]) -> Vec<i32> {
                    let mut seen = HashSet::new();
                    let mut result = Vec::new();
                    for value in values {
                        if seen.insert(*value) {
                            result.push(*value);
                        }
                    }
                    result
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "if seen.insert(*value)",
                     "replace": "if !seen.insert(*value)"},
        "initial_failure": dict(HIDDEN_FAILURE),
    },
    {
        "id": "semantic-07-csv-quoting",
        "category": "semantic-repair",
        "difficulty": "hard",
        "capability": "Escape embedded quotes and quote newline fields, not only comma fields.",
        "prompt_intro": (
            "The CSV field encoder builds and its comma example passes. The contract "
            "requires quoting when the field contains a comma, a double quote or a "
            "newline, and every embedded double quote must be doubled. Fix `field`. "
            "Keep `pub fn field(value: &str) -> String`."),
        "observed_failure": (
            "the crate builds and tests pass; field(\"a\\\"b\") returns `a\"b` where the\n"
            "contract requires `\"a\"\"b\"`"),
        "files": {
            "src/lib.rs": code("""
                //! CSV field quoting, currently only for commas.

                pub fn field(value: &str) -> String {
                    if value.contains(',') {
                        format!("\\"{value}\\"")
                    } else {
                        value.to_string()
                    }
                }

                #[cfg(test)]
                mod tests {
                    use super::field;

                    #[test]
                    fn quotes_a_comma_field() {
                        assert_eq!(field("a,b"), "\\"a,b\\"");
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use semantic_07_csv_quoting::field;

                #[test]
                fn plain_fields_are_unquoted() {
                    assert_eq!(field("plain"), "plain");
                }

                #[test]
                fn comma_fields_are_quoted() {
                    assert_eq!(field("a,b"), "\\"a,b\\"");
                }

                #[test]
                fn embedded_quotes_are_doubled() {
                    assert_eq!(field("a\\"b"), "\\"a\\"\\"b\\"");
                }

                #[test]
                fn newline_fields_are_quoted() {
                    assert_eq!(field("a\\nb"), "\\"a\\nb\\"");
                }

                #[test]
                fn empty_is_unquoted() {
                    assert_eq!(field(""), "");
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! CSV field quoting.

                pub fn field(value: &str) -> String {
                    if value.contains(',') || value.contains('"') || value.contains('\\n') {
                        format!("\\"{}\\"", value.replace('"', "\\"\\""))
                    } else {
                        value.to_string()
                    }
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "value.replace('\"', \"\\\"\\\"\")",
                     "replace": "value.replace('\"', \"'\")"},
        "initial_failure": dict(HIDDEN_FAILURE),
    },
    {
        "id": "semantic-08-char-truncate",
        "category": "semantic-repair",
        "difficulty": "hard",
        "capability": "Truncate by characters, not bytes, so multi-byte input is not split.",
        "prompt_intro": (
            "The truncation helper builds and its ASCII example passes. The contract "
            "limits the result to `max_chars` characters; it must not split a UTF-8 "
            "character. Fix `truncate`. Keep "
            "`pub fn truncate(text: &str, max_chars: usize) -> String`."),
        "observed_failure": (
            "the crate builds and tests pass; truncate(\"héllo\", 2) panics with\n"
            "'byte index 2 is not a char boundary'"),
        "files": {
            "src/lib.rs": code("""
                //! Truncation by byte index.

                pub fn truncate(text: &str, max_chars: usize) -> String {
                    if text.len() <= max_chars {
                        text.to_string()
                    } else {
                        text[..max_chars].to_string()
                    }
                }

                #[cfg(test)]
                mod tests {
                    use super::truncate;

                    #[test]
                    fn truncates_ascii() {
                        assert_eq!(truncate("abcdef", 3), "abc");
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use semantic_08_char_truncate::truncate;

                #[test]
                fn truncates_ascii_exactly() {
                    assert_eq!(truncate("abcdef", 3), "abc");
                }

                #[test]
                fn does_not_split_a_multibyte_character() {
                    assert_eq!(truncate("héllo", 2), "hé");
                }

                #[test]
                fn zero_characters_is_empty() {
                    assert_eq!(truncate("abc", 0), "");
                }

                #[test]
                fn a_longer_limit_returns_everything() {
                    assert_eq!(truncate("hé", 10), "hé");
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Truncation by character count.

                pub fn truncate(text: &str, max_chars: usize) -> String {
                    text.chars().take(max_chars).collect()
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "take(max_chars)",
                     "replace": "take(max_chars + 1)"},
        "initial_failure": dict(HIDDEN_FAILURE),
    },
    {
        "id": "semantic-09-token-bucket-refill",
        "category": "semantic-repair",
        "difficulty": "hard",
        "capability": "Refill proportionally to elapsed time and respect the capacity bound.",
        "prompt_intro": (
            "The token-bucket helper builds and its one-second example passes. The "
            "contract refills proportionally for sub-second elapsed time and never "
            "exceeds `capacity`. `refill(0, 500, 10, 100)` must return 5 and "
            "`refill(90, 1000, 20, 100)` must return 100. Fix `refill`. Keep the "
            "signature."),
        "observed_failure": (
            "the crate builds and tests pass; refill(0, 500, 10, 100) returns 0 where\n"
            "proportional refill requires 5, and the capacity bound is ignored"),
        "files": {
            "src/lib.rs": code("""
                //! Token-bucket refill, currently whole-second and uncapped.

                pub fn refill(tokens: u32, elapsed_ms: u64, rate_per_sec: u32, capacity: u32) -> u32 {
                    let gained = (elapsed_ms / 1000) * rate_per_sec as u64;
                    tokens.saturating_add(gained as u32)
                }

                #[cfg(test)]
                mod tests {
                    use super::refill;

                    #[test]
                    fn one_second_refills_the_rate() {
                        assert_eq!(refill(0, 1000, 5, 10), 5);
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use semantic_09_token_bucket_refill::refill;

                #[test]
                fn one_second_refills_the_whole_rate() {
                    assert_eq!(refill(0, 1000, 5, 10), 5);
                }

                #[test]
                fn sub_second_time_refills_proportionally() {
                    assert_eq!(refill(0, 500, 10, 100), 5);
                }

                #[test]
                fn the_capacity_bound_is_respected() {
                    assert_eq!(refill(90, 1000, 20, 100), 100);
                }

                #[test]
                fn no_time_gains_nothing() {
                    assert_eq!(refill(7, 0, 10, 100), 7);
                }

                #[test]
                fn quarter_second_refills_a_quarter() {
                    assert_eq!(refill(0, 250, 8, 100), 2);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Proportional, capped token-bucket refill.

                pub fn refill(tokens: u32, elapsed_ms: u64, rate_per_sec: u32, capacity: u32) -> u32 {
                    let gained = (elapsed_ms.saturating_mul(rate_per_sec as u64) / 1000) as u32;
                    tokens.saturating_add(gained).min(capacity)
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": ".min(capacity)",
                     "replace": ".min(capacity.saturating_sub(1))"},
        "initial_failure": dict(HIDDEN_FAILURE),
    },
]
