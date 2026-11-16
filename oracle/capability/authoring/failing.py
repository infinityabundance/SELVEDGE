"""Failing-test debugging (four tasks).

The initial state compiles and its visible test (or its own run) fails. The
hidden tests are a superset: a repair that only satisfies the visible assertion
still fails.
"""

TASKS = [
    {
        "id": "failing-01-chunk-boundary",
        "category": "failing-test",
        "capability": "Debug a rounding boundary in a chunk count, including overflow safety.",
        "prompt_intro": (
            "The visible unit test fails. `chunk_count` must return the number of "
            "chunks needed to hold `len` items in groups of `size`, treating a size "
            "of zero as zero chunks, and it must not overflow for any `usize` "
            "input. Keep the signature."),
        "observed_failure": (
            "running 1 test\n"
            "test tests::five_items_in_chunks_of_two_need_three_chunks ... FAILED\n"
            "\n"
            "thread 'tests::five_items_in_chunks_of_two_need_three_chunks' panicked at\n"
            "src/lib.rs:16:9:\n"
            "assertion `left == right` failed\n"
            "  left: 2\n"
            " right: 3\n"
            "\n"
            "test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out"),
        "files": {
            "src/lib.rs": (
                "//! Chunk arithmetic. The visible test fails: the count is short by one\n"
                "//! whenever the length is not an exact multiple of the chunk size.\n"
                "\n"
                "pub fn chunk_count(len: usize, size: usize) -> usize {\n"
                "    if size == 0 {\n"
                "        return 0;\n"
                "    }\n"
                "    len / size\n"
                "}\n"
                "\n"
                "#[cfg(test)]\n"
                "mod tests {\n"
                "    use super::chunk_count;\n"
                "\n"
                "    #[test]\n"
                "    fn five_items_in_chunks_of_two_need_three_chunks() {\n"
                "        assert_eq!(chunk_count(5, 2), 3);\n"
                "    }\n"
                "}\n"),
        },
        "hidden": {
            "tests/check.rs": (
                "use failing_01_chunk_boundary::chunk_count;\n"
                "\n"
                "#[test]\n"
                "fn exact_multiple() {\n"
                "    assert_eq!(chunk_count(6, 3), 2);\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn empty_input() {\n"
                "    assert_eq!(chunk_count(0, 4), 0);\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn zero_size() {\n"
                "    assert_eq!(chunk_count(9, 0), 0);\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn partial_chunk() {\n"
                "    assert_eq!(chunk_count(5, 2), 3);\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn does_not_overflow() {\n"
                "    assert_eq!(chunk_count(usize::MAX, 1), usize::MAX);\n"
                "}\n"),
        },
        "reference": {
            "src/lib.rs": (
                "//! Chunk arithmetic.\n"
                "\n"
                "pub fn chunk_count(len: usize, size: usize) -> usize {\n"
                "    if size == 0 {\n"
                "        return 0;\n"
                "    }\n"
                "    len.div_ceil(size)\n"
                "}\n"),
        },
        "mutation": {"file": "src/lib.rs", "find": "len.div_ceil(size)", "replace": "len / size"},
        "initial_failure": {"kind": "test_failure", "must_contain": ["test result: FAILED"]},
    },
    {
        "id": "failing-02-empty-input",
        "category": "failing-test",
        "capability": "Fix a missing empty/short-input case without changing the signature.",
        "prompt_intro": (
            "The visible test fails with an index-out-of-bounds panic. "
            "`max_gap` must return the largest difference between consecutive "
            "values, and 0 for slices with fewer than two elements. Keep the "
            "signature `pub fn max_gap(values: &[i64]) -> i64`."),
        "observed_failure": (
            "thread 'tests::single_value_has_no_gap' panicked at src/lib.rs:5:24:\n"
            "index out of bounds: the len is 1 but the index is 1\n"
            "\n"
            "test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out"),
        "files": {
            "src/lib.rs": (
                "//! Largest gap between consecutive values.\n"
                "\n"
                "pub fn max_gap(values: &[i64]) -> i64 {\n"
                "    let mut best = values[1] - values[0];\n"
                "    for pair in values.windows(2) {\n"
                "        let gap = pair[1] - pair[0];\n"
                "        if gap > best {\n"
                "            best = gap;\n"
                "        }\n"
                "    }\n"
                "    best\n"
                "}\n"
                "\n"
                "#[cfg(test)]\n"
                "mod tests {\n"
                "    use super::max_gap;\n"
                "\n"
                "    #[test]\n"
                "    fn single_value_has_no_gap() {\n"
                "        assert_eq!(max_gap(&[5]), 0);\n"
                "    }\n"
                "}\n"),
        },
        "hidden": {
            "tests/check.rs": (
                "use failing_02_empty_input::max_gap;\n"
                "\n"
                "#[test]\n"
                "fn empty_slice() {\n"
                "    assert_eq!(max_gap(&[]), 0);\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn single_value() {\n"
                "    assert_eq!(max_gap(&[5]), 0);\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn largest_gap_wins() {\n"
                "    assert_eq!(max_gap(&[1, 4, 2, 9]), 7);\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn negative_gaps_are_reported() {\n"
                "    assert_eq!(max_gap(&[9, 1]), -8);\n"
                "}\n"),
        },
        "reference": {
            "src/lib.rs": (
                "//! Largest gap between consecutive values.\n"
                "\n"
                "pub fn max_gap(values: &[i64]) -> i64 {\n"
                "    values\n"
                "        .windows(2)\n"
                "        .map(|pair| pair[1] - pair[0])\n"
                "        .max()\n"
                "        .unwrap_or(0)\n"
                "}\n"),
        },
        "mutation": {"file": "src/lib.rs", "find": ".unwrap_or(0)", "replace": ".unwrap_or(1)"},
        "initial_failure": {"kind": "test_failure", "must_contain": ["index out of bounds"]},
    },
    {
        "id": "failing-03-overflow-boundary",
        "category": "failing-test",
        "capability": "Debug a debug-build overflow panic and choose the registered saturation rule.",
        "prompt_intro": (
            "The visible test fails with an arithmetic overflow panic. `running_total` "
            "must sum bytes and saturate at 255 instead of overflowing or wrapping. "
            "Keep the signature `pub fn running_total(values: &[u8]) -> u8`."),
        "observed_failure": (
            "thread 'tests::saturates_instead_of_overflowing' panicked at src/lib.rs:5:28:\n"
            "attempt to add with overflow\n"
            "\n"
            "test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out"),
        "files": {
            "src/lib.rs": (
                "//! Running total over bytes.\n"
                "\n"
                "pub fn running_total(values: &[u8]) -> u8 {\n"
                "    let mut total = 0u8;\n"
                "    for value in values {\n"
                "        total += *value;\n"
                "    }\n"
                "    total\n"
                "}\n"
                "\n"
                "#[cfg(test)]\n"
                "mod tests {\n"
                "    use super::running_total;\n"
                "\n"
                "    #[test]\n"
                "    fn saturates_instead_of_overflowing() {\n"
                "        assert_eq!(running_total(&[250, 10]), 255);\n"
                "    }\n"
                "}\n"),
        },
        "hidden": {
            "tests/check.rs": (
                "use failing_03_overflow_boundary::running_total;\n"
                "\n"
                "#[test]\n"
                "fn saturates_at_the_boundary() {\n"
                "    assert_eq!(running_total(&[250, 10]), 255);\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn saturates_after_the_boundary() {\n"
                "    assert_eq!(running_total(&[255, 255]), 255);\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn sums_below_the_boundary() {\n"
                "    assert_eq!(running_total(&[1, 2, 3]), 6);\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn empty_is_zero() {\n"
                "    assert_eq!(running_total(&[]), 0);\n"
                "}\n"),
        },
        "reference": {
            "src/lib.rs": (
                "//! Running total over bytes.\n"
                "\n"
                "pub fn running_total(values: &[u8]) -> u8 {\n"
                "    values\n"
                "        .iter()\n"
                "        .fold(0u8, |total, value| total.saturating_add(*value))\n"
                "}\n"),
        },
        "mutation": {"file": "src/lib.rs", "find": "saturating_add", "replace": "wrapping_add"},
        "initial_failure": {"kind": "test_failure",
                            "must_contain": ["attempt to add with overflow"]},
    },
    {
        "id": "failing-04-nan-ordering",
        "category": "failing-test",
        "capability": "Replace a partial comparator with a total order that puts NaN last.",
        "prompt_intro": (
            "The visible test fails because `f64::partial_cmp` returns `None` for "
            "NaN and the comparator treats that as equal. Define a total order: "
            "finite values ascending, and every NaN after every finite value. "
            "Keep the signature `pub fn sort_ascending(values: &mut [f64])`."),
        "observed_failure": (
            "thread 'tests::nan_sorts_last' panicked at src/lib.rs:15:9:\n"
            "assertion `left == right` failed\n"
            "  left: [1.0, NaN, 2.0]\n"
            " right: [1.0, 2.0, 3.0]\n"
            "\n"
            "test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out"),
        "files": {
            "src/lib.rs": (
                "//! Sort floats ascending with NaN ordered last.\n"
                "\n"
                "pub fn sort_ascending(values: &mut [f64]) {\n"
                "    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));\n"
                "}\n"
                "\n"
                "#[cfg(test)]\n"
                "mod tests {\n"
                "    use super::sort_ascending;\n"
                "\n"
                "    #[test]\n"
                "    fn nan_sorts_last() {\n"
                "        let mut values = [3.0, f64::NAN, 1.0, 2.0];\n"
                "        sort_ascending(&mut values);\n"
                "        assert_eq!(&values[..3], &[1.0, 2.0, 3.0]);\n"
                "        assert!(values[3].is_nan());\n"
                "    }\n"
                "}\n"),
        },
        "hidden": {
            "tests/check.rs": (
                "use failing_04_nan_ordering::sort_ascending;\n"
                "\n"
                "#[test]\n"
                "fn sorts_finite_values() {\n"
                "    let mut values = [3.0, 1.0, 2.0];\n"
                "    sort_ascending(&mut values);\n"
                "    assert_eq!(values, [1.0, 2.0, 3.0]);\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn nan_sorts_after_finite_values() {\n"
                "    let mut values = [3.0, f64::NAN, 1.0, 2.0];\n"
                "    sort_ascending(&mut values);\n"
                "    assert_eq!(&values[..3], &[1.0, 2.0, 3.0]);\n"
                "    assert!(values[3].is_nan());\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn many_nans_stay_last() {\n"
                "    let mut values = [f64::NAN, 2.0, f64::NAN, 1.0];\n"
                "    sort_ascending(&mut values);\n"
                "    assert_eq!(&values[..2], &[1.0, 2.0]);\n"
                "    assert!(values[2].is_nan() && values[3].is_nan());\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn all_nan_is_stable_in_content() {\n"
                "    let mut values = [f64::NAN, f64::NAN];\n"
                "    sort_ascending(&mut values);\n"
                "    assert!(values.iter().all(|value| value.is_nan()));\n"
                "}\n"),
        },
        "reference": {
            "src/lib.rs": (
                "//! Sort floats ascending with NaN ordered last.\n"
                "\n"
                "pub fn sort_ascending(values: &mut [f64]) {\n"
                "    values.sort_by(|a, b| match (a.is_nan(), b.is_nan()) {\n"
                "        (true, true) => std::cmp::Ordering::Equal,\n"
                "        (true, false) => std::cmp::Ordering::Greater,\n"
                "        (false, true) => std::cmp::Ordering::Less,\n"
                "        (false, false) => a.partial_cmp(b).expect(\"finite values compare\"),\n"
                "    });\n"
                "}\n"),
        },
        "mutation": {"file": "src/lib.rs", "find": "(true, false) => std::cmp::Ordering::Greater",
                     "replace": "(true, false) => std::cmp::Ordering::Less"},
        "initial_failure": {"kind": "test_failure", "must_contain": ["test result: FAILED"]},
    },
]
