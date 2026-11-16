"""Expansion batch three: the scored suite grows from 72 to 144 items (P14.4).

Nine further tasks per capability category, authored under the same discipline
as batches one and two: a failing state with a real diagnostic, hidden tests the
arm never sees, one reference that must pass, one declared mutation the tests
must reject, and (for one task per category) a declared *naive* fix that
compiles but fails a hidden behavioural test.

The diagnostic of every item in this module was produced by compiling the
initial state before it was written down; the codes below are observed, not
guessed. Distinctness is measured, not asserted: `scripts/capability_similarity.py`
screens diagnostics, code structure and test assertions and refuses a pair above
the documented threshold.
"""

from .common import code

# ---------------------------------------------------------------------------
# lifetime-borrow
# ---------------------------------------------------------------------------

LIFETIME = [
    {
        "id": "borrow-10-assign-while-borrowed",
        "category": "lifetime-borrow",
        "difficulty": "easy",
        "capability": "Split a read of a value from the write that follows it.",
        "prompt_intro": (
            "This crate does not compile. `compare_with_last` keeps a reference to "
            "`current` and then assigns to `current` while the reference is still "
            "live. Remove the overlapping borrow: compare the previous value with "
            "the last reading, or with `current` itself when the slice is empty. "
            "Do not change the signature."),
        "observed_failure": (
            "error[E0506]: cannot assign to `current` because it is borrowed\n"
            " --> src/lib.rs:4:5\n"
            "  |\n"
            "3 |     let previous = &current;\n"
            "  |                    -------- borrow of `current` occurs here\n"
            "4 |     current = readings.last().copied().unwrap_or(current);\n"
            "  |     ^^^^^^^ assignment to borrowed `current` occurs here"),
        "files": {
            "src/lib.rs": code("""
                //! Compare the previous value with the last reading. The crate does
                //! not build: `current` is assigned while a reference to it is alive.

                pub fn compare_with_last(current: i64, readings: &[i64]) -> i64 {
                    let mut current = current;
                    let previous = &current;
                    current = readings.last().copied().unwrap_or(current);
                    *previous + current
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use borrow_10_assign_while_borrowed::compare_with_last;

                #[test]
                fn adds_the_previous_and_the_last_reading() {
                    assert_eq!(compare_with_last(5, &[7, 9]), 14);
                }

                #[test]
                fn an_empty_slice_compares_with_itself() {
                    assert_eq!(compare_with_last(5, &[]), 10);
                }

                #[test]
                fn negative_values_are_preserved() {
                    assert_eq!(compare_with_last(-3, &[-8]), -11);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Compare the previous value with the last reading.

                pub fn compare_with_last(current: i64, readings: &[i64]) -> i64 {
                    let previous = current;
                    let last = readings.last().copied().unwrap_or(current);
                    previous + last
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "unwrap_or(current)",
                     "replace": "unwrap_or(0)"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0506"]},
    },
    {
        "id": "borrow-11-move-out-of-borrowed",
        "category": "lifetime-borrow",
        "difficulty": "easy",
        "capability": "Copy out of a shared reference instead of moving the referent.",
        "prompt_intro": (
            "This crate does not compile. `sorted_copy` moves the `Vec` out of a "
            "shared reference. Return a sorted copy and leave the caller's list "
            "usable: the caller must still see every element after the call. Keep "
            "the signature."),
        "observed_failure": (
            "error[E0507]: cannot move out of `*values` which is behind a shared reference\n"
            " --> src/lib.rs:5:21\n"
            "  |\n"
            "5 |     let mut owned = *values;\n"
            "  |                     ^^^^^^^ move occurs because `*values` has type "
            "`Vec<String>`, which does not implement the `Copy` trait"),
        "files": {
            "src/lib.rs": code("""
                //! A sorted copy of a borrowed list. The crate does not build: it
                //! moves the vector out of a shared reference.

                pub fn sorted_copy(values: &Vec<String>) -> Vec<String> {
                    let mut owned = *values;
                    owned.sort();
                    owned
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use borrow_11_move_out_of_borrowed::sorted_copy;

                #[test]
                fn returns_the_elements_in_order() {
                    let values = vec!["beta".to_string(), "alpha".to_string()];
                    assert_eq!(sorted_copy(&values), vec!["alpha".to_string(),
                                                         "beta".to_string()]);
                }

                #[test]
                fn the_caller_keeps_every_element() {
                    let values = vec!["b".to_string(), "a".to_string()];
                    let _copy = sorted_copy(&values);
                    assert_eq!(values, vec!["b".to_string(), "a".to_string()]);
                }

                #[test]
                fn duplicates_are_kept() {
                    let values = vec!["x".to_string(), "x".to_string()];
                    assert_eq!(sorted_copy(&values).len(), 2);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! A sorted copy of a borrowed list.

                pub fn sorted_copy(values: &Vec<String>) -> Vec<String> {
                    let mut owned = values.clone();
                    owned.sort();
                    owned
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "owned.sort();",
                     "replace": "owned.sort_by(|a, b| b.cmp(a));"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0507"]},
    },
    {
        "id": "borrow-12-move-out-of-drop-type",
        "category": "lifetime-borrow",
        "difficulty": "medium",
        "capability": "Hand a field out of a type that implements Drop without moving it.",
        "prompt_intro": (
            "This crate does not compile. `Buffer` implements `Drop`, so its `data` "
            "field cannot be moved out of it. Return the buffered bytes by taking "
            "them out of the value instead, leaving the destructor with an empty "
            "buffer. Keep the signature."),
        "observed_failure": (
            "error[E0509]: cannot move out of type `Buffer`, which implements the "
            "`Drop` trait\n --> src/lib.rs:11:5\n  |\n"
            "11 |     buffer.data\n   |     ^^^^^^^^^^^ cannot move out of here"),
        "files": {
            "src/lib.rs": code("""
                //! Buffer hand-off. The crate does not build: `Buffer` implements
                //! `Drop`, so its field cannot be moved out.

                pub struct Buffer {
                    pub data: Vec<u8>,
                }

                impl Buffer {
                    pub fn new(data: Vec<u8>) -> Self {
                        Self { data }
                    }
                }

                impl Drop for Buffer {
                    fn drop(&mut self) {
                        self.data.clear();
                    }
                }

                pub fn into_data(buffer: Buffer) -> Vec<u8> {
                    buffer.data
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use borrow_12_move_out_of_drop_type::{into_data, Buffer};

                #[test]
                fn returns_the_buffered_bytes() {
                    assert_eq!(into_data(Buffer::new(vec![1, 2, 3])), vec![1, 2, 3]);
                }

                #[test]
                fn an_empty_buffer_returns_no_bytes() {
                    assert!(into_data(Buffer::new(Vec::new())).is_empty());
                }

                #[test]
                fn repeated_hand_offs_are_independent() {
                    let first = into_data(Buffer::new(vec![9]));
                    let second = into_data(Buffer::new(vec![4, 5]));
                    assert_eq!(first, vec![9]);
                    assert_eq!(second, vec![4, 5]);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Buffer hand-off.

                pub struct Buffer {
                    pub data: Vec<u8>,
                }

                impl Buffer {
                    pub fn new(data: Vec<u8>) -> Self {
                        Self { data }
                    }
                }

                impl Drop for Buffer {
                    fn drop(&mut self) {
                        self.data.clear();
                    }
                }

                pub fn into_data(mut buffer: Buffer) -> Vec<u8> {
                    std::mem::take(&mut buffer.data)
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "std::mem::take(&mut buffer.data)",
                     "replace": "Vec::new()"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0509"]},
    },
    {
        "id": "borrow-13-mutate-through-shared",
        "category": "lifetime-borrow",
        "difficulty": "easy",
        "capability": "Take a mutable reference when a function mutates its argument.",
        "prompt_intro": (
            "This crate does not compile. `stamp` increments every entry, but it "
            "receives `&Vec<u32>` and cannot write through a shared reference. "
            "Change the parameter to a mutable reference and return the sum of the "
            "incremented entries. The caller's vector must hold the new values "
            "afterwards."),
        "observed_failure": (
            "error[E0596]: cannot borrow `*entries` as mutable, as it is behind a "
            "`&` reference\n --> src/lib.rs:5:37\n  |\n"
            "5 |     for index in 0..entries.len() { entries[index] += 1; }\n"
            "  |                                     ^^^^^^^^^^^^^^^ cannot borrow as mutable"),
        "files": {
            "src/lib.rs": code("""
                //! Increment every entry in place. The crate does not build: the
                //! vector is borrowed immutably but mutated.

                pub fn stamp(entries: &Vec<u32>) -> u32 {
                    for index in 0..entries.len() {
                        entries[index] += 1;
                    }
                    entries.iter().sum()
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use borrow_13_mutate_through_shared::stamp;

                #[test]
                fn increments_every_entry_and_sums() {
                    let mut entries = vec![1, 2];
                    assert_eq!(stamp(&mut entries), 5);
                    assert_eq!(entries, vec![2, 3]);
                }

                #[test]
                fn an_empty_vector_sums_to_zero() {
                    let mut entries: Vec<u32> = Vec::new();
                    assert_eq!(stamp(&mut entries), 0);
                    assert!(entries.is_empty());
                }

                #[test]
                fn a_single_entry_is_incremented() {
                    let mut entries = vec![41];
                    assert_eq!(stamp(&mut entries), 42);
                    assert_eq!(entries, vec![42]);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Increment every entry in place.

                pub fn stamp(entries: &mut Vec<u32>) -> u32 {
                    for index in 0..entries.len() {
                        entries[index] += 1;
                    }
                    entries.iter().sum()
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "entries[index] += 1;",
                     "replace": "entries[index] += 2;"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0596"]},
    },
    {
        "id": "borrow-14-borrow-after-move",
        "category": "lifetime-borrow",
        "difficulty": "easy",
        "capability": "Compute from a value before consuming it.",
        "prompt_intro": (
            "This crate does not compile. `join_then_measure` consumes `parts` and "
            "then asks for its length. Compute the length before the consumption. "
            "The result is the joined text, a colon, the first part (empty when "
            "there is none), and the number of parts."),
        "observed_failure": (
            "error[E0382]: borrow of moved value: `parts`\n --> src/lib.rs:5:17\n  |\n"
            "3 |     let first = parts.into_iter().next().unwrap_or_default();\n"
            "  |                 ----- value moved here\n"
            "5 |     let count = parts.len();\n"
            "  |                 ^^^^^ value borrowed here after move"),
        "files": {
            "src/lib.rs": code("""
                //! Join a list and measure it. The crate does not build: the list is
                //! used after it is moved.

                pub fn join_then_measure(parts: Vec<String>) -> (String, usize) {
                    let joined = parts.join("-");
                    let first = parts.into_iter().next().unwrap_or_default();
                    let count = parts.len();
                    (format!("{joined}:{first}"), count)
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use borrow_14_borrow_after_move::join_then_measure;

                #[test]
                fn joins_and_counts() {
                    let parts = vec!["a".to_string(), "b".to_string()];
                    assert_eq!(join_then_measure(parts),
                               ("a-b:a".to_string(), 2));
                }

                #[test]
                fn an_empty_list_has_an_empty_first_part() {
                    assert_eq!(join_then_measure(Vec::new()), (":".to_string(), 0));
                }

                #[test]
                fn a_single_part_is_joined_with_itself() {
                    let parts = vec!["x".to_string()];
                    assert_eq!(join_then_measure(parts), ("x:x".to_string(), 1));
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Join a list and measure it.

                pub fn join_then_measure(parts: Vec<String>) -> (String, usize) {
                    let joined = parts.join("-");
                    let count = parts.len();
                    let first = parts.into_iter().next().unwrap_or_default();
                    (format!("{joined}:{first}"), count)
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": 'parts.join("-")',
                     "replace": 'parts.join("+")'},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0382"]},
    },
    {
        "id": "borrow-15-return-local-table-entry",
        "category": "lifetime-borrow",
        "difficulty": "medium",
        "capability": "Return owned data instead of a reference into a local table.",
        "prompt_intro": (
            "This crate does not compile. `seed` builds a local `BTreeMap` and "
            "returns a reference into it. Change the return type to `String` and "
            "return the seeded value, or the empty string when the key is absent."),
        "observed_failure": (
            "error[E0515]: cannot return value referencing local variable `table`\n"
            " --> src/lib.rs:6:5\n  |\n"
            "4 |     table.insert(key.to_string(), format!(\"value-{key}\"));\n"
            "  |     ----- `table` is borrowed here\n"
            "6 |     table.get(key).map(|v| v.as_str()).unwrap_or(\"\")\n"
            "  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ returns a value "
            "referencing data owned by the current function"),
        "files": {
            "src/lib.rs": code("""
                //! Seed a table and read it back. The crate does not build: the
                //! result borrows a table created inside the function.

                use std::collections::BTreeMap;

                pub fn seed<'a>(key: &str) -> &'a str {
                    let mut table = BTreeMap::new();
                    table.insert(key.to_string(), format!("value-{key}"));
                    table.get(key).map(|value| value.as_str()).unwrap_or("")
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use borrow_15_return_local_table_entry::seed;

                #[test]
                fn returns_the_seeded_value() {
                    assert_eq!(seed("k"), "value-k".to_string());
                }

                #[test]
                fn successive_calls_are_independent() {
                    let first = seed("one");
                    let second = seed("two");
                    assert_eq!(first, "value-one");
                    assert_eq!(second, "value-two");
                }

                #[test]
                fn a_long_key_is_included_verbatim() {
                    assert_eq!(seed("alpha-beta"), "value-alpha-beta".to_string());
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Seed a table and read it back.

                use std::collections::BTreeMap;

                pub fn seed(key: &str) -> String {
                    let mut table = BTreeMap::new();
                    table.insert(key.to_string(), format!("value-{key}"));
                    table.get(key).cloned().unwrap_or_default()
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": 'format!("value-{key}")',
                     "replace": 'format!("value")'},
        "naive": {"file": "src/lib.rs", "find": "table.get(key).cloned().unwrap_or_default()",
                  "replace": "key.to_string()"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0515"]},
    },
    {
        "id": "borrow-16-closure-borrows-local",
        "category": "lifetime-borrow",
        "difficulty": "medium",
        "capability": "Move captured state into a closure that outlives the call.",
        "prompt_intro": (
            "This crate does not compile. `make_counter` returns a closure that "
            "borrows the local counter, so the closure cannot outlive the call. "
            "Move the captured state into the closure. Each call returns the next "
            "value, starting at `seed + 1`."),
        "observed_failure": (
            "error[E0373]: closure may outlive the current function, but it borrows "
            "`value`, which is owned by the current function\n --> src/lib.rs:5:5\n"
            "  |\n5 |     || {\n  |     ^^ may outlive borrowed value `value`\n"
            "6 |         value += 1;\n  |         ----- `value` is borrowed here\n"
            "  |\nhelp: to force the closure to take ownership of `value` (and any "
            "other referenced variables), use the `move` keyword"),
        "files": {
            "src/lib.rs": code("""
                //! A counter closure. The crate does not build: the returned closure
                //! borrows a local binding.

                pub fn make_counter(seed: u64) -> impl FnMut() -> u64 {
                    let mut value = seed;
                    || {
                        value += 1;
                        value
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use borrow_16_closure_borrows_local::make_counter;

                #[test]
                fn the_first_call_returns_seed_plus_one() {
                    let mut counter = make_counter(3);
                    assert_eq!(counter(), 4);
                    assert_eq!(counter(), 5);
                }

                #[test]
                fn counters_are_independent() {
                    let mut first = make_counter(10);
                    let mut second = make_counter(0);
                    assert_eq!(first(), 11);
                    assert_eq!(second(), 1);
                    assert_eq!(first(), 12);
                }

                #[test]
                fn the_counter_starts_above_the_seed_only() {
                    let mut counter = make_counter(41);
                    assert_eq!(counter(), 42);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! A counter closure.

                pub fn make_counter(seed: u64) -> impl FnMut() -> u64 {
                    let mut value = seed;
                    move || {
                        value += 1;
                        value
                    }
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "value += 1;",
                     "replace": "value += 2;"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0373"]},
    },
    {
        "id": "borrow-17-two-input-lifetimes",
        "category": "lifetime-borrow",
        "difficulty": "medium",
        "capability": "Unify two input lifetimes when the result may borrow from either.",
        "prompt_intro": (
            "This crate does not compile. `longer` declares two independent input "
            "lifetimes but can return either input, so the result cannot be tied to "
            "one of them. Give both inputs the same lifetime. Return the longer "
            "text, and the first argument when the lengths are equal."),
        "observed_failure": (
            "error: lifetime may not live long enough\n --> src/lib.rs:2:35\n  |\n"
            "1 | pub fn longer<'a, 'b>(left: &'a str, right: &'b str) -> &'a str {\n"
            "  |               --        - let's call the lifetime of this reference `'1`\n"
            "2 |     if right.len() > left.len() { right } else { left }\n"
            "  |                                   ^^^^^ returning this value requires "
            "that `'1` must outlive `'a`"),
        "files": {
            "src/lib.rs": code("""
                //! Choose the longer of two texts. The crate does not build: the
                //! result may borrow from the second input.

                pub fn longer<'a, 'b>(left: &'a str, right: &'b str) -> &'a str {
                    if right.len() > left.len() {
                        right
                    } else {
                        left
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use borrow_17_two_input_lifetimes::longer;

                #[test]
                fn returns_the_longer_text() {
                    assert_eq!(longer("ab", "cde"), "cde");
                }

                #[test]
                fn a_tie_returns_the_first_argument() {
                    assert_eq!(longer("ab", "cd"), "ab");
                }

                #[test]
                fn an_empty_first_argument_yields_the_second() {
                    assert_eq!(longer("", "x"), "x");
                }

                #[test]
                fn the_result_can_be_used_after_both_inputs_are_read() {
                    let left = String::from("left");
                    let right = String::from("longer");
                    let chosen = longer(&left, &right);
                    assert_eq!(chosen.len(), 6);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Choose the longer of two texts.

                pub fn longer<'a>(left: &'a str, right: &'a str) -> &'a str {
                    if right.len() > left.len() {
                        right
                    } else {
                        left
                    }
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "right.len() > left.len()",
                     "replace": "right.len() >= left.len()"},
        "initial_failure": {"kind": "compile_error",
                            "must_contain": ["lifetime may not live long enough"]},
    },
    {
        "id": "borrow-18-refcell-double-borrow",
        "category": "lifetime-borrow",
        "difficulty": "medium",
        "capability": "Hold one interior borrow while reading and writing a ledger.",
        "prompt_intro": (
            "The crate builds and its visible test fails at run time: `record` takes "
            "a mutable borrow of the ledger's cell, keeps it, and then takes a "
            "second one. Hold a single borrow and run the ledger forward: each "
            "record appends the running total plus the new value. Keep the "
            "signatures."),
        "observed_failure": (
            "thread 'tests::visible_record_panics' panicked at src/lib.rs:10:37:\n"
            "RefCell already borrowed"),
        "files": {
            "src/lib.rs": code("""
                //! A ledger behind an interior-mutability cell. The visible test
                //! fails at run time: `record` takes two live mutable borrows.

                use std::cell::RefCell;

                pub struct Ledger {
                    entries: RefCell<Vec<i64>>,
                }

                impl Ledger {
                    pub fn new() -> Self {
                        Self { entries: RefCell::new(Vec::new()) }
                    }

                    pub fn record(&self, value: i64) {
                        let entries = self.entries.borrow_mut();
                        let total: i64 = entries.iter().sum();
                        self.entries.borrow_mut().push(total + value);
                    }

                    pub fn entries(&self) -> Vec<i64> {
                        self.entries.borrow().clone()
                    }
                }

                #[cfg(test)]
                mod tests {
                    use super::Ledger;

                    #[test]
                    fn visible_record_panics() {
                        let ledger = Ledger::new();
                        ledger.record(1);
                        assert_eq!(ledger.entries(), vec![1]);
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use borrow_18_refcell_double_borrow::Ledger;

                #[test]
                fn the_first_record_is_its_own_total() {
                    let ledger = Ledger::new();
                    ledger.record(1);
                    assert_eq!(ledger.entries(), vec![1]);
                }

                #[test]
                fn each_record_runs_the_total_forward() {
                    let ledger = Ledger::new();
                    ledger.record(1);
                    ledger.record(2);
                    ledger.record(5);
                    assert_eq!(ledger.entries(), vec![1, 3, 9]);
                }

                #[test]
                fn zero_is_recorded() {
                    let ledger = Ledger::new();
                    ledger.record(0);
                    assert_eq!(ledger.entries(), vec![0]);
                }

                #[test]
                fn reads_do_not_change_the_ledger() {
                    let ledger = Ledger::new();
                    ledger.record(4);
                    assert_eq!(ledger.entries(), vec![4]);
                    assert_eq!(ledger.entries(), vec![4]);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! A ledger behind an interior-mutability cell.

                use std::cell::RefCell;

                pub struct Ledger {
                    entries: RefCell<Vec<i64>>,
                }

                impl Ledger {
                    pub fn new() -> Self {
                        Self { entries: RefCell::new(Vec::new()) }
                    }

                    pub fn record(&self, value: i64) {
                        let mut entries = self.entries.borrow_mut();
                        let total: i64 = entries.iter().sum();
                        entries.push(total + value);
                    }

                    pub fn entries(&self) -> Vec<i64> {
                        self.entries.borrow().clone()
                    }
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "entries.push(total + value);",
                     "replace": "entries.push(value);"},
        "initial_failure": {"kind": "test_failure",
                            "must_contain": ["already borrowed"]},
    },
]

# ---------------------------------------------------------------------------
# failing-test
# ---------------------------------------------------------------------------

FAILING = [
    {
        "id": "failing-10-last-duplicate-index",
        "category": "failing-test",
        "difficulty": "medium",
        "capability": "Find the last index of a duplicated value with a correct bound.",
        "prompt_intro": (
            "The visible test fails. `last_index` returns the first matching "
            "position instead of the last. It must return the highest index whose "
            "value equals the target, or `None`. The input is sorted ascending."),
        "observed_failure": (
            "thread 'tests::visible_last_position' panicked at src/lib.rs:21:9:\n"
            "assertion `left == right` failed\n  left: Some(1)\n right: Some(2)"),
        "files": {
            "src/lib.rs": code("""
                //! Last index of a value. The visible test fails: the search stops at
                //! the first match.

                pub fn last_index(values: &[i32], target: i32) -> Option<usize> {
                    let mut low = 0usize;
                    let mut high = values.len();
                    while low < high {
                        let middle = low + (high - low) / 2;
                        if values[middle] == target {
                            return Some(middle);
                        }
                        if values[middle] < target {
                            low = middle + 1;
                        } else {
                            high = middle;
                        }
                    }
                    None
                }

                #[cfg(test)]
                mod tests {
                    use super::last_index;

                    #[test]
                    fn visible_last_position() {
                        assert_eq!(last_index(&[1, 2, 2], 2), Some(2));
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use failing_10_last_duplicate_index::last_index;

                #[test]
                fn finds_the_last_of_several_duplicates() {
                    assert_eq!(last_index(&[1, 2, 2, 2, 3], 2), Some(3));
                }

                #[test]
                fn finds_a_unique_value() {
                    assert_eq!(last_index(&[1, 3, 5], 5), Some(2));
                }

                #[test]
                fn a_missing_value_is_none() {
                    assert_eq!(last_index(&[1, 2, 3], 4), None);
                }

                #[test]
                fn an_empty_slice_is_none() {
                    assert_eq!(last_index(&[], 0), None);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Last index of a value.

                pub fn last_index(values: &[i32], target: i32) -> Option<usize> {
                    let mut low = 0usize;
                    let mut high = values.len();
                    while low < high {
                        let middle = low + (high - low) / 2;
                        if values[middle] <= target {
                            low = middle + 1;
                        } else {
                            high = middle;
                        }
                    }
                    if low > 0 && values[low - 1] == target {
                        Some(low - 1)
                    } else {
                        None
                    }
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "values[low - 1] == target",
                     "replace": "values[low - 1] != target"},
        "initial_failure": {"kind": "test_failure",
                            "must_contain": ["assertion `left == right` failed"]},
    },
    {
        "id": "failing-11-wrap-counts-the-space",
        "category": "failing-test",
        "difficulty": "medium",
        "capability": "Count the separating space when deciding a greedy line break.",
        "prompt_intro": (
            "The visible test fails. `wrap` places words greedily so that no line "
            "exceeds `width` characters, but it forgets the space it is about to "
            "insert when it measures the candidate line. A word longer than `width` "
            "is kept whole on its own line. Empty input yields no lines."),
        "observed_failure": (
            "thread 'tests::visible_line_breaks' panicked at src/lib.rs:25:9:\n"
            "assertion `left == right` failed\n  left: [\"aaa bbb\"]\n right: [\"aaa\", \"bbb\"]"),
        "files": {
            "src/lib.rs": code("""
                //! Greedy word wrapping. The visible test fails: the space between
                //! words is not counted when a line break is decided.

                pub fn wrap(text: &str, width: usize) -> Vec<String> {
                    let mut lines: Vec<String> = Vec::new();
                    let mut current = String::new();
                    for word in text.split_whitespace() {
                        if current.is_empty() {
                            current.push_str(word);
                        } else if current.len() + word.len() > width {
                            lines.push(std::mem::take(&mut current));
                            current.push_str(word);
                        } else {
                            current.push(' ');
                            current.push_str(word);
                        }
                    }
                    if !current.is_empty() {
                        lines.push(current);
                    }
                    lines
                }

                #[cfg(test)]
                mod tests {
                    use super::wrap;

                    #[test]
                    fn visible_line_breaks() {
                        assert_eq!(wrap("aaa bbb", 6), vec!["aaa", "bbb"]);
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use failing_11_wrap_counts_the_space::wrap;

                #[test]
                fn a_space_counts_against_the_width() {
                    assert_eq!(wrap("aaa bbb", 6), vec!["aaa", "bbb"]);
                }

                #[test]
                fn several_words_fill_one_line() {
                    assert_eq!(wrap("aa bb cc", 5), vec!["aa bb", "cc"]);
                }

                #[test]
                fn empty_input_has_no_lines() {
                    assert!(wrap("", 4).is_empty());
                }

                #[test]
                fn an_overlong_word_stays_whole() {
                    assert_eq!(wrap("abcdef", 3), vec!["abcdef"]);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Greedy word wrapping.

                pub fn wrap(text: &str, width: usize) -> Vec<String> {
                    let mut lines: Vec<String> = Vec::new();
                    let mut current = String::new();
                    for word in text.split_whitespace() {
                        if current.is_empty() {
                            current.push_str(word);
                        } else if current.len() + 1 + word.len() > width {
                            lines.push(std::mem::take(&mut current));
                            current.push_str(word);
                        } else {
                            current.push(' ');
                            current.push_str(word);
                        }
                    }
                    if !current.is_empty() {
                        lines.push(current);
                    }
                    lines
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "current.len() + 1 + word.len() > width",
                     "replace": "current.len() + word.len() > width"},
        "initial_failure": {"kind": "test_failure",
                            "must_contain": ["assertion `left == right` failed"]},
    },
    {
        "id": "failing-12-roman-subtractive",
        "category": "failing-test",
        "difficulty": "medium",
        "capability": "Encode Roman numerals with the subtractive pairs.",
        "prompt_intro": (
            "The visible test fails. `to_roman` encodes 1..=3999 but only uses "
            "additive symbols, so 4 becomes `IIII` instead of `IV`. Use the "
            "subtractive pairs (IV, IX, XL, XC, CD, CM). Values outside 1..=3999 "
            "return the empty string."),
        "observed_failure": (
            "thread 'tests::visible_subtractive_four' panicked at src/lib.rs:22:9:\n"
            "assertion `left == right` failed\n  left: \"IIII\"\n right: \"IV\""),
        "files": {
            "src/lib.rs": code("""
                //! Roman numerals. The visible test fails: there are no subtractive
                //! pairs.

                pub fn to_roman(value: u32) -> String {
                    if value == 0 || value > 3999 {
                        return String::new();
                    }
                    let table = [
                        (1000, "M"), (500, "D"), (100, "C"), (50, "L"),
                        (10, "X"), (5, "V"), (1, "I"),
                    ];
                    let mut remaining = value;
                    let mut out = String::new();
                    for (amount, symbol) in table {
                        while remaining >= amount {
                            out.push_str(symbol);
                            remaining -= amount;
                        }
                    }
                    out
                }

                #[cfg(test)]
                mod tests {
                    use super::to_roman;

                    #[test]
                    fn visible_subtractive_four() {
                        assert_eq!(to_roman(4), "IV");
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use failing_12_roman_subtractive::to_roman;

                #[test]
                fn every_subtractive_pair_is_used() {
                    assert_eq!(to_roman(4), "IV");
                    assert_eq!(to_roman(9), "IX");
                    assert_eq!(to_roman(40), "XL");
                    assert_eq!(to_roman(90), "XC");
                    assert_eq!(to_roman(400), "CD");
                    assert_eq!(to_roman(900), "CM");
                }

                #[test]
                fn a_compound_number_is_exact() {
                    assert_eq!(to_roman(1994), "MCMXCIV");
                }

                #[test]
                fn the_bounds_are_respected() {
                    assert_eq!(to_roman(1), "I");
                    assert_eq!(to_roman(3999), "MMMCMXCIX");
                    assert_eq!(to_roman(0), "");
                    assert_eq!(to_roman(4000), "");
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Roman numerals.

                pub fn to_roman(value: u32) -> String {
                    if value == 0 || value > 3999 {
                        return String::new();
                    }
                    let table = [
                        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
                        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
                        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
                    ];
                    let mut remaining = value;
                    let mut out = String::new();
                    for (amount, symbol) in table {
                        while remaining >= amount {
                            out.push_str(symbol);
                            remaining -= amount;
                        }
                    }
                    out
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": '(900, "CM"), (500, "D"), (400, "CD")',
                     "replace": '(900, "DCCCC"), (500, "D"), (400, "CCCC")'},
        "initial_failure": {"kind": "test_failure",
                            "must_contain": ["assertion `left == right` failed"]},
    },
    {
        "id": "failing-13-duration-minutes-field",
        "category": "failing-test",
        "difficulty": "easy",
        "capability": "Reduce the minute field modulo sixty when formatting a duration.",
        "prompt_intro": (
            "The visible test fails. `format_duration` prints `hh:mm:ss` with two "
            "digits per field, but the minute field is the total number of minutes "
            "rather than the minutes within the hour. Hours are not capped at 24."),
        "observed_failure": (
            "thread 'tests::visible_one_hour' panicked at src/lib.rs:9:9:\n"
            "assertion `left == right` failed\n  left: \"01:61:01\"\n right: \"01:01:01\""),
        "files": {
            "src/lib.rs": code("""
                //! Duration formatting. The visible test fails: the minute field is
                //! the total number of minutes.

                pub fn format_duration(total_seconds: u64) -> String {
                    let hours = total_seconds / 3600;
                    let minutes = total_seconds / 60;
                    let seconds = total_seconds % 60;
                    format!("{hours:02}:{minutes:02}:{seconds:02}")
                }

                #[cfg(test)]
                mod tests {
                    use super::format_duration;

                    #[test]
                    fn visible_one_hour() {
                        assert_eq!(format_duration(3661), "01:01:01");
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use failing_13_duration_minutes_field::format_duration;

                #[test]
                fn zero_is_all_zeroes() {
                    assert_eq!(format_duration(0), "00:00:00");
                }

                #[test]
                fn sub_minute_and_sub_hour_values() {
                    assert_eq!(format_duration(59), "00:00:59");
                    assert_eq!(format_duration(60), "00:01:00");
                }

                #[test]
                fn an_hour_plus_one_second() {
                    assert_eq!(format_duration(3661), "01:01:01");
                }

                #[test]
                fn hours_exceed_a_day() {
                    assert_eq!(format_duration(90061), "25:01:01");
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Duration formatting.

                pub fn format_duration(total_seconds: u64) -> String {
                    let hours = total_seconds / 3600;
                    let minutes = (total_seconds / 60) % 60;
                    let seconds = total_seconds % 60;
                    format!("{hours:02}:{minutes:02}:{seconds:02}")
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "let minutes = (total_seconds / 60) % 60;",
                     "replace": "let minutes = total_seconds / 60;"},
        "initial_failure": {"kind": "test_failure",
                            "must_contain": ["assertion `left == right` failed"]},
    },
    {
        "id": "failing-14-century-leap-year",
        "category": "failing-test",
        "difficulty": "easy",
        "capability": "Apply the century rule of the Gregorian leap year.",
        "prompt_intro": (
            "The visible test fails. `days_in_month` treats every fourth year as a "
            "leap year, so February 1900 has 29 days instead of 28. A year is a "
            "leap year when it is divisible by 4, except centuries that are not "
            "divisible by 400. A month outside 1..=12 has 0 days."),
        "observed_failure": (
            "thread 'tests::visible_nineteen_hundred' panicked at src/lib.rs:15:9:\n"
            "assertion `left == right` failed\n  left: 29\n right: 28"),
        "files": {
            "src/lib.rs": code("""
                //! Days in a month. The visible test fails: the century rule is
                //! missing from the leap-year test.

                pub fn days_in_month(year: u32, month: u32) -> u32 {
                    let leap = year % 4 == 0;
                    match month {
                        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
                        4 | 6 | 9 | 11 => 30,
                        2 if leap => 29,
                        2 => 28,
                        _ => 0,
                    }
                }

                #[cfg(test)]
                mod tests {
                    use super::days_in_month;

                    #[test]
                    fn visible_nineteen_hundred() {
                        assert_eq!(days_in_month(1900, 2), 28);
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use failing_14_century_leap_year::days_in_month;

                #[test]
                fn centuries_that_are_not_leap_years() {
                    assert_eq!(days_in_month(1900, 2), 28);
                    assert_eq!(days_in_month(2100, 2), 28);
                }

                #[test]
                fn the_four_hundred_year_rule() {
                    assert_eq!(days_in_month(2000, 2), 29);
                    assert_eq!(days_in_month(1600, 2), 29);
                }

                #[test]
                fn ordinary_years() {
                    assert_eq!(days_in_month(2023, 2), 28);
                    assert_eq!(days_in_month(2024, 2), 29);
                }

                #[test]
                fn month_lengths_and_invalid_months() {
                    assert_eq!(days_in_month(2023, 4), 30);
                    assert_eq!(days_in_month(2023, 12), 31);
                    assert_eq!(days_in_month(2023, 0), 0);
                    assert_eq!(days_in_month(2023, 13), 0);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Days in a month.

                pub fn days_in_month(year: u32, month: u32) -> u32 {
                    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
                    match month {
                        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
                        4 | 6 | 9 | 11 => 30,
                        2 if leap => 29,
                        2 => 28,
                        _ => 0,
                    }
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "year % 400 == 0",
                     "replace": "year % 500 == 0"},
        "naive": {"file": "src/lib.rs",
                  "find": "let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);",
                  "replace": "let leap = year % 4 == 0 && year % 100 != 0;"},
        "initial_failure": {"kind": "test_failure",
                            "must_contain": ["assertion `left == right` failed"]},
    },
    {
        "id": "failing-15-run-length-count",
        "category": "failing-test",
        "difficulty": "easy",
        "capability": "Reset a run counter to one, not zero, after emitting a run.",
        "prompt_intro": (
            "The visible test fails. `encode` counts every run but restarts the "
            "counter at zero after a run is emitted, so the final run of every "
            "input is one short. A run is its character followed by its decimal "
            "length, always. Empty input yields the empty string."),
        "observed_failure": (
            "thread 'tests::visible_two_as' panicked at src/lib.rs:21:9:\n"
            "assertion `left == right` failed\n  left: \"a2b0\"\n right: \"a2b1\""),
        "files": {
            "src/lib.rs": code("""
                //! Run-length encoding. The visible test fails: the counter restarts
                //! at zero after a run is emitted.

                pub fn encode(text: &str) -> String {
                    let mut out = String::new();
                    let mut characters = text.chars();
                    let mut current = match characters.next() {
                        Some(character) => character,
                        None => return out,
                    };
                    let mut count = 1u32;
                    for character in characters {
                        if character == current {
                            count += 1;
                        } else {
                            out.push(current);
                            out.push_str(&count.to_string());
                            current = character;
                            count = 0;
                        }
                    }
                    out.push(current);
                    out.push_str(&count.to_string());
                    out
                }

                #[cfg(test)]
                mod tests {
                    use super::encode;

                    #[test]
                    fn visible_two_as() {
                        assert_eq!(encode("aab"), "a2b1");
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use failing_15_run_length_count::encode;

                #[test]
                fn every_run_carries_its_length() {
                    assert_eq!(encode("aab"), "a2b1");
                    assert_eq!(encode("abbcccd"), "a1b2c3d1");
                }

                #[test]
                fn a_single_character_is_a_run_of_one() {
                    assert_eq!(encode("a"), "a1");
                }

                #[test]
                fn empty_input_is_empty() {
                    assert_eq!(encode(""), "");
                }

                #[test]
                fn a_long_run_is_counted_exactly() {
                    assert_eq!(encode("aaaa"), "a4");
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Run-length encoding.

                pub fn encode(text: &str) -> String {
                    let mut out = String::new();
                    let mut characters = text.chars();
                    let mut current = match characters.next() {
                        Some(character) => character,
                        None => return out,
                    };
                    let mut count = 1u32;
                    for character in characters {
                        if character == current {
                            count += 1;
                        } else {
                            out.push(current);
                            out.push_str(&count.to_string());
                            current = character;
                            count = 1;
                        }
                    }
                    out.push(current);
                    out.push_str(&count.to_string());
                    out
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "count += 1;",
                     "replace": "count += 2;"},
        "initial_failure": {"kind": "test_failure",
                            "must_contain": ["assertion `left == right` failed"]},
    },
    {
        "id": "failing-16-merge-adjacent",
        "category": "failing-test",
        "difficulty": "medium",
        "capability": "Merge adjacent intervals as well as overlapping ones.",
        "prompt_intro": (
            "The visible test fails. `merge` merges strictly overlapping intervals "
            "but leaves touching intervals apart. Two intervals that share an "
            "endpoint must become one. Input order is arbitrary; the output is "
            "sorted by start."),
        "observed_failure": (
            "thread 'tests::visible_adjacent' panicked at src/lib.rs:23:9:\n"
            "assertion `left == right` failed\n"
            "  left: [(1, 2), (2, 3)]\n right: [(1, 3)]"),
        "files": {
            "src/lib.rs": code("""
                //! Interval merging. The visible test fails: touching intervals are
                //! not merged.

                pub fn merge(ranges: &[(i32, i32)]) -> Vec<(i32, i32)> {
                    let mut sorted = ranges.to_vec();
                    sorted.sort_by_key(|range| range.0);
                    let mut out: Vec<(i32, i32)> = Vec::new();
                    for range in sorted {
                        match out.last_mut() {
                            Some(current) if range.0 < current.1 => {
                                if range.1 > current.1 {
                                    current.1 = range.1;
                                }
                            }
                            _ => out.push(range),
                        }
                    }
                    out
                }

                #[cfg(test)]
                mod tests {
                    use super::merge;

                    #[test]
                    fn visible_adjacent() {
                        assert_eq!(merge(&[(1, 2), (2, 3)]), vec![(1, 3)]);
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use failing_16_merge_adjacent::merge;

                #[test]
                fn touching_intervals_merge() {
                    assert_eq!(merge(&[(1, 2), (2, 3)]), vec![(1, 3)]);
                }

                #[test]
                fn overlapping_intervals_merge() {
                    assert_eq!(merge(&[(1, 5), (2, 3)]), vec![(1, 5)]);
                }

                #[test]
                fn disjoint_intervals_are_sorted_and_kept() {
                    assert_eq!(merge(&[(5, 6), (1, 2)]), vec![(1, 2), (5, 6)]);
                }

                #[test]
                fn empty_input_is_empty() {
                    assert!(merge(&[]).is_empty());
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Interval merging.

                pub fn merge(ranges: &[(i32, i32)]) -> Vec<(i32, i32)> {
                    let mut sorted = ranges.to_vec();
                    sorted.sort_by_key(|range| range.0);
                    let mut out: Vec<(i32, i32)> = Vec::new();
                    for range in sorted {
                        match out.last_mut() {
                            Some(current) if range.0 <= current.1 => {
                                if range.1 > current.1 {
                                    current.1 = range.1;
                                }
                            }
                            _ => out.push(range),
                        }
                    }
                    out
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "range.0 <= current.1",
                     "replace": "range.0 < current.1"},
        "initial_failure": {"kind": "test_failure",
                            "must_contain": ["assertion `left == right` failed"]},
    },
    {
        "id": "failing-17-partial-last-page",
        "category": "failing-test",
        "difficulty": "easy",
        "capability": "Clamp the last page to the number of items.",
        "prompt_intro": (
            "The visible test fails. `page_bounds` returns the half-open range of "
            "a one-based page, but the last page ends past the number of items. "
            "Clamp the end to `total`. A zero page, a zero page size, or a page "
            "past the end yields an empty range."),
        "observed_failure": (
            "thread 'tests::visible_partial_page' panicked at src/lib.rs:17:9:\n"
            "assertion `left == right` failed\n  left: (4, 6)\n right: (4, 5)"),
        "files": {
            "src/lib.rs": code("""
                //! Page bounds. The visible test fails: the last page is not clamped.

                pub fn page_bounds(total: usize, page_size: usize, page: usize) -> (usize, usize) {
                    if page_size == 0 || page == 0 {
                        return (0, 0);
                    }
                    let start = (page - 1) * page_size;
                    if start >= total {
                        return (total, total);
                    }
                    (start, start + page_size)
                }

                #[cfg(test)]
                mod tests {
                    use super::page_bounds;

                    #[test]
                    fn visible_partial_page() {
                        assert_eq!(page_bounds(5, 2, 3), (4, 5));
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use failing_17_partial_last_page::page_bounds;

                #[test]
                fn full_pages() {
                    assert_eq!(page_bounds(5, 2, 1), (0, 2));
                    assert_eq!(page_bounds(5, 2, 2), (2, 4));
                }

                #[test]
                fn the_partial_last_page_is_clamped() {
                    assert_eq!(page_bounds(5, 2, 3), (4, 5));
                }

                #[test]
                fn pages_outside_the_range_are_empty() {
                    assert_eq!(page_bounds(5, 2, 4), (5, 5));
                    assert_eq!(page_bounds(0, 3, 1), (0, 0));
                    assert_eq!(page_bounds(5, 0, 1), (0, 0));
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Page bounds.

                pub fn page_bounds(total: usize, page_size: usize, page: usize) -> (usize, usize) {
                    if page_size == 0 || page == 0 {
                        return (0, 0);
                    }
                    let start = (page - 1) * page_size;
                    if start >= total {
                        return (total, total);
                    }
                    (start, (start + page_size).min(total))
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "(start + page_size).min(total)",
                     "replace": "(start + page_size).max(total)"},
        "initial_failure": {"kind": "test_failure",
                            "must_contain": ["assertion `left == right` failed"]},
    },
    {
        "id": "failing-18-negative-caesar-shift",
        "category": "failing-test",
        "difficulty": "medium",
        "capability": "Rotate letters by a negative amount without underflow.",
        "prompt_intro": (
            "The visible test fails. `shift` rotates ASCII letters and leaves every "
            "other character alone, but a negative amount is converted to `u8` "
            "before the rotation, so shifting backwards produces the wrong letters. "
            "Perform the arithmetic with a signed type and a non-negative "
            "remainder."),
        "observed_failure": (
            "thread 'tests::visible_backwards' panicked at src/lib.rs:18:9:\n"
            "assertion `left == right` failed\n  left: \"vab\"\n right: \"zab\""),
        "files": {
            "src/lib.rs": code("""
                //! Caesar rotation. The visible test fails: a negative amount is
                //! converted to `u8` before the rotation.

                pub fn shift(text: &str, amount: i32) -> String {
                    text.chars()
                        .map(|character| {
                            if character.is_ascii_lowercase() {
                                let offset = (character as u8 - b'a').wrapping_add(amount as u8) % 26;
                                (b'a' + offset) as char
                            } else if character.is_ascii_uppercase() {
                                let offset = (character as u8 - b'A').wrapping_add(amount as u8) % 26;
                                (b'A' + offset) as char
                            } else {
                                character
                            }
                        })
                        .collect()
                }

                #[cfg(test)]
                mod tests {
                    use super::shift;

                    #[test]
                    fn visible_backwards() {
                        assert_eq!(shift("abc", -1), "zab");
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use failing_18_negative_caesar_shift::shift;

                #[test]
                fn shifts_backwards_across_the_alphabet() {
                    assert_eq!(shift("abc", -1), "zab");
                }

                #[test]
                fn preserves_case_and_punctuation() {
                    assert_eq!(shift("Hello, World!", 3), "Khoor, Zruog!");
                }

                #[test]
                fn wraps_forwards_and_backwards() {
                    assert_eq!(shift("xyz", 3), "abc");
                    assert_eq!(shift("ABC", -3), "XYZ");
                }

                #[test]
                fn a_zero_shift_changes_nothing() {
                    assert_eq!(shift("a1!", 0), "a1!");
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Caesar rotation.

                pub fn shift(text: &str, amount: i32) -> String {
                    text.chars()
                        .map(|character| {
                            if character.is_ascii_lowercase() {
                                let offset = (character as i32 - 'a' as i32 + amount)
                                    .rem_euclid(26) as u8;
                                (b'a' + offset) as char
                            } else if character.is_ascii_uppercase() {
                                let offset = (character as i32 - 'A' as i32 + amount)
                                    .rem_euclid(26) as u8;
                                (b'A' + offset) as char
                            } else {
                                character
                            }
                        })
                        .collect()
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": ".rem_euclid(26) as u8;",
                     "replace": ".rem_euclid(25) as u8;"},
        "initial_failure": {"kind": "test_failure",
                            "must_contain": ["assertion `left == right` failed"]},
    },
]

# ---------------------------------------------------------------------------
# api-migration
#
# Each item vendors a two-version dependency in `dep/`; the failing state is the
# caller written against version 1 and the crate is locked to version 2. The
# migration is always a change to the caller, never to the vendored library.
# ---------------------------------------------------------------------------


def _app(name: str, dependency: str) -> str:
    return (
        "[package]\n"
        f'name = "{name}"\n'
        'version = "0.1.0"\n'
        'edition = "2021"\n'
        "\n"
        "[workspace]\n"
        'members = [".", "dep"]\n'
        "\n"
        "[lib]\n"
        'path = "src/lib.rs"\n'
        "\n"
        "[dependencies]\n"
        f"{dependency}\n"
    )


def _dep(name: str, source: str, features: str = "") -> dict:
    manifest = (
        "[package]\n"
        f'name = "{name}"\n'
        'version = "2.0.0"\n'
        'edition = "2021"\n'
        "\n"
        "[lib]\n"
        'path = "src/lib.rs"\n'
    )
    if features:
        manifest += "\n[features]\n" + features
    return {"dep/Cargo.toml": manifest, "dep/src/lib.rs": code(source)}


MIGRATION = [
    {
        "id": "migrate-10-method-renamed",
        "category": "api-migration",
        "difficulty": "easy",
        "capability": "Follow a renamed method across a vendored dependency upgrade.",
        "prompt_intro": (
            "This crate does not build against the vendored `stats` 2.0.0. Version "
            "1 exposed `Tally::sum`; version 2 renamed it to `Tally::total` with the "
            "same behaviour. Adapt the caller. Do not modify `dep/`. Keep "
            "`running_total(&[i64]) -> i64` summing every value."),
        "observed_failure": (
            "error[E0599]: no method named `sum` found for struct `Tally` in the "
            "current scope\n --> src/lib.rs:6:11\n  |\n"
            "6 |     tally.sum()\n  |           ^^^ help: there is a method with a "
            "similar name: `total`"),
        "files": {
            **_dep("stats", """
                //! Tally, version 2. Version 1 exposed `Tally::sum`; version 2 calls
                //! the same method `Tally::total`.

                pub struct Tally {
                    values: Vec<i64>,
                }

                impl Tally {
                    pub fn new() -> Self {
                        Self { values: Vec::new() }
                    }

                    pub fn push(&mut self, value: i64) {
                        self.values.push(value);
                    }

                    pub fn total(&self) -> i64 {
                        self.values.iter().sum()
                    }
                }
            """),
            "src/lib.rs": code("""
                //! Running totals through the tally crate. This does not build: the
                //! method was renamed in version 2.

                pub fn running_total(values: &[i64]) -> i64 {
                    let mut tally = stats::Tally::new();
                    for value in values {
                        tally.push(*value);
                    }
                    tally.sum()
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use migrate_10_method_renamed::running_total;

                #[test]
                fn sums_every_value() {
                    assert_eq!(running_total(&[1, 2, 3]), 6);
                }

                #[test]
                fn an_empty_slice_sums_to_zero() {
                    assert_eq!(running_total(&[]), 0);
                }

                #[test]
                fn negative_values_are_included() {
                    assert_eq!(running_total(&[-4, 4, -2]), -2);
                }
            """),
        },
        "readonly": ["dep/"],
        "reference": {
            "src/lib.rs": code("""
                //! Running totals through the tally crate.

                pub fn running_total(values: &[i64]) -> i64 {
                    let mut tally = stats::Tally::new();
                    for value in values {
                        tally.push(*value);
                    }
                    tally.total()
                }
            """),
        },
        "manifest": _app("migrate_10_method_renamed", 'stats = { path = "dep" }'),
        "mutation": {"file": "src/lib.rs", "find": "tally.total()",
                     "replace": "tally.total() * 2"},
        "naive": {"file": "src/lib.rs", "find": "tally.total()",
                  "replace": "tally.total().max(0)"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0599"]},
    },
    {
        "id": "migrate-11-owned-argument",
        "category": "api-migration",
        "difficulty": "easy",
        "capability": "Convert a borrowed argument where version 2 takes ownership.",
        "prompt_intro": (
            "This crate does not build against the vendored `nameplate` 2.0.0. "
            "Version 2 takes the text by value (`String`) instead of by reference, "
            "so the caller must produce an owned string. Adapt `plate`. Do not "
            "modify `dep/`."),
        "observed_failure": (
            "error[E0308]: mismatched types\n --> src/lib.rs:4:22\n  |\n"
            "4 |     nameplate::render(text)\n"
            "  |              ------ ^^^^ expected `String`, found `&str`"),
        "files": {
            **_dep("nameplate", """
                //! Nameplate, version 2. `render` takes the text by value.

                pub fn render(text: String) -> String {
                    format!("[{text}]")
                }
            """),
            "src/lib.rs": code("""
                //! Render a plate. This does not build: version 2 takes the text by
                //! value.

                pub fn plate(text: &str) -> String {
                    nameplate::render(text)
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use migrate_11_owned_argument::plate;

                #[test]
                fn wraps_the_text_in_brackets() {
                    assert_eq!(plate("x"), "[x]");
                }

                #[test]
                fn empty_text_is_still_wrapped() {
                    assert_eq!(plate(""), "[]");
                }

                #[test]
                fn spaces_are_preserved() {
                    assert_eq!(plate("a b"), "[a b]");
                }
            """),
        },
        "readonly": ["dep/"],
        "reference": {
            "src/lib.rs": code("""
                //! Render a plate.

                pub fn plate(text: &str) -> String {
                    nameplate::render(text.to_string())
                }
            """),
        },
        "manifest": _app("migrate_11_owned_argument", 'nameplate = { path = "dep" }'),
        "mutation": {"file": "src/lib.rs", "find": "text.to_string()",
                     "replace": "text.to_ascii_uppercase()"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0308"]},
    },
    {
        "id": "migrate-12-option-result-change",
        "category": "api-migration",
        "difficulty": "medium",
        "capability": "Handle an Option-returning lookup after an API change.",
        "prompt_intro": (
            "This crate does not build against the vendored `registry` 2.0.0. "
            "Version 1 returned the stored count directly; version 2 returns "
            "`Option<i64>` because a key may be absent. `next_count` must return "
            "the stored count plus one, or `0` when the key is absent. Do not "
            "modify `dep/`."),
        "observed_failure": (
            "error[E0369]: cannot add `{integer}` to `Option<i64>`\n"
            " --> src/lib.rs:4:5\n  |\n"
            "4 |     registry.lookup(key) + 1\n"
            "  |     --------------------- ^ - {integer}\n"
            "  |     Option<i64>"),
        "files": {
            **_dep("registry", """
                //! Registry, version 2. `lookup` returns `Option` because a key may
                //! be absent; version 1 returned the count directly.

                use std::collections::BTreeMap;

                pub struct Registry {
                    items: BTreeMap<String, i64>,
                }

                impl Registry {
                    pub fn new() -> Self {
                        Self { items: BTreeMap::new() }
                    }

                    pub fn insert(&mut self, key: &str, value: i64) {
                        self.items.insert(key.to_string(), value);
                    }

                    pub fn lookup(&self, key: &str) -> Option<i64> {
                        self.items.get(key).copied()
                    }
                }
            """),
            "src/lib.rs": code("""
                //! Counts through the registry crate. This does not build: version 2
                //! returns `Option`.

                pub use registry::Registry;

                pub fn next_count(registry: &Registry, key: &str) -> i64 {
                    registry.lookup(key) + 1
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use migrate_12_option_result_change::{next_count, Registry};

                fn registry() -> Registry {
                    let mut registry = Registry::new();
                    registry.insert("alpha", 4);
                    registry.insert("zero", 0);
                    registry
                }

                #[test]
                fn a_present_key_is_incremented() {
                    assert_eq!(next_count(&registry(), "alpha"), 5);
                }

                #[test]
                fn a_zeroth_value_is_incremented() {
                    assert_eq!(next_count(&registry(), "zero"), 1);
                }

                #[test]
                fn a_missing_key_is_zero() {
                    assert_eq!(next_count(&registry(), "missing"), 0);
                }
            """),
        },
        "readonly": ["dep/"],
        "reference": {
            "src/lib.rs": code("""
                //! Counts through the registry crate.

                pub use registry::Registry;

                pub fn next_count(registry: &Registry, key: &str) -> i64 {
                    registry.lookup(key).map(|count| count + 1).unwrap_or(0)
                }
            """),
        },
        "manifest": _app("migrate_12_option_result_change", 'registry = { path = "dep" }'),
        "mutation": {"file": "src/lib.rs", "find": "count + 1",
                     "replace": "count"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0369"]},
    },
    {
        "id": "migrate-13-const-generic",
        "category": "api-migration",
        "difficulty": "medium",
        "capability": "Supply a const generic argument that version 2 made mandatory.",
        "prompt_intro": (
            "This crate does not build against the vendored `frame` 2.0.0. Version 2 "
            "made the frame length a const generic (`Frame<const N: usize>`), so "
            "every use must name the length. `make_counter` must return a frame of "
            "four cells. Do not modify `dep/`."),
        "observed_failure": (
            "error[E0107]: missing generics for struct `Frame`\n --> src/lib.rs:4:37\n"
            "  |\n4 | pub fn make_counter() -> Frame { Frame::new() }\n"
            "  |                                 ^^^^^ expected 1 generic argument"),
        "files": {
            **_dep("frame", """
                //! Frame, version 2. The length became a const generic parameter.

                pub struct Frame<const N: usize> {
                    pub cells: [u8; N],
                }

                impl<const N: usize> Frame<N> {
                    pub fn new() -> Self {
                        Self { cells: [0; N] }
                    }

                    pub fn set(&mut self, index: usize, value: u8) {
                        if index < N {
                            self.cells[index] = value;
                        }
                    }

                    pub fn used(&self) -> usize {
                        self.cells.iter().filter(|cell| **cell != 0).count()
                    }
                }
            """),
            "src/lib.rs": code("""
                //! A frame counter. This does not build: version 2 requires a const
                //! generic length.

                pub use frame::Frame;

                pub fn make_counter() -> Frame {
                    Frame::new()
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use migrate_13_const_generic::make_counter;

                #[test]
                fn a_fresh_frame_has_no_used_cells() {
                    assert_eq!(make_counter().used(), 0);
                }

                #[test]
                fn a_value_can_be_stored() {
                    let mut frame = make_counter();
                    frame.set(1, 7);
                    assert_eq!(frame.used(), 1);
                }

                #[test]
                fn the_frame_has_four_cells() {
                    let mut frame = make_counter();
                    frame.set(3, 1);
                    frame.set(4, 1);
                    assert_eq!(frame.used(), 1);
                }

                #[test]
                fn frames_are_independent() {
                    let mut first = make_counter();
                    let second = make_counter();
                    first.set(0, 9);
                    assert_eq!(first.used(), 1);
                    assert_eq!(second.used(), 0);
                }
            """),
        },
        "readonly": ["dep/"],
        "reference": {
            "src/lib.rs": code("""
                //! A frame counter.

                pub use frame::Frame;

                pub fn make_counter() -> Frame<4> {
                    Frame::new()
                }
            """),
        },
        "manifest": _app("migrate_13_const_generic", 'frame = { path = "dep" }'),
        "mutation": {"file": "src/lib.rs", "find": "Frame<4>",
                     "replace": "Frame<3>"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0107"]},
    },
    {
        "id": "migrate-14-new-enum-variant",
        "category": "api-migration",
        "difficulty": "medium",
        "capability": "Cover a variant added to a vendored enum.",
        "prompt_intro": (
            "This crate does not build against the vendored `status` 2.0.0. Version 2 "
            "added `State::Draft`, and the caller's `match` is no longer exhaustive. "
            "Map `Draft` to `0` and keep `Live` at `1`, `Paused` at `2`. Do not "
            "modify `dep/`."),
        "observed_failure": (
            "error[E0004]: non-exhaustive patterns: `State::Draft` not covered\n"
            " --> src/lib.rs:4:11\n  |\n"
            "4 |     match state {\n  |           ^^^^^ pattern `State::Draft` not covered"),
        "files": {
            **_dep("status", """
                //! Status, version 2. Version 2 added the `Draft` state.

                #[derive(Clone, Copy, Debug, PartialEq)]
                pub enum State {
                    Live,
                    Paused,
                    Draft,
                }
            """),
            "src/lib.rs": code("""
                //! Numeric state codes. This does not build: version 2 added a
                //! variant the match does not cover.

                pub use status::State;

                pub fn code(state: State) -> u8 {
                    match state {
                        State::Live => 1,
                        State::Paused => 2,
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use migrate_14_new_enum_variant::{code, State};

                #[test]
                fn live_is_one() {
                    assert_eq!(code(State::Live), 1);
                }

                #[test]
                fn paused_is_two() {
                    assert_eq!(code(State::Paused), 2);
                }

                #[test]
                fn draft_is_zero() {
                    assert_eq!(code(State::Draft), 0);
                }
            """),
        },
        "readonly": ["dep/"],
        "reference": {
            "src/lib.rs": code("""
                //! Numeric state codes.

                pub use status::State;

                pub fn code(state: State) -> u8 {
                    match state {
                        State::Live => 1,
                        State::Paused => 2,
                        State::Draft => 0,
                    }
                }
            """),
        },
        "manifest": _app("migrate_14_new_enum_variant", 'status = { path = "dep" }'),
        "mutation": {"file": "src/lib.rs", "find": "State::Live => 1",
                     "replace": "State::Live => 9"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0004"]},
    },
    {
        "id": "migrate-15-function-renamed",
        "category": "api-migration",
        "difficulty": "easy",
        "capability": "Call a top-level function that version 2 renamed.",
        "prompt_intro": (
            "This crate does not build against the vendored `units` 2.0.0. Version 2 "
            "renamed the free function `parse` to `parse_value`; the behaviour is "
            "unchanged. Adapt the caller. `double` returns twice the parsed value, "
            "or `None` when the text is not a number. Do not modify `dep/`."),
        "observed_failure": (
            "error[E0425]: cannot find function `parse` in crate `units`\n"
            " --> src/lib.rs:4:12\n  |\n"
            "4 |     units::parse(input)\n"
            "  |            ^^^^^ not found in `units`\n  |\n"
            "help: a function with a similar name exists: `parse_value`"),
        "files": {
            **_dep("units", """
                //! Units, version 2. `parse` is now `parse_value`.

                pub fn parse_value(input: &str) -> Option<f64> {
                    input.trim().parse().ok()
                }
            """),
            "src/lib.rs": code("""
                //! Doubling through the units crate. This does not build: the
                //! function was renamed in version 2.

                pub fn double(input: &str) -> Option<f64> {
                    units::parse(input).map(|value| value * 2.0)
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use migrate_15_function_renamed::double;

                #[test]
                fn doubles_a_number() {
                    assert_eq!(double("2.5"), Some(5.0));
                }

                #[test]
                fn ignores_surrounding_space() {
                    assert_eq!(double(" 3 "), Some(6.0));
                }

                #[test]
                fn rejects_text() {
                    assert_eq!(double("x"), None);
                }
            """),
        },
        "readonly": ["dep/"],
        "reference": {
            "src/lib.rs": code("""
                //! Doubling through the units crate.

                pub fn double(input: &str) -> Option<f64> {
                    units::parse_value(input).map(|value| value * 2.0)
                }
            """),
        },
        "manifest": _app("migrate_15_function_renamed", 'units = { path = "dep" }'),
        "mutation": {"file": "src/lib.rs", "find": "value * 2.0",
                     "replace": "value + 2.0"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0425"]},
    },
    {
        "id": "migrate-16-required-config-argument",
        "category": "api-migration",
        "difficulty": "easy",
        "capability": "Pass a new required argument that version 2 introduced.",
        "prompt_intro": (
            "This crate does not build against the vendored `label` 2.0.0. Version 2 "
            "requires an explicit width argument. `render_label` must render at "
            "width 10: the text, padded with spaces on the right to at least ten "
            "characters, and never truncated. Do not modify `dep/`."),
        "observed_failure": (
            "error[E0061]: this function takes 2 arguments but 1 argument was supplied\n"
            " --> src/lib.rs:4:5\n  |\n"
            "4 |     label::render(text)\n"
            "  |     ^^^^^^^^^^^^^ ----- argument #1 of type `&str` is supplied\n"
            "  |\n  = note: expected parameter `&str`, `usize`"),
        "files": {
            **_dep("label", """
                //! Label, version 2. `render` requires an explicit width.

                pub fn render(text: &str, width: usize) -> String {
                    let mut out = text.to_string();
                    while out.len() < width {
                        out.push(' ');
                    }
                    out
                }
            """),
            "src/lib.rs": code("""
                //! Render a label. This does not build: version 2 requires a width.

                pub fn render_label(text: &str) -> String {
                    label::render(text)
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use migrate_16_required_config_argument::render_label;

                #[test]
                fn short_text_is_padded_to_ten() {
                    assert_eq!(render_label("ab"), "ab        ");
                    assert_eq!(render_label("ab").len(), 10);
                }

                #[test]
                fn long_text_is_not_truncated() {
                    assert_eq!(render_label("abcdefghijk"), "abcdefghijk");
                }

                #[test]
                fn empty_text_becomes_spaces() {
                    assert_eq!(render_label(""), "          ");
                }
            """),
        },
        "readonly": ["dep/"],
        "reference": {
            "src/lib.rs": code("""
                //! Render a label.

                pub fn render_label(text: &str) -> String {
                    label::render(text, 10)
                }
            """),
        },
        "manifest": _app("migrate_16_required_config_argument", 'label = { path = "dep" }'),
        "mutation": {"file": "src/lib.rs", "find": "label::render(text, 10)",
                     "replace": "label::render(text, 8)"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0061"]},
    },
    {
        "id": "migrate-17-field-renamed",
        "category": "api-migration",
        "difficulty": "easy",
        "capability": "Read a renamed struct field after a dependency upgrade.",
        "prompt_intro": (
            "This crate does not build against the vendored `measure` 2.0.0. Version "
            "2 renamed the struct field `value` to `amount`. `doubled` returns twice "
            "the parsed amount, or `None`. Do not modify `dep/`."),
        "observed_failure": (
            "error[E0609]: no field `value` on type `Measure`\n --> src/lib.rs:4:39\n"
            "  |\n4 |     measure::load(text).map(|measure| measure.value * 2.0)\n"
            "  |                                               ^^^^^ unknown field\n"
            "  |\n  = help: a field with a similar name exists: `amount`"),
        "files": {
            **_dep("measure", """
                //! Measure, version 2. The field `value` is now `amount`.

                pub struct Measure {
                    pub amount: f64,
                }

                pub fn load(text: &str) -> Option<Measure> {
                    text.trim().parse().ok().map(|amount| Measure { amount })
                }
            """),
            "src/lib.rs": code("""
                //! Doubling a measurement. This does not build: version 2 renamed the
                //! field.

                pub fn doubled(text: &str) -> Option<f64> {
                    measure::load(text).map(|measure| measure.value * 2.0)
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use migrate_17_field_renamed::doubled;

                #[test]
                fn doubles_the_amount() {
                    assert_eq!(doubled("2.5"), Some(5.0));
                }

                #[test]
                fn zero_doubles_to_zero() {
                    assert_eq!(doubled("0"), Some(0.0));
                }

                #[test]
                fn text_is_rejected() {
                    assert_eq!(doubled("nope"), None);
                }
            """),
        },
        "readonly": ["dep/"],
        "reference": {
            "src/lib.rs": code("""
                //! Doubling a measurement.

                pub fn doubled(text: &str) -> Option<f64> {
                    measure::load(text).map(|measure| measure.amount * 2.0)
                }
            """),
        },
        "manifest": _app("migrate_17_field_renamed", 'measure = { path = "dep" }'),
        "mutation": {"file": "src/lib.rs", "find": "measure.amount * 2.0",
                     "replace": "measure.amount"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0609"]},
    },
    {
        "id": "migrate-18-conversion-bound",
        "category": "api-migration",
        "difficulty": "medium",
        "capability": "Satisfy a new generic conversion bound introduced in version 2.",
        "prompt_intro": (
            "This crate does not build against the vendored `scale` 2.0.0. Version 2 "
            "made the input generic over `Into<f64>`, which `u64` does not "
            "implement. Convert the value explicitly. `scale_units` returns the "
            "value as `f64`. Do not modify `dep/`."),
        "observed_failure": (
            "error[E0277]: the trait bound `f64: From<u64>` is not satisfied\n"
            " --> src/lib.rs:4:18\n  |\n"
            "4 |     scale::scale(value)\n"
            "  |     ------------ ^^^^^ the trait `From<u64>` is not implemented for "
            "`f64`"),
        "files": {
            **_dep("scale", """
                //! Scale, version 2. The input is generic over `Into<f64>`.

                pub fn scale<T: Into<f64>>(value: T) -> f64 {
                    value.into()
                }
            """),
            "src/lib.rs": code("""
                //! Unit scaling. This does not build: `u64` does not convert into
                //! `f64` implicitly.

                pub fn scale_units(value: u64) -> f64 {
                    scale::scale(value)
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use migrate_18_conversion_bound::scale_units;

                #[test]
                fn a_small_value_scales() {
                    assert_eq!(scale_units(3), 3.0);
                }

                #[test]
                fn zero_scales_to_zero() {
                    assert_eq!(scale_units(0), 0.0);
                }

                #[test]
                fn a_large_value_keeps_precision_to_2_pow_40() {
                    assert_eq!(scale_units(1 << 40), 1_099_511_627_776.0);
                }
            """),
        },
        "readonly": ["dep/"],
        "reference": {
            "src/lib.rs": code("""
                //! Unit scaling.

                pub fn scale_units(value: u64) -> f64 {
                    scale::scale(value as f64)
                }
            """),
        },
        "manifest": _app("migrate_18_conversion_bound", 'scale = { path = "dep" }'),
        "mutation": {"file": "src/lib.rs", "find": "value as f64",
                     "replace": "(value / 2) as f64"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0277"]},
    },
]

# ---------------------------------------------------------------------------
# dependency-compat
#
# Every item is a real Cargo failure caused by the manifest, with the dependency
# vendored in `dep/` so the diagnosis needs no network. The arm may edit the
# manifest; the vendored tree is read-only.
# ---------------------------------------------------------------------------

_ANSWER = "pub fn answer() -> i32 {\n    42\n}\n"


def _plain(name: str, edition: str = "2021", extra: str = "") -> str:
    return (
        "[package]\n"
        f'name = "{name}"\n'
        'version = "0.1.0"\n'
        f'edition = "{edition}"\n'
        "\n"
        "[workspace]\n"
        "\n"
        "[lib]\n"
        'path = "src/lib.rs"\n'
        + extra
    )


def _dep_manifest(name: str, extra: str = "") -> str:
    return (
        "[package]\n"
        f'name = "{name}"\n'
        'version = "2.0.0"\n'
        'edition = "2021"\n'
        "\n"
        "[lib]\n"
        'path = "src/lib.rs"\n'
        + extra
    )


DEPENDENCY = [
    {
        "id": "deps-10-edition-unsafe-attribute",
        "category": "dependency-compat",
        "difficulty": "easy",
        "capability": "Recognise an edition-2024 rule that makes an attribute unsafe.",
        "prompt_intro": (
            "This crate does not build under its declared edition 2024. Edition 2024 "
            "requires `#[no_mangle]` to be written as an unsafe attribute. Keep the "
            "exported function and its behaviour; you may edit `Cargo.toml` or "
            "`src/lib.rs`."),
        "observed_failure": (
            "error: unsafe attribute used without unsafe\n --> src/lib.rs:1:3\n"
            "  |\n1 | #[no_mangle]\n  |   ^^^^^^^^^ attribute is unsafe\n"
            "  |\n  = help: consider adding `unsafe(...)` to this attribute"),
        "files": {
            "Cargo.toml": _plain("deps_10_edition_unsafe_attribute", "2024"),
            "src/lib.rs": code("""
                //! An exported entry point. This does not build under edition 2024.

                #[no_mangle]
                pub extern "C" fn answer() -> i32 {
                    42
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use deps_10_edition_unsafe_attribute::answer;

                #[test]
                fn the_exported_function_answers() {
                    assert_eq!(answer(), 42);
                }
            """),
        },
        "manifest_policy": "arm",
        "reference": {
            "Cargo.toml": _plain("deps_10_edition_unsafe_attribute", "2021"),
        },
        "mutation": {"file": "Cargo.toml", "find": 'edition = "2021"',
                     "replace": 'edition = "2024"'},
        "initial_failure": {"kind": "compile_error",
                            "must_contain": ["unsafe attribute used without unsafe"]},
    },
    {
        "id": "deps-11-lib-name-mismatch",
        "category": "dependency-compat",
        "difficulty": "medium",
        "capability": "Alias a dependency whose library target name differs from its package name.",
        "prompt_intro": (
            "This crate does not build. The vendored package is named `codec`, but "
            "its library target is named `codec_impl`, and the code imports "
            "`codec`. Rename the dependency to the library target name and keep the "
            "package mapping. You may edit `Cargo.toml` and `src/lib.rs`; do not "
            "modify `dep/`."),
        "observed_failure": (
            "error[E0433]: cannot find module or crate `codec` in this scope\n"
            " --> src/lib.rs:4:5\n  |\n4 |     codec::label()\n  |     ^^^^^ use of "
            "unresolved module or unlinked crate `codec`"),
        "files": {
            "Cargo.toml": _plain(
                "deps_11_lib_name_mismatch", extra='\n[dependencies]\ncodec = { path = "dep" }\n'),
            "dep/Cargo.toml": (
                "[package]\n"
                'name = "codec"\n'
                'version = "2.0.0"\n'
                'edition = "2021"\n'
                "\n"
                "[lib]\n"
                'name = "codec_impl"\n'
                'path = "src/lib.rs"\n'),
            "dep/src/lib.rs": code("""
                //! Codec, version 2. The library target is `codec_impl` while the
                //! package is `codec`.

                pub fn label() -> &'static str {
                    "codec 2.0.0"
                }
            """),
            "src/lib.rs": code("""
                //! A codec label. This does not build: the import name is not the
                //! library target name.

                pub fn codec_label() -> &'static str {
                    codec::label()
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use deps_11_lib_name_mismatch::codec_label;

                #[test]
                fn the_vendored_library_is_reached() {
                    assert_eq!(codec_label(), "codec 2.0.0");
                }
            """),
        },
        "readonly": ["dep/"],
        "manifest_policy": "arm",
        "reference": {
            "Cargo.toml": _plain(
                "deps_11_lib_name_mismatch",
                extra='\n[dependencies]\ncodec_impl = { path = "dep", package = "codec" }\n'),
            "src/lib.rs": code("""
                //! A codec label.

                pub fn codec_label() -> &'static str {
                    codec_impl::label()
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "codec_impl::label()",
                     "replace": "codec::label()"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0433"]},
    },
    {
        "id": "deps-12-edition-2015-paths",
        "category": "dependency-compat",
        "difficulty": "medium",
        "capability": "Diagnose the pre-2018 extern crate requirement for a dependency path.",
        "prompt_intro": (
            "This crate does not build. It declares edition 2015, where a dependency "
            "must be introduced with `extern crate` before a `use` path can name "
            "it. Move the crate to an edition where the import works without that "
            "declaration. Keep the behaviour: `label_of` returns the vendored "
            "label."),
        "observed_failure": (
            "error[E0432]: unresolved import `textkit`\n --> src/lib.rs:3:5\n  |\n"
            "3 | use textkit::label;\n  |     ^^^^^^^ maybe a missing crate `textkit`?\n"
            "  |\n  = help: consider adding `extern crate textkit` to use the crate"),
        "files": {
            "Cargo.toml": _plain("deps_12_edition_2015_paths", "2015",
                                 '\n[dependencies]\ntextkit = { path = "dep" }\n'),
            "dep/Cargo.toml": _dep_manifest("textkit"),
            "dep/src/lib.rs": code("""
                //! Textkit, version 2.

                pub fn label() -> &'static str {
                    "textkit 2.0.0"
                }
            """),
            "src/lib.rs": code("""
                //! A textkit label. This does not build under edition 2015.

                use textkit::label;

                pub fn label_of() -> &'static str {
                    label()
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use deps_12_edition_2015_paths::label_of;

                #[test]
                fn the_dependency_is_reachable() {
                    assert_eq!(label_of(), "textkit 2.0.0");
                }
            """),
        },
        "readonly": ["dep/"],
        "manifest_policy": "arm",
        "reference": {
            "Cargo.toml": _plain("deps_12_edition_2015_paths", "2021",
                                 '\n[dependencies]\ntextkit = { path = "dep" }\n'),
        },
        "mutation": {"file": "Cargo.toml", "find": 'edition = "2021"',
                     "replace": 'edition = "2015"'},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0432"]},
    },
    {
        "id": "deps-13-wrong-lib-path",
        "category": "dependency-compat",
        "difficulty": "easy",
        "capability": "Repair a library target path that points at a missing file.",
        "prompt_intro": (
            "This crate does not build. Its manifest points the library target at "
            "`src/other.rs`, which does not exist; the source lives at "
            "`src/lib.rs`. Repair the manifest. Keep the behaviour."),
        "observed_failure": (
            "error: can't find lib `deps_13_wrong_lib_path` at path "
            "`/work/.../src/other.rs`\n --> Cargo.toml\n"
            "error: could not compile due to 1 previous target resolution error"),
        "files": {
            "Cargo.toml": (
                "[package]\n"
                'name = "deps_13_wrong_lib_path"\n'
                'version = "0.1.0"\n'
                'edition = "2021"\n'
                "\n"
                "[workspace]\n"
                "\n"
                "[lib]\n"
                'path = "src/other.rs"\n'),
            "src/lib.rs": _ANSWER,
        },
        "hidden": {
            "tests/check.rs": code("""
                use deps_13_wrong_lib_path::answer;

                #[test]
                fn the_library_is_found() {
                    assert_eq!(answer(), 42);
                }
            """),
        },
        "manifest_policy": "arm",
        "reference": {
            "Cargo.toml": _plain("deps_13_wrong_lib_path"),
        },
        "mutation": {"file": "Cargo.toml", "find": 'path = "src/lib.rs"',
                     "replace": 'path = "src/other.rs"'},
        "initial_failure": {"kind": "compile_error", "must_contain": ["can't find lib"]},
    },
    {
        "id": "deps-14-workspace-inheritance",
        "category": "dependency-compat",
        "difficulty": "medium",
        "capability": "Resolve a workspace-inherited field that the root does not define.",
        "prompt_intro": (
            "This crate does not build. Its package inherits `version` from a "
            "workspace root that defines no `[workspace.package]` values, so the "
            "field has no source. Give the package an explicit version. Keep the "
            "behaviour."),
        "observed_failure": (
            "error: failed to parse manifest at `.../Cargo.toml`\nCaused by:\n"
            "  error inheriting `version` from workspace root manifest's "
            "`workspace.package.version`\n\nCaused by:\n"
            "  `workspace.package.version` was not defined"),
        "files": {
            "Cargo.toml": (
                "[package]\n"
                'name = "deps_14_workspace_inheritance"\n'
                "version.workspace = true\n"
                'edition = "2021"\n'
                "\n"
                "[workspace]\n"
                "\n"
                "[lib]\n"
                'path = "src/lib.rs"\n'),
            "src/lib.rs": _ANSWER,
        },
        "hidden": {
            "tests/check.rs": code("""
                use deps_14_workspace_inheritance::answer;

                #[test]
                fn the_package_has_a_version() {
                    assert_eq!(answer(), 42);
                }
            """),
        },
        "manifest_policy": "arm",
        "reference": {
            "Cargo.toml": _plain("deps_14_workspace_inheritance"),
        },
        "mutation": {"file": "Cargo.toml", "find": 'version = "0.1.0"',
                     "replace": "version.workspace = true"},
        "initial_failure": {"kind": "compile_error",
                            "must_contain": ["error inheriting `version`"]},
    },
    {
        "id": "deps-15-missing-package-name",
        "category": "dependency-compat",
        "difficulty": "easy",
        "capability": "Restore a required package field that the manifest omits.",
        "prompt_intro": (
            "This crate does not build. Its `[package]` table has no `name`, and "
            "Cargo cannot name the target. Add the package name "
            "`deps_15_missing_package_name`. Keep the behaviour."),
        "observed_failure": (
            "error: failed to parse manifest at `.../Cargo.toml`\nCaused by:\n"
            "  missing field `package.name`"),
        "files": {
            "Cargo.toml": (
                "[package]\n"
                'version = "0.1.0"\n'
                'edition = "2021"\n'
                "\n"
                "[workspace]\n"
                "\n"
                "[lib]\n"
                'path = "src/lib.rs"\n'),
            "src/lib.rs": _ANSWER,
        },
        "hidden": {
            "tests/check.rs": code("""
                use deps_15_missing_package_name::answer;

                #[test]
                fn the_package_is_named() {
                    assert_eq!(answer(), 42);
                }
            """),
        },
        "manifest_policy": "arm",
        "reference": {
            "Cargo.toml": _plain("deps_15_missing_package_name"),
        },
        "mutation": {"file": "Cargo.toml",
                     "find": 'name = "deps_15_missing_package_name"\n',
                     "replace": ""},
        "initial_failure": {"kind": "compile_error",
                            "must_contain": ["missing field `package.name`"]},
    },
    {
        "id": "deps-16-missing-dependency-dir",
        "category": "dependency-compat",
        "difficulty": "easy",
        "capability": "Point a path dependency at the directory that actually exists.",
        "prompt_intro": (
            "This crate does not build. Its manifest points the `codec` dependency "
            "at `dep`, but the vendored copy lives in `vendor`. Correct the path. "
            "Do not modify `vendor/`. Keep the behaviour."),
        "observed_failure": (
            "error: failed to load manifest for dependency `codec`\nCaused by:\n"
            "  failed to read `.../dep/Cargo.toml`\nCaused by:\n"
            "  No such file or directory (os error 2)"),
        "files": {
            "Cargo.toml": _plain(
                "deps_16_missing_dependency_dir",
                extra='\n[dependencies]\ncodec = { path = "dep" }\n'),
            "vendor/Cargo.toml": _dep_manifest("codec"),
            "vendor/src/lib.rs": code("""
                //! Codec, version 2.

                pub fn label() -> &'static str {
                    "codec 2.0.0"
                }
            """),
            "src/lib.rs": code("""
                //! A codec label. This does not build: the dependency path is wrong.

                pub fn codec_label() -> &'static str {
                    codec::label()
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use deps_16_missing_dependency_dir::codec_label;

                #[test]
                fn the_vendored_dependency_is_found() {
                    assert_eq!(codec_label(), "codec 2.0.0");
                }
            """),
        },
        "readonly": ["vendor/"],
        "manifest_policy": "arm",
        "reference": {
            "Cargo.toml": _plain(
                "deps_16_missing_dependency_dir",
                extra='\n[dependencies]\ncodec = { path = "vendor" }\n'),
            "src/lib.rs": code("""
                //! A codec label.

                pub fn codec_label() -> &'static str {
                    codec::label()
                }
            """),
        },
        "mutation": {"file": "Cargo.toml", "find": 'path = "vendor"',
                     "replace": 'path = "dep"'},
        "naive": {"file": "src/lib.rs", "find": "codec::label()",
                  "replace": "\"codec\""},
        "initial_failure": {"kind": "compile_error",
                            "must_contain": ["failed to load manifest for dependency"]},
    },
    {
        "id": "deps-17-invalid-package-name",
        "category": "dependency-compat",
        "difficulty": "easy",
        "capability": "Repair a package name that is not a valid crate identifier.",
        "prompt_intro": (
            "This crate does not build. Its package name contains a space, which "
            "Cargo rejects. Rename the package to `deps_17_invalid_package_name`. "
            "Keep the behaviour."),
        "observed_failure": (
            "error: invalid character ` ` in package name: `deps 17`, characters "
            "must be Unicode XID characters (numbers, `-`, `_`, or most letters)\n"
            " --> Cargo.toml:2:8\n  |\n2 | name = \"deps 17\""),
        "files": {
            "Cargo.toml": (
                "[package]\n"
                'name = "deps 17"\n'
                'version = "0.1.0"\n'
                'edition = "2021"\n'
                "\n"
                "[workspace]\n"
                "\n"
                "[lib]\n"
                'path = "src/lib.rs"\n'),
            "src/lib.rs": _ANSWER,
        },
        "hidden": {
            "tests/check.rs": code("""
                use deps_17_invalid_package_name::answer;

                #[test]
                fn the_package_name_is_a_crate_identifier() {
                    assert_eq!(answer(), 42);
                }
            """),
        },
        "manifest_policy": "arm",
        "reference": {
            "Cargo.toml": _plain("deps_17_invalid_package_name"),
        },
        "mutation": {"file": "Cargo.toml",
                     "find": 'name = "deps_17_invalid_package_name"',
                     "replace": 'name = "deps 17"'},
        "initial_failure": {"kind": "compile_error",
                            "must_contain": ["invalid character"]},
    },
    {
        "id": "deps-18-default-features-type",
        "category": "dependency-compat",
        "difficulty": "easy",
        "capability": "Correct a dependency field whose value has the wrong TOML type.",
        "prompt_intro": (
            "This crate does not build. The dependency's `default-features` value is "
            "the string `\"false\"`; Cargo requires a boolean. Correct the type. Do "
            "not modify `dep/`. Keep the behaviour."),
        "observed_failure": (
            "error: invalid type: string \"false\", expected a boolean\n"
            "  --> Cargo.toml:12:9\n   |\n12 | codec = { path = \"dep\", "
            "default-features = \"false\" }\n   |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^"),
        "files": {
            "Cargo.toml": _plain(
                "deps_18_default_features_type",
                extra='\n[dependencies]\ncodec = { path = "dep", default-features = "false" }\n'),
            "dep/Cargo.toml": _dep_manifest("codec", "\n[features]\ndefault = []\n"),
            "dep/src/lib.rs": code("""
                //! Codec, version 2. With default features disabled the same label is
                //! returned; the defect is the manifest type.

                pub fn label() -> &'static str {
                    "codec 2.0.0"
                }
            """),
            "src/lib.rs": code("""
                //! A codec label. This does not build: `default-features` is a string.

                pub fn codec_label() -> &'static str {
                    codec::label()
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use deps_18_default_features_type::codec_label;

                #[test]
                fn the_dependency_is_built_without_default_features() {
                    assert_eq!(codec_label(), "codec 2.0.0");
                }
            """),
        },
        "readonly": ["dep/"],
        "manifest_policy": "arm",
        "reference": {
            "Cargo.toml": _plain(
                "deps_18_default_features_type",
                extra='\n[dependencies]\ncodec = { path = "dep", default-features = false }\n'),
        },
        "mutation": {"file": "Cargo.toml", "find": "default-features = false",
                     "replace": 'default-features = "false"'},
        "initial_failure": {"kind": "compile_error",
                            "must_contain": ["expected a boolean"]},
    },
]

# ---------------------------------------------------------------------------
# concurrency
# ---------------------------------------------------------------------------

CONCURRENCY = [
    {
        "id": "concurrency-10-condvar-without-predicate",
        "category": "concurrency",
        "difficulty": "medium",
        "capability": "Wait on a condition variable inside a predicate loop.",
        "prompt_intro": (
            "The crate builds, but `run` never returns: the readiness signal is sent "
            "before the waiter parks, and `wait_for_ready` waits without checking "
            "the predicate, so the wake-up is lost. Wait in a predicate loop and "
            "return the flag. A signal delivered before the wait must not be lost."),
        "observed_failure": (
            "running 1 test\ntest tests::visible_run ... "),
        "files": {
            "src/lib.rs": code("""
                //! A readiness signal. The crate builds, but `run` never returns: the
                //! notification is sent before the waiter parks.

                use std::sync::{Arc, Condvar, Mutex};

                pub fn wait_for_ready(state: Arc<(Mutex<bool>, Condvar)>) -> bool {
                    let (lock, ready) = &*state;
                    let guard = lock.lock().unwrap();
                    let _released = ready.wait(guard).unwrap();
                    true
                }

                pub fn run() -> bool {
                    let state = Arc::new((Mutex::new(false), Condvar::new()));
                    {
                        let (lock, ready) = &*state;
                        let mut flag = lock.lock().unwrap();
                        *flag = true;
                        ready.notify_all();
                    }
                    wait_for_ready(state)
                }

                #[cfg(test)]
                mod tests {
                    #[test]
                    fn visible_run_returns() {
                        assert!(super::run());
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use std::sync::{Arc, Condvar, Mutex};

                use concurrency_10_condvar_without_predicate::{run, wait_for_ready};

                #[test]
                fn a_signal_before_the_wait_is_not_lost() {
                    assert!(run());
                }

                #[test]
                fn an_already_ready_state_returns_immediately() {
                    let state = Arc::new((Mutex::new(true), Condvar::new()));
                    assert!(wait_for_ready(state));
                }

                #[test]
                fn a_late_signal_wakes_the_waiter() {
                    let state = Arc::new((Mutex::new(false), Condvar::new()));
                    let clone = Arc::clone(&state);
                    let handle = std::thread::spawn(move || {
                        std::thread::sleep(std::time::Duration::from_millis(20));
                        let (lock, ready) = &*clone;
                        let mut flag = lock.lock().unwrap();
                        *flag = true;
                        ready.notify_all();
                    });
                    assert!(wait_for_ready(state));
                    handle.join().unwrap();
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! A readiness signal.

                use std::sync::{Arc, Condvar, Mutex};

                pub fn wait_for_ready(state: Arc<(Mutex<bool>, Condvar)>) -> bool {
                    let (lock, ready) = &*state;
                    let mut guard = lock.lock().unwrap();
                    while !*guard {
                        guard = ready.wait(guard).unwrap();
                    }
                    *guard
                }

                pub fn run() -> bool {
                    let state = Arc::new((Mutex::new(false), Condvar::new()));
                    {
                        let (lock, ready) = &*state;
                        let mut flag = lock.lock().unwrap();
                        *flag = true;
                        ready.notify_all();
                    }
                    wait_for_ready(state)
                }
            """),
        },
        "limits": {"wall_seconds": 25.0, "cpu_seconds": 40},
        "mutation": {"file": "src/lib.rs", "find": "while !*guard {",
                     "replace": "while !*guard && false {"},
        "initial_failure": {"kind": "timeout", "must_contain": ["running 1 test"]},
    },
    {
        "id": "concurrency-11-cell-not-sync",
        "category": "concurrency",
        "difficulty": "easy",
        "capability": "Replace an unsynchronised cell with an atomic shared across threads.",
        "prompt_intro": (
            "This crate does not compile. `count_with` shares a `Cell<u64>` between "
            "scoped threads, and `Cell` is not `Sync`. Use an `AtomicU64` so every "
            "worker adds exactly one, and return the final count. Change the "
            "parameter type; keep the name and the thread count."),
        "observed_failure": (
            "error[E0277]: `Cell<u64>` cannot be shared between threads safely\n"
            " --> src/lib.rs:5:24\n  |\n5 |             scope.spawn(|| {\n"
            "  |                   ----- ^^ `Cell<u64>` is not `Sync`"),
        "files": {
            "src/lib.rs": code("""
                //! Count from several threads. The crate does not compile: `Cell` is
                //! not `Sync`.

                use std::cell::Cell;

                pub fn count_with(counter: &Cell<u64>, threads: usize) -> u64 {
                    std::thread::scope(|scope| {
                        for _ in 0..threads {
                            scope.spawn(|| {
                                counter.set(counter.get() + 1);
                            });
                        }
                    });
                    counter.get()
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use std::sync::atomic::{AtomicU64, Ordering};

                use concurrency_11_cell_not_sync::count_with;

                #[test]
                fn every_thread_adds_one() {
                    let counter = AtomicU64::new(0);
                    assert_eq!(count_with(&counter, 4), 4);
                    assert_eq!(counter.load(Ordering::SeqCst), 4);
                }

                #[test]
                fn the_count_starts_from_the_current_value() {
                    let counter = AtomicU64::new(10);
                    assert_eq!(count_with(&counter, 3), 13);
                }

                #[test]
                fn zero_threads_change_nothing() {
                    let counter = AtomicU64::new(7);
                    assert_eq!(count_with(&counter, 0), 7);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Count from several threads.

                use std::sync::atomic::{AtomicU64, Ordering};

                pub fn count_with(counter: &AtomicU64, threads: usize) -> u64 {
                    std::thread::scope(|scope| {
                        for _ in 0..threads {
                            scope.spawn(|| {
                                counter.fetch_add(1, Ordering::SeqCst);
                            });
                        }
                    });
                    counter.load(Ordering::SeqCst)
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "counter.fetch_add(1, Ordering::SeqCst)",
                     "replace": "counter.fetch_add(2, Ordering::SeqCst)"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0277"]},
    },
    {
        "id": "concurrency-12-shared-arc-moved",
        "category": "concurrency",
        "difficulty": "easy",
        "capability": "Clone a shared handle for each worker instead of moving it once.",
        "prompt_intro": (
            "This crate does not compile. The worker loop moves the `Arc` into the "
            "first closure and then uses it again for the next worker. Clone the "
            "handle per worker. Each worker sums the values at indices congruent to "
            "its own index modulo the thread count."),
        "observed_failure": (
            "error[E0382]: use of moved value: `values`\n --> src/lib.rs:9:41\n  |\n"
            "7 |         handles.push(std::thread::spawn(move || {\n"
            "  |                                        ------- value moved into closure here, "
            "in previous iteration of loop\n9 |         handles.push(std::thread::spawn(move || {\n"
            "  |                                         ^^^^ value used here after move"),
        "files": {
            "src/lib.rs": code("""
                //! A chunked sum. The crate does not compile: the shared handle is
                //! moved into the first worker and reused in the next.

                use std::sync::Arc;

                pub fn chunked_sum(values: Arc<Vec<u64>>, threads: usize) -> u64 {
                    let mut handles = Vec::new();
                    for index in 0..threads {
                        handles.push(std::thread::spawn(move || {
                            values.iter().skip(index).step_by(threads).sum::<u64>()
                        }));
                    }
                    handles.into_iter().map(|handle| handle.join().unwrap()).sum()
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use std::sync::Arc;

                use concurrency_12_shared_arc_moved::chunked_sum;

                #[test]
                fn two_workers_cover_every_index() {
                    let values = Arc::new(vec![1, 2, 3, 4]);
                    assert_eq!(chunked_sum(values, 2), 10);
                }

                #[test]
                fn one_worker_sums_everything() {
                    let values = Arc::new(vec![5, 6, 7]);
                    assert_eq!(chunked_sum(values, 1), 18);
                }

                #[test]
                fn an_empty_vector_sums_to_zero() {
                    let values = Arc::new(Vec::new());
                    assert_eq!(chunked_sum(values, 3), 0);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! A chunked sum.

                use std::sync::Arc;

                pub fn chunked_sum(values: Arc<Vec<u64>>, threads: usize) -> u64 {
                    let mut handles = Vec::new();
                    for index in 0..threads {
                        let values = Arc::clone(&values);
                        handles.push(std::thread::spawn(move || {
                            values.iter().skip(index).step_by(threads).sum::<u64>()
                        }));
                    }
                    handles.into_iter().map(|handle| handle.join().unwrap()).sum()
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "step_by(threads)", "replace": "step_by(1)"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0382"]},
    },
    {
        "id": "concurrency-13-atomic-fetch-or",
        "category": "concurrency",
        "difficulty": "easy",
        "capability": "Use the atomic operation that matches the accumulated operation.",
        "prompt_intro": (
            "The visible test fails. `accumulate` needs to add each value to the "
            "counter, but it uses a bitwise OR. Replace it with the atomic add that "
            "matches the contract, and keep the starting value."),
        "observed_failure": (
            "thread 'tests::visible_pair' panicked at src/lib.rs:20:9:\n"
            "assertion `left == right` failed\n  left: 1\n right: 2"),
        "files": {
            "src/lib.rs": code("""
                //! Accumulate values into a counter. The visible test fails: the
                //! counter is combined with OR instead of addition.

                use std::sync::atomic::{AtomicUsize, Ordering};

                pub fn accumulate(counter: &AtomicUsize, values: &[usize]) -> usize {
                    for value in values {
                        counter.fetch_or(*value, Ordering::SeqCst);
                    }
                    counter.load(Ordering::SeqCst)
                }

                #[cfg(test)]
                mod tests {
                    use super::accumulate;
                    use std::sync::atomic::AtomicUsize;

                    #[test]
                    fn visible_pair() {
                        let counter = AtomicUsize::new(0);
                        assert_eq!(accumulate(&counter, &[1, 1]), 2);
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use std::sync::atomic::AtomicUsize;

                use concurrency_13_atomic_fetch_or::accumulate;

                #[test]
                fn repeated_values_add() {
                    let counter = AtomicUsize::new(0);
                    assert_eq!(accumulate(&counter, &[1, 1]), 2);
                }

                #[test]
                fn distinct_values_add() {
                    let counter = AtomicUsize::new(0);
                    assert_eq!(accumulate(&counter, &[2, 3]), 5);
                }

                #[test]
                fn the_starting_value_is_kept() {
                    let counter = AtomicUsize::new(10);
                    assert_eq!(accumulate(&counter, &[8, 1]), 19);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Accumulate values into a counter.

                use std::sync::atomic::{AtomicUsize, Ordering};

                pub fn accumulate(counter: &AtomicUsize, values: &[usize]) -> usize {
                    for value in values {
                        counter.fetch_add(*value, Ordering::SeqCst);
                    }
                    counter.load(Ordering::SeqCst)
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "counter.fetch_add(*value, Ordering::SeqCst)",
                     "replace": "counter.fetch_add(*value + 1, Ordering::SeqCst)"},
        "naive": {"file": "src/lib.rs",
                  "find": "counter.fetch_add(*value, Ordering::SeqCst);",
                  "replace": "counter.fetch_add(1, Ordering::SeqCst);"},
        "initial_failure": {"kind": "test_failure",
                            "must_contain": ["assertion `left == right` failed"]},
    },
    {
        "id": "concurrency-14-poisoned-mutex",
        "category": "concurrency",
        "difficulty": "hard",
        "capability": "Recover the data of a mutex poisoned by a panicking worker.",
        "prompt_intro": (
            "The visible test fails: a worker panics while holding the lock, which "
            "poisons the mutex, and the subsequent `lock().unwrap()` panics on the "
            "poison marker. The bytes the worker recorded before the panic are still "
            "valid. Recover the poisoned guard and return the recorded length."),
        "observed_failure": (
            "thread 'tests::visible_after_panic' panicked at src/lib.rs:21:34:\n"
            "called `Result::unwrap()` on an `Err` value: PoisonError { .. }"),
        "files": {
            "src/lib.rs": code("""
                //! Record bytes and survive a worker panic. The visible test fails:
                //! the poisoned mutex is unwrapped instead of recovered.

                use std::sync::Mutex;

                pub fn record_with_panic(shared: &Mutex<Vec<u8>>) -> usize {
                    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        std::thread::scope(|scope| {
                            scope.spawn(|| {
                                let mut guard = shared.lock().unwrap();
                                guard.push(7);
                                panic!("worker failed");
                            });
                        });
                    }));
                    let _ = caught;
                    let guard = shared.lock().unwrap();
                    guard.len()
                }

                #[cfg(test)]
                mod tests {
                    use super::record_with_panic;
                    use std::sync::Mutex;

                    #[test]
                    fn visible_after_panic() {
                        let shared = Mutex::new(Vec::new());
                        assert_eq!(record_with_panic(&shared), 1);
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use std::sync::Mutex;

                use concurrency_14_poisoned_mutex::record_with_panic;

                #[test]
                fn the_recorded_bytes_survive_the_panic() {
                    let shared = Mutex::new(Vec::new());
                    assert_eq!(record_with_panic(&shared), 1);
                }

                #[test]
                fn the_worker_byte_is_visible_afterwards() {
                    let shared = Mutex::new(Vec::new());
                    record_with_panic(&shared);
                    let guard = shared.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                    assert_eq!(*guard, vec![7]);
                }

                #[test]
                fn existing_bytes_are_kept() {
                    let shared = Mutex::new(vec![1, 2]);
                    assert_eq!(record_with_panic(&shared), 3);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Record bytes and survive a worker panic.

                use std::sync::Mutex;

                pub fn record_with_panic(shared: &Mutex<Vec<u8>>) -> usize {
                    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        std::thread::scope(|scope| {
                            scope.spawn(|| {
                                let mut guard = shared.lock().unwrap();
                                guard.push(7);
                                panic!("worker failed");
                            });
                        });
                    }));
                    let _ = caught;
                    let guard = shared
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    guard.len()
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "guard.push(7);",
                     "replace": "guard.push(8);"},
        "initial_failure": {"kind": "test_failure", "must_contain": ["PoisonError"]},
    },
    {
        "id": "concurrency-15-mutex-read-modify-write",
        "category": "concurrency",
        "difficulty": "medium",
        "capability": "Hold the mutex across a read-modify-write instead of around each half.",
        "prompt_intro": (
            "The visible test fails. Both workers read the counter under the lock, "
            "meet at a barrier after dropping it, and then write the value they "
            "read, so an update is lost. Hold the lock across the whole update. "
            "`bump_pair` adds two, and `bump_many(counter, threads, per_thread)` "
            "adds `threads * per_thread`."),
        "observed_failure": (
            "thread 'tests::visible_pair' panicked at src/lib.rs:33:9:\n"
            "assertion `left == right` failed\n  left: 1\n right: 2"),
        "files": {
            "src/lib.rs": code("""
                //! A mutex-protected counter. The visible test fails: the read and
                //! the write are in separate critical sections.

                use std::sync::{Barrier, Mutex};

                pub fn bump_pair(counter: &Mutex<u64>) {
                    let barrier = Barrier::new(2);
                    std::thread::scope(|scope| {
                        for _ in 0..2 {
                            scope.spawn(|| {
                                let value = { *counter.lock().unwrap() };
                                barrier.wait();
                                let mut guard = counter.lock().unwrap();
                                *guard = value + 1;
                            });
                        }
                    });
                }

                pub fn bump_many(counter: &Mutex<u64>, threads: usize, per_thread: usize) {
                    let barrier = Barrier::new(threads);
                    std::thread::scope(|scope| {
                        for _ in 0..threads {
                            scope.spawn(|| {
                                for _ in 0..per_thread {
                                    let value = { *counter.lock().unwrap() };
                                    barrier.wait();
                                    let mut guard = counter.lock().unwrap();
                                    *guard = value + 1;
                                }
                            });
                        }
                    });
                }

                #[cfg(test)]
                mod tests {
                    use super::bump_pair;
                    use std::sync::Mutex;

                    #[test]
                    fn visible_pair() {
                        let counter = Mutex::new(0);
                        bump_pair(&counter);
                        assert_eq!(*counter.lock().unwrap(), 2);
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use std::sync::Mutex;

                use concurrency_15_mutex_read_modify_write::{bump_many, bump_pair};

                #[test]
                fn a_pair_adds_two() {
                    let counter = Mutex::new(0);
                    bump_pair(&counter);
                    assert_eq!(*counter.lock().unwrap(), 2);
                }

                #[test]
                fn many_updates_are_not_lost() {
                    let counter = Mutex::new(0);
                    bump_many(&counter, 4, 250);
                    assert_eq!(*counter.lock().unwrap(), 1000);
                }

                #[test]
                fn updates_start_from_the_current_value() {
                    let counter = Mutex::new(41);
                    bump_pair(&counter);
                    assert_eq!(*counter.lock().unwrap(), 43);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! A mutex-protected counter.

                use std::sync::{Barrier, Mutex};

                pub fn bump_pair(counter: &Mutex<u64>) {
                    let barrier = Barrier::new(2);
                    std::thread::scope(|scope| {
                        for _ in 0..2 {
                            scope.spawn(|| {
                                barrier.wait();
                                let mut guard = counter.lock().unwrap();
                                *guard += 1;
                            });
                        }
                    });
                }

                pub fn bump_many(counter: &Mutex<u64>, threads: usize, per_thread: usize) {
                    std::thread::scope(|scope| {
                        for _ in 0..threads {
                            scope.spawn(|| {
                                for _ in 0..per_thread {
                                    let mut guard = counter.lock().unwrap();
                                    *guard += 1;
                                }
                            });
                        }
                    });
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "*guard += 1;",
                     "replace": "*guard += 2;"},
        "initial_failure": {"kind": "test_failure",
                            "must_contain": ["assertion `left == right` failed"]},
    },
    {
        "id": "concurrency-16-non-reentrant-lock",
        "category": "concurrency",
        "difficulty": "medium",
        "capability": "Take a non-reentrant mutex exactly once on a code path.",
        "prompt_intro": (
            "The crate builds, but `read_twice` never returns: `std::sync::Mutex` is "
            "not reentrant, and the function takes the same lock twice. Take it "
            "once and return twice the value. The lock is uncontended; there is no "
            "second writer."),
        "observed_failure": (
            "running 1 test\ntest tests::visible_read ... "),
        "files": {
            "src/lib.rs": code("""
                //! Read a mutex-protected value twice. The crate builds, but the
                //! function takes the same non-reentrant lock twice.

                use std::sync::Mutex;

                pub fn read_twice(shared: &Mutex<u32>) -> u32 {
                    let first = shared.lock().unwrap();
                    let second = shared.lock().unwrap();
                    *first + *second
                }

                #[cfg(test)]
                mod tests {
                    use super::read_twice;
                    use std::sync::Mutex;

                    #[test]
                    fn visible_read() {
                        let shared = Mutex::new(21);
                        assert_eq!(read_twice(&shared), 42);
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use std::sync::Mutex;

                use concurrency_16_non_reentrant_lock::read_twice;

                #[test]
                fn doubles_the_value() {
                    let shared = Mutex::new(21);
                    assert_eq!(read_twice(&shared), 42);
                }

                #[test]
                fn zero_doubles_to_zero() {
                    let shared = Mutex::new(0);
                    assert_eq!(read_twice(&shared), 0);
                }

                #[test]
                fn the_lock_is_released_afterwards() {
                    let shared = Mutex::new(4);
                    assert_eq!(read_twice(&shared), 8);
                    assert_eq!(*shared.lock().unwrap(), 4);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Read a mutex-protected value twice.

                use std::sync::Mutex;

                pub fn read_twice(shared: &Mutex<u32>) -> u32 {
                    let value = *shared.lock().unwrap();
                    value * 2
                }
            """),
        },
        "limits": {"wall_seconds": 25.0, "cpu_seconds": 40},
        "mutation": {"file": "src/lib.rs", "find": "value * 2", "replace": "value * 3"},
        "initial_failure": {"kind": "timeout", "must_contain": ["running 1 test"]},
    },
    {
        "id": "concurrency-17-compare-exchange-claim",
        "category": "concurrency",
        "difficulty": "medium",
        "capability": "Claim a one-shot flag with compare-exchange on the released value.",
        "prompt_intro": (
            "The visible test fails. `claim` must succeed exactly once for a flag: it "
            "sets the flag only when it is still free, and reports whether this caller "
            "did the claiming. The comparison currently uses the claimed value as the "
            "expected value, so no caller ever wins."),
        "observed_failure": (
            "thread 'tests::visible_first_claim' panicked at src/lib.rs:12:9:\n"
            "assertion failed: claim(&flag)"),
        "files": {
            "src/lib.rs": code("""
                //! A one-shot claim. The visible test fails: the compare-exchange
                //! compares against the claimed value, not the free one.

                use std::sync::atomic::{AtomicBool, Ordering};

                pub fn claim(flag: &AtomicBool) -> bool {
                    flag.compare_exchange(true, true, Ordering::SeqCst, Ordering::SeqCst)
                        .is_ok()
                }

                #[cfg(test)]
                mod tests {
                    use super::claim;
                    use std::sync::atomic::AtomicBool;

                    #[test]
                    fn visible_first_claim() {
                        let flag = AtomicBool::new(false);
                        assert!(claim(&flag));
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use std::sync::atomic::{AtomicBool, Ordering};

                use concurrency_17_compare_exchange_claim::claim;

                #[test]
                fn the_first_claim_wins() {
                    let flag = AtomicBool::new(false);
                    assert!(claim(&flag));
                }

                #[test]
                fn the_second_claim_loses() {
                    let flag = AtomicBool::new(false);
                    assert!(claim(&flag));
                    assert!(!claim(&flag));
                }

                #[test]
                fn a_released_flag_can_be_claimed_again() {
                    let flag = AtomicBool::new(true);
                    flag.store(false, Ordering::SeqCst);
                    assert!(claim(&flag));
                }

                #[test]
                fn independent_flags_are_independent() {
                    let first = AtomicBool::new(false);
                    let second = AtomicBool::new(false);
                    assert!(claim(&first));
                    assert!(claim(&second));
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! A one-shot claim.

                use std::sync::atomic::{AtomicBool, Ordering};

                pub fn claim(flag: &AtomicBool) -> bool {
                    flag.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                        .is_ok()
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "compare_exchange(false, true,",
                     "replace": "compare_exchange(false, false,"},
        "initial_failure": {"kind": "test_failure", "must_contain": ["assertion failed"]},
    },
    {
        "id": "concurrency-18-barrier-early-return",
        "category": "concurrency",
        "difficulty": "hard",
        "capability": "Match the barrier's participant count to the threads that reach it.",
        "prompt_intro": (
            "The crate builds, but `sum_nonzero` never returns for input containing "
            "a zero: the barrier is sized for every element, and the zero workers "
            "return before the rendezvous, so the others wait forever. Size the "
            "barrier for the workers that actually rendezvous. Zero values are "
            "ignored; the result is the sum of the non-zero values."),
        "observed_failure": (
            "running 1 test\ntest tests::visible_with_zero ... "),
        "files": {
            "src/lib.rs": code("""
                //! A barrier-guarded sum. The crate builds, but the barrier is sized
                //! for workers that leave before it.

                use std::sync::atomic::{AtomicUsize, Ordering};
                use std::sync::Barrier;

                pub fn sum_nonzero(values: &[u8]) -> usize {
                    let barrier = Barrier::new(values.len());
                    let total = AtomicUsize::new(0);
                    std::thread::scope(|scope| {
                        for value in values {
                            scope.spawn(|| {
                                if *value == 0 {
                                    return;
                                }
                                barrier.wait();
                                total.fetch_add(*value as usize, Ordering::SeqCst);
                            });
                        }
                    });
                    total.load(Ordering::SeqCst)
                }

                #[cfg(test)]
                mod tests {
                    #[test]
                    fn visible_with_zero() {
                        assert_eq!(super::sum_nonzero(&[1, 0, 2]), 3);
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use concurrency_18_barrier_early_return::sum_nonzero;

                #[test]
                fn a_zero_is_ignored() {
                    assert_eq!(sum_nonzero(&[1, 0, 2]), 3);
                }

                #[test]
                fn every_value_is_summed() {
                    assert_eq!(sum_nonzero(&[2, 3]), 5);
                }

                #[test]
                fn all_zeroes_sum_to_zero() {
                    assert_eq!(sum_nonzero(&[0, 0]), 0);
                }

                #[test]
                fn empty_input_sums_to_zero() {
                    assert_eq!(sum_nonzero(&[]), 0);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! A barrier-guarded sum.

                use std::sync::atomic::{AtomicUsize, Ordering};
                use std::sync::Barrier;

                pub fn sum_nonzero(values: &[u8]) -> usize {
                    let participants = values.iter().filter(|value| **value != 0).count().max(1);
                    let barrier = Barrier::new(participants);
                    let total = AtomicUsize::new(0);
                    std::thread::scope(|scope| {
                        for value in values {
                            scope.spawn(|| {
                                if *value == 0 {
                                    return;
                                }
                                barrier.wait();
                                total.fetch_add(*value as usize, Ordering::SeqCst);
                            });
                        }
                    });
                    total.load(Ordering::SeqCst)
                }
            """),
        },
        "limits": {"wall_seconds": 25.0, "cpu_seconds": 40},
        "mutation": {"file": "src/lib.rs", "find": "total.fetch_add(*value as usize, Ordering::SeqCst)",
                     "replace": "total.fetch_add(*value as usize * 2, Ordering::SeqCst)"},
        "initial_failure": {"kind": "timeout", "must_contain": ["running 1 test"]},
    },
]

# ---------------------------------------------------------------------------
# trait-impl
# ---------------------------------------------------------------------------

TRAITS = [
    {
        "id": "traits-10-associated-type-value",
        "category": "trait-impl",
        "difficulty": "medium",
        "capability": "Make an associated type agree with the values the method yields.",
        "prompt_intro": (
            "This crate does not compile. `Counter` declares `type Item = u32` but "
            "its `next_item` yields a `u64`. Fix the associated type so the impl "
            "matches the method and the generic `drain` helper accepts `Counter`."),
        "observed_failure": (
            "error[E0308]: mismatched types\n --> src/lib.rs:12:54\n  |\n"
            "12 |     fn next_item(&mut self) -> Option<Self::Item> { Some(self.current as u64) }\n"
            "  |                                                      ^^^^^^^^^^^^^^^^^^^^^^^^ "
            "expected `u32`, found `u64`"),
        "files": {
            "src/lib.rs": code("""
                //! A sequence with an associated item type. The crate does not build:
                //! the associated type and the produced value disagree.

                pub trait Sequence {
                    type Item;
                    fn next_item(&mut self) -> Option<Self::Item>;
                }

                pub struct Counter {
                    pub current: u32,
                }

                impl Sequence for Counter {
                    type Item = u32;

                    fn next_item(&mut self) -> Option<Self::Item> {
                        self.current += 1;
                        Some(self.current as u64)
                    }
                }

                pub fn drain<S: Sequence<Item = u64>>(sequence: &mut S, limit: usize) -> Vec<u64> {
                    let mut out = Vec::new();
                    for _ in 0..limit {
                        match sequence.next_item() {
                            Some(item) => out.push(item),
                            None => break,
                        }
                    }
                    out
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use traits_10_associated_type_value::{drain, Counter};

                #[test]
                fn drains_the_requested_number_of_items() {
                    let mut counter = Counter { current: 0 };
                    assert_eq!(drain(&mut counter, 3), vec![1, 2, 3]);
                }

                #[test]
                fn a_zero_limit_drains_nothing() {
                    let mut counter = Counter { current: 0 };
                    assert!(drain(&mut counter, 0).is_empty());
                }

                #[test]
                fn the_counter_advances() {
                    let mut counter = Counter { current: 10 };
                    assert_eq!(drain(&mut counter, 1), vec![11]);
                    assert_eq!(counter.current, 11);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! A sequence with an associated item type.

                pub trait Sequence {
                    type Item;
                    fn next_item(&mut self) -> Option<Self::Item>;
                }

                pub struct Counter {
                    pub current: u32,
                }

                impl Sequence for Counter {
                    type Item = u64;

                    fn next_item(&mut self) -> Option<Self::Item> {
                        self.current += 1;
                        Some(self.current as u64)
                    }
                }

                pub fn drain<S: Sequence<Item = u64>>(sequence: &mut S, limit: usize) -> Vec<u64> {
                    let mut out = Vec::new();
                    for _ in 0..limit {
                        match sequence.next_item() {
                            Some(item) => out.push(item),
                            None => break,
                        }
                    }
                    out
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "self.current += 1;",
                     "replace": "self.current += 2;"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0308"]},
    },
    {
        "id": "traits-11-blanket-impl-bound",
        "category": "trait-impl",
        "difficulty": "medium",
        "capability": "Add the bound a blanket impl's body requires.",
        "prompt_intro": (
            "This crate does not compile. The blanket impl formats `self`, which "
            "requires `Display`, but the impl has no such bound. Add the bound; do "
            "not restrict the impl to a concrete type."),
        "observed_failure": (
            "error[E0277]: `T` doesn't implement `std::fmt::Display`\n"
            " --> src/lib.rs:4:49\n  |\n"
            "4 |     fn summary(&self) -> String { format!(\"{}\", self) }\n"
            "  |                                                 ^^^^ `T` cannot be formatted "
            "with the default formatter"),
        "files": {
            "src/lib.rs": code("""
                //! A blanket summary. The crate does not build: the impl formats `T`
                //! without requiring `Display`.

                pub trait Summary {
                    fn summary(&self) -> String;
                }

                impl<T> Summary for T {
                    fn summary(&self) -> String {
                        format!("{}", self)
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use traits_11_blanket_impl_bound::Summary;

                #[test]
                fn integers_are_summarised() {
                    assert_eq!(12u8.summary(), "12");
                }

                #[test]
                fn strings_are_summarised() {
                    assert_eq!("alpha".summary(), "alpha");
                }

                #[test]
                fn the_display_form_is_used() {
                    assert_eq!(format!("{:.2}", 1.5f64).summary(), "1.50");
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! A blanket summary.

                use std::fmt::Display;

                pub trait Summary {
                    fn summary(&self) -> String;
                }

                impl<T: Display> Summary for T {
                    fn summary(&self) -> String {
                        format!("{}", self)
                    }
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": 'format!("{}", self)',
                     "replace": 'format!("<{}>", self)'},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0277"]},
    },
    {
        "id": "traits-12-associated-const",
        "category": "trait-impl",
        "difficulty": "easy",
        "capability": "Supply an associated constant the trait requires.",
        "prompt_intro": (
            "This crate does not compile. The `Limits` trait requires an associated "
            "constant `MAX`, and the `Small` impl omits it. Add `MAX = 8` and cap "
            "through it so the constant and the behaviour agree."),
        "observed_failure": (
            "error[E0046]: not all trait items implemented, missing: `MAX`\n"
            " --> src/lib.rs:5:1\n  |\n5 | impl Limits for Small {\n"
            "  | ^^^^^^^^^^^^^^^^^^^^^^ missing `MAX` in implementation"),
        "files": {
            "src/lib.rs": code("""
                //! A trait with an associated constant. The crate does not build: the
                //! impl omits `MAX`.

                pub trait Limits {
                    const MAX: usize;
                    fn capped(&self, value: usize) -> usize;
                }

                pub struct Small;

                impl Limits for Small {
                    fn capped(&self, value: usize) -> usize {
                        value.min(8)
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use traits_12_associated_const::{Limits, Small};

                #[test]
                fn the_constant_is_eight() {
                    assert_eq!(Small::MAX, 8);
                }

                #[test]
                fn values_above_the_limit_are_capped() {
                    assert_eq!(Small.capped(20), 8);
                }

                #[test]
                fn values_below_the_limit_are_kept() {
                    assert_eq!(Small.capped(3), 3);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! A trait with an associated constant.

                pub trait Limits {
                    const MAX: usize;
                    fn capped(&self, value: usize) -> usize;
                }

                pub struct Small;

                impl Limits for Small {
                    const MAX: usize = 8;

                    fn capped(&self, value: usize) -> usize {
                        value.min(Self::MAX)
                    }
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "const MAX: usize = 8;",
                     "replace": "const MAX: usize = 9;"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0046"]},
    },
    {
        "id": "traits-13-default-method-missing-item",
        "category": "trait-impl",
        "difficulty": "easy",
        "capability": "Call the trait item that exists from a default method.",
        "prompt_intro": (
            "This crate does not compile. The trait's default `render` calls "
            "`self.heading()`, but the trait declares `title`, not `heading`. Use "
            "the declared item and keep the `[title]` form."),
        "observed_failure": (
            "error[E0599]: no method named `heading` found for reference `&Self` in "
            "the current scope\n --> src/lib.rs:3:55\n  |\n"
            "3 |     fn render(&self) -> String { format!(\"[{}]\", self.heading()) }\n"
            "  |                                                       ^^^^^^^ help: there is a "
            "method with a similar name: `title`"),
        "files": {
            "src/lib.rs": code("""
                //! A report trait with a default method. The crate does not build: the
                //! default method calls an item the trait does not declare.

                pub trait Report {
                    fn title(&self) -> String;

                    fn render(&self) -> String {
                        format!("[{}]", self.heading())
                    }
                }

                pub struct Doc {
                    pub text: String,
                }

                impl Report for Doc {
                    fn title(&self) -> String {
                        self.text.clone()
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use traits_13_default_method_missing_item::{Doc, Report};

                #[test]
                fn the_default_render_wraps_the_title() {
                    let doc = Doc { text: "x".to_string() };
                    assert_eq!(doc.render(), "[x]");
                }

                #[test]
                fn the_title_is_available_directly() {
                    let doc = Doc { text: "alpha".to_string() };
                    assert_eq!(doc.title(), "alpha");
                }

                #[test]
                fn an_empty_title_renders_empty_brackets() {
                    let doc = Doc { text: String::new() };
                    assert_eq!(doc.render(), "[]");
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! A report trait with a default method.

                pub trait Report {
                    fn title(&self) -> String;

                    fn render(&self) -> String {
                        format!("[{}]", self.title())
                    }
                }

                pub struct Doc {
                    pub text: String,
                }

                impl Report for Doc {
                    fn title(&self) -> String {
                        self.text.clone()
                    }
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": 'format!("[{}]", self.title())',
                     "replace": 'format!("({})", self.title())'},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0599"]},
    },
    {
        "id": "traits-14-extra-type-parameter",
        "category": "trait-impl",
        "difficulty": "medium",
        "capability": "Match the trait method's generic parameter list exactly.",
        "prompt_intro": (
            "This crate does not compile. The `Encode` trait's `encode` method has no "
            "type parameters, but the impl declares one. Remove the extra parameter "
            "and keep the encoded text."),
        "observed_failure": (
            "error[E0049]: method `encode` has 1 type parameter but its trait "
            "declaration has 0 type parameters\n --> src/lib.rs:9:15\n  |\n"
            "9 |     fn encode<T>(&self) -> String {\n  |               ^ unexpected type parameter"),
        "files": {
            "src/lib.rs": code("""
                //! An encoder trait. The crate does not build: the impl declares a type
                //! parameter the trait does not have.

                pub trait Encode {
                    fn encode(&self) -> String;
                }

                pub struct Doc {
                    pub text: String,
                }

                impl Encode for Doc {
                    fn encode<T>(&self) -> String {
                        self.text.clone()
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use traits_14_extra_type_parameter::{Doc, Encode};

                #[test]
                fn encodes_the_text() {
                    let doc = Doc { text: "abc".to_string() };
                    assert_eq!(doc.encode(), "abc");
                }

                #[test]
                fn an_empty_document_encodes_to_empty() {
                    let doc = Doc { text: String::new() };
                    assert_eq!(doc.encode(), "");
                }

                #[test]
                fn the_generic_caller_works() {
                    fn render<E: Encode>(value: &E) -> String {
                        value.encode()
                    }
                    let doc = Doc { text: "x".to_string() };
                    assert_eq!(render(&doc), "x");
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! An encoder trait.

                pub trait Encode {
                    fn encode(&self) -> String;
                }

                pub struct Doc {
                    pub text: String,
                }

                impl Encode for Doc {
                    fn encode(&self) -> String {
                        self.text.clone()
                    }
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "self.text.clone()",
                     "replace": "self.text.to_uppercase()"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0049"]},
    },
    {
        "id": "traits-15-generic-trait-missing-argument",
        "category": "trait-impl",
        "difficulty": "medium",
        "capability": "Name the generic argument of a generic trait in its impl.",
        "prompt_intro": (
            "This crate does not compile. `Encode` takes a type parameter for the "
            "value it encodes, and the impl omits it. Implement `Encode<u8>` and "
            "keep the two-digit lowercase hexadecimal form."),
        "observed_failure": (
            "error[E0107]: missing generics for trait `Encode`\n --> src/lib.rs:7:12\n"
            "  |\n7 | impl Encode for Hex {\n  |      ^^^^^^ expected 1 generic argument"),
        "files": {
            "src/lib.rs": code("""
                //! A generic encoder. The crate does not build: the impl omits the
                //! trait's type argument.

                pub trait Encode<T> {
                    fn encode(&self, value: T) -> String;
                }

                pub struct Hex;

                impl Encode for Hex {
                    fn encode(&self, value: u8) -> String {
                        format!("{value:02x}")
                    }
                }

                pub fn encode_byte(value: u8) -> String {
                    use Encode;
                    Hex.encode(value)
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use traits_15_generic_trait_missing_argument::encode_byte;

                #[test]
                fn fifteen_is_lowercase_hex() {
                    assert_eq!(encode_byte(15), "0f");
                }

                #[test]
                fn zero_is_padded() {
                    assert_eq!(encode_byte(0), "00");
                }

                #[test]
                fn the_maximum_byte_is_two_digits() {
                    assert_eq!(encode_byte(255), "ff");
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! A generic encoder.

                pub trait Encode<T> {
                    fn encode(&self, value: T) -> String;
                }

                pub struct Hex;

                impl Encode<u8> for Hex {
                    fn encode(&self, value: u8) -> String {
                        format!("{value:02x}")
                    }
                }

                pub fn encode_byte(value: u8) -> String {
                    use Encode;
                    Hex.encode(value)
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": 'format!("{value:02x}")',
                     "replace": 'format!("{value:02X}")'},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0107"]},
    },
    {
        "id": "traits-16-method-arity",
        "category": "trait-impl",
        "difficulty": "easy",
        "capability": "Match the trait method's parameter list.",
        "prompt_intro": (
            "This crate does not compile. The `Scale` trait declares `scale` with a "
            "factor parameter, and the impl omits it. Accept the factor and return "
            "the value multiplied by it."),
        "observed_failure": (
            "error[E0050]: method `scale` has 1 parameter but the declaration in trait "
            "`Scale::scale` has 2\n --> src/lib.rs:5:33\n  |\n"
            "5 |     fn scale(&self) -> u64 { self.value }\n"
            "  |                                 ^^^^ expected 2 parameters, found 1"),
        "files": {
            "src/lib.rs": code("""
                //! A scaling trait. The crate does not build: the impl's parameter list
                //! is shorter than the trait's.

                pub trait Scale {
                    fn scale(&self, factor: u32) -> u64;
                }

                pub struct Meter {
                    pub value: u64,
                }

                impl Scale for Meter {
                    fn scale(&self) -> u64 {
                        self.value
                    }
                }

                pub fn scaled(meter: &Meter, factor: u32) -> u64 {
                    use Scale;
                    meter.scale(factor)
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use traits_16_method_arity::{scaled, Meter};

                #[test]
                fn a_factor_multiplies_the_value() {
                    assert_eq!(scaled(&Meter { value: 3 }, 4), 12);
                }

                #[test]
                fn a_zero_factor_is_zero() {
                    assert_eq!(scaled(&Meter { value: 3 }, 0), 0);
                }

                #[test]
                fn a_factor_of_one_is_the_identity() {
                    assert_eq!(scaled(&Meter { value: 9 }, 1), 9);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! A scaling trait.

                pub trait Scale {
                    fn scale(&self, factor: u32) -> u64;
                }

                pub struct Meter {
                    pub value: u64,
                }

                impl Scale for Meter {
                    fn scale(&self, factor: u32) -> u64 {
                        self.value * factor as u64
                    }
                }

                pub fn scaled(meter: &Meter, factor: u32) -> u64 {
                    use Scale;
                    meter.scale(factor)
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "self.value * factor as u64",
                     "replace": "self.value + factor as u64"},
        "naive": {"file": "src/lib.rs", "find": "self.value * factor as u64",
                  "replace": "self.value"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0050"]},
    },
    {
        "id": "traits-17-method-not-in-trait",
        "category": "trait-impl",
        "difficulty": "easy",
        "capability": "Put a method that the trait does not declare in an inherent impl.",
        "prompt_intro": (
            "This crate does not compile. The `Named` impl adds a `rename` method "
            "that the trait does not declare. Move it to an inherent impl for `Tag` "
            "and keep the behaviour."),
        "observed_failure": (
            "error[E0407]: method `rename` is not a member of trait `Named`\n"
            " --> src/lib.rs:10:5\n  |\n"
            "10 |     fn rename(&mut self, new: &str) { self.name = new.to_string(); }\n"
            "  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ not a member of trait `Named`"),
        "files": {
            "src/lib.rs": code("""
                //! A named tag. The crate does not build: the impl defines a method the
                //! trait does not declare.

                pub trait Named {
                    fn name(&self) -> &str;
                }

                pub struct Tag {
                    pub name: String,
                }

                impl Named for Tag {
                    fn name(&self) -> &str {
                        &self.name
                    }

                    fn rename(&mut self, new: &str) {
                        self.name = new.to_string();
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use traits_17_method_not_in_trait::{Named, Tag};

                #[test]
                fn the_tag_can_be_renamed() {
                    let mut tag = Tag { name: "old".to_string() };
                    tag.rename("new");
                    assert_eq!(tag.name(), "new");
                }

                #[test]
                fn the_trait_method_sees_the_new_name() {
                    let mut tag = Tag { name: "old".to_string() };
                    tag.rename("alpha");
                    assert_eq!(Named::name(&tag), "alpha");
                }

                #[test]
                fn an_empty_name_is_allowed() {
                    let mut tag = Tag { name: "old".to_string() };
                    tag.rename("");
                    assert_eq!(tag.name(), "");
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! A named tag.

                pub trait Named {
                    fn name(&self) -> &str;
                }

                pub struct Tag {
                    pub name: String,
                }

                impl Tag {
                    pub fn rename(&mut self, new: &str) {
                        self.name = new.to_string();
                    }
                }

                impl Named for Tag {
                    fn name(&self) -> &str {
                        &self.name
                    }
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "self.name = new.to_string();",
                     "replace": "self.name.push_str(new);"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0407"]},
    },
    {
        "id": "traits-18-duplicate-inherent-method",
        "category": "trait-impl",
        "difficulty": "easy",
        "capability": "Merge duplicate inherent impl blocks that define the same method.",
        "prompt_intro": (
            "This crate does not compile. Two inherent impl blocks for `Meter` both "
            "define `ratio`. Keep one definition that returns the value unchanged."),
        "observed_failure": (
            "error[E0592]: duplicate definitions with name `ratio`\n --> src/lib.rs:3:14\n"
            "  |\n3 | impl Meter { pub fn ratio(&self) -> u64 { self.value * 2 } }\n"
            "  |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ duplicate definitions for `ratio`"),
        "files": {
            "src/lib.rs": code("""
                //! A meter with two inherent impl blocks. The crate does not build: both
                //! blocks define `ratio`.

                pub struct Meter {
                    pub value: u64,
                }

                impl Meter {
                    pub fn ratio(&self) -> u64 {
                        self.value
                    }
                }

                impl Meter {
                    pub fn ratio(&self) -> u64 {
                        self.value * 2
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use traits_18_duplicate_inherent_method::Meter;

                #[test]
                fn the_ratio_is_the_value() {
                    assert_eq!(Meter { value: 5 }.ratio(), 5);
                }

                #[test]
                fn zero_is_zero() {
                    assert_eq!(Meter { value: 0 }.ratio(), 0);
                }

                #[test]
                fn the_method_is_available_on_references() {
                    let meter = Meter { value: 9 };
                    assert_eq!((&meter).ratio(), 9);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! A meter with one inherent impl block.

                pub struct Meter {
                    pub value: u64,
                }

                impl Meter {
                    pub fn ratio(&self) -> u64 {
                        self.value
                    }
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "self.value\n    }",
                     "replace": "self.value * 2\n    }"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0592"]},
    },
]

# ---------------------------------------------------------------------------
# repo-scale
#
# Every item spans three or four files; the repair must be propagated through
# the module boundary rather than patched at one call site.
# ---------------------------------------------------------------------------

REPO = [
    {
        "id": "repo-10-missing-module-file",
        "category": "repo-scale",
        "difficulty": "medium",
        "capability": "Restore a module file the rest of the crate depends on.",
        "prompt_intro": (
            "This crate does not compile. `lib.rs` declares `mod config`, and "
            "`service.rs` reads `Settings` from it, but `src/config.rs` is missing. "
            "Recreate it: `Settings` has `retries: u32` and `label: String`, and "
            "`Settings::default()` is `retries: 3`, `label: \"default\"`. Derive "
            "`Debug`, `Clone` and `PartialEq`."),
        "observed_failure": (
            "error[E0583]: file not found for module `config`\n --> src/lib.rs:4:1\n"
            "  |\n4 | pub mod config;\n  | ^^^^^^^^^^^^^^^\n"
            "  = help: to create the module `config`, create file \"src/config.rs\" or "
            "\"src/config/mod.rs\""),
        "files": {
            "src/lib.rs": code("""
                //! Settings and retries.

                pub mod config;
                pub mod service;

                pub use config::Settings;
                pub use service::retries;
            """),
            "src/service.rs": code("""
                //! Retry policy.

                use crate::config::Settings;

                pub fn retries() -> u32 {
                    Settings::default().retries
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use repo_10_missing_module_file::{retries, Settings};

                #[test]
                fn the_default_retry_count_is_three() {
                    assert_eq!(retries(), 3);
                }

                #[test]
                fn the_default_label_is_available() {
                    assert_eq!(Settings::default().label, "default");
                }

                #[test]
                fn settings_are_cloneable_and_comparable() {
                    let settings = Settings { retries: 5, label: "x".to_string() };
                    let copy = settings.clone();
                    assert_eq!(settings, copy);
                }
            """),
        },
        "reference": {
            "src/config.rs": code("""
                //! Settings.

                #[derive(Debug, Clone, PartialEq)]
                pub struct Settings {
                    pub retries: u32,
                    pub label: String,
                }

                impl Default for Settings {
                    fn default() -> Self {
                        Self { retries: 3, label: String::from("default") }
                    }
                }
            """),
        },
        "mutation": {"file": "src/config.rs", "find": "retries: 3",
                     "replace": "retries: 4"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0583"]},
    },
    {
        "id": "repo-11-private-field",
        "category": "repo-scale",
        "difficulty": "easy",
        "capability": "Use a constructor instead of constructing a struct with a private field.",
        "prompt_intro": (
            "This crate does not compile. `model::Config` keeps its `value` field "
            "private and offers a constructor; `build` constructs the struct "
            "directly from another module. Use the constructor so the module "
            "boundary is respected."),
        "observed_failure": (
            "error[E0451]: field `value` of struct `Config` is private\n"
            " --> src/build.rs:4:5\n  |\n4 |     Config { value: 2 }\n"
            "  |     ^^^^^^^^^^^^^^^^^^^^ field `value` is private"),
        "files": {
            "src/lib.rs": code("""
                //! Configuration and its builder.

                pub mod build;
                pub mod model;

                pub use build::make;
                pub use model::Config;
            """),
            "src/model.rs": code("""
                //! The configuration type.

                pub struct Config {
                    value: u8,
                }

                impl Config {
                    pub fn new(value: u8) -> Self {
                        Self { value }
                    }

                    pub fn value(&self) -> u8 {
                        self.value
                    }
                }
            """),
            "src/build.rs": code("""
                //! Build a configuration. This does not compile: it writes a private
                //! field from another module.

                use crate::model::Config;

                pub fn make() -> Config {
                    Config { value: 2 }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use repo_11_private_field::{make, Config};

                #[test]
                fn the_builder_uses_the_constructor() {
                    assert_eq!(make().value(), 2);
                }

                #[test]
                fn the_constructor_is_public() {
                    assert_eq!(Config::new(9).value(), 9);
                }

                #[test]
                fn the_field_stays_private() {
                    let config = make();
                    assert_eq!(config.value(), 2);
                }
            """),
        },
        "reference": {
            "src/build.rs": code("""
                //! Build a configuration.

                use crate::model::Config;

                pub fn make() -> Config {
                    Config::new(2)
                }
            """),
        },
        "mutation": {"file": "src/build.rs", "find": "Config::new(2)",
                     "replace": "Config::new(3)"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0451"]},
    },
    {
        "id": "repo-12-returned-local-reference",
        "category": "repo-scale",
        "difficulty": "medium",
        "capability": "Return owned data when the producing module owns the buffer.",
        "prompt_intro": (
            "This crate does not compile. `format::normalise` returns an owned "
            "`String`, and `cache::cached` tries to return a reference to that local "
            "value. Return an owned `String` from `cached` as well."),
        "observed_failure": (
            "error[E0515]: cannot return value referencing local variable `value`\n"
            " --> src/cache.rs:5:5\n  |\n4 |     let value = crate::format::normalise(text);\n"
            "  |         ----- `value` is borrowed here\n5 |     &value\n"
            "  |     ^^^^^^ returns a value referencing data owned by the current function"),
        "files": {
            "src/lib.rs": code("""
                //! Text formatting and its cache.

                pub mod cache;
                pub mod format;

                pub use cache::cached;
            """),
            "src/format.rs": code("""
                //! Text normalisation.

                pub fn normalise(text: &str) -> String {
                    text.trim().to_uppercase()
                }
            """),
            "src/cache.rs": code("""
                //! A cache over the formatter. This does not compile: it returns a
                //! reference to a local buffer.

                pub fn cached<'a>(text: &str) -> &'a str {
                    let value = crate::format::normalise(text);
                    &value
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use repo_12_returned_local_reference::cached;

                #[test]
                fn trims_and_uppercases() {
                    assert_eq!(cached(" ab "), "AB".to_string());
                }

                #[test]
                fn empty_input_stays_empty() {
                    assert_eq!(cached(""), String::new());
                }

                #[test]
                fn successive_calls_are_independent() {
                    let first = cached("x");
                    let second = cached("y");
                    assert_eq!(first, "X");
                    assert_eq!(second, "Y");
                }
            """),
        },
        "reference": {
            "src/cache.rs": code("""
                //! A cache over the formatter.

                pub fn cached(text: &str) -> String {
                    crate::format::normalise(text)
                }
            """),
        },
        "mutation": {"file": "src/cache.rs", "find": "crate::format::normalise(text)",
                     "replace": "text.to_string()"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0515"]},
    },
    {
        "id": "repo-13-undeclared-module-path",
        "category": "repo-scale",
        "difficulty": "easy",
        "capability": "Resolve a crate-local module path from a sibling module.",
        "prompt_intro": (
            "This crate does not compile. `api.rs` imports `util::slug`, which the "
            "compiler resolves as an external crate, but `util` is a sibling module "
            "of this crate. Use the crate-relative path."),
        "observed_failure": (
            "error[E0433]: failed to resolve: use of undeclared crate or module `util`\n"
            " --> src/api.rs:3:5\n  |\n3 | use util::slug;\n  |     ^^^^ use of "
            "undeclared crate or module `util`"),
        "files": {
            "src/lib.rs": code("""
                //! Slugs.

                mod util;
                pub mod api;

                pub use api::slugify;
            """),
            "src/util.rs": code("""
                //! Slug construction.

                pub fn slug(text: &str) -> String {
                    text.trim()
                        .to_lowercase()
                        .split_whitespace()
                        .collect::<Vec<_>>()
                        .join("-")
                }
            """),
            "src/api.rs": code("""
                //! The public slug API. This does not compile: the sibling module is
                //! named as if it were an external crate.

                pub fn slugify(text: &str) -> String {
                    util::slug(text)
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use repo_13_undeclared_module_path::slugify;

                #[test]
                fn words_are_joined_with_hyphens() {
                    assert_eq!(slugify("A B"), "a-b");
                }

                #[test]
                fn surrounding_space_is_trimmed() {
                    assert_eq!(slugify("  x  "), "x");
                }

                #[test]
                fn empty_input_is_empty() {
                    assert_eq!(slugify(""), "");
                }
            """),
        },
        "reference": {
            "src/api.rs": code("""
                //! The public slug API.

                use crate::util::slug;

                pub fn slugify(text: &str) -> String {
                    slug(text)
                }
            """),
        },
        "mutation": {"file": "src/api.rs", "find": "slug(text)",
                     "replace": "text.to_string()"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0433"]},
    },
    {
        "id": "repo-14-crossmodule-borrow",
        "category": "repo-scale",
        "difficulty": "medium",
        "capability": "Finish reading a collection before mutating it from another module.",
        "prompt_intro": (
            "This crate does not compile. `sync::append_all` holds a borrow of the "
            "first stored item while it pushes new items through the same `Store`. "
            "Take the measurement before mutating, and return `0` when the store is "
            "empty."),
        "observed_failure": (
            "error[E0502]: cannot borrow `store.items` as mutable because it is also "
            "borrowed as immutable\n --> src/sync.rs:5:19\n  |\n"
            "4 |     let first = &store.items[0];\n  |                  ----------- immutable "
            "borrow occurs here\n5 |     for item in extra {\n"
            "  |                   ^^^^^ mutable borrow occurs here"),
        "files": {
            "src/lib.rs": code("""
                //! A store and its synchroniser.

                pub mod store;
                pub mod sync;

                pub use store::Store;
                pub use sync::append_all;
            """),
            "src/store.rs": code("""
                //! The item store.

                pub struct Store {
                    pub items: Vec<String>,
                }

                impl Store {
                    pub fn new() -> Self {
                        Self { items: Vec::new() }
                    }
                }
            """),
            "src/sync.rs": code("""
                //! Append items to a store. This does not compile: a read borrow is held
                //! across the mutation.

                use crate::store::Store;

                pub fn append_all(store: &mut Store, extra: &[String]) -> usize {
                    let first = &store.items[0];
                    for item in extra {
                        store.items.push(item.clone());
                    }
                    first.len()
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use repo_14_crossmodule_borrow::{append_all, Store};

                fn store(items: &[&str]) -> Store {
                    let mut store = Store::new();
                    for item in items {
                        store.items.push(item.to_string());
                    }
                    store
                }

                #[test]
                fn returns_the_first_length_and_appends() {
                    let mut store = store(&["a", "bb"]);
                    assert_eq!(append_all(&mut store, &["c".to_string()]), 1);
                    assert_eq!(store.items.len(), 3);
                }

                #[test]
                fn an_empty_store_reports_zero() {
                    let mut store = Store::new();
                    assert_eq!(append_all(&mut store, &["x".to_string()]), 0);
                    assert_eq!(store.items.len(), 1);
                }

                #[test]
                fn appending_nothing_changes_nothing() {
                    let mut store = store(&["abc"]);
                    assert_eq!(append_all(&mut store, &[]), 3);
                    assert_eq!(store.items.len(), 1);
                }
            """),
        },
        "reference": {
            "src/sync.rs": code("""
                //! Append items to a store.

                use crate::store::Store;

                pub fn append_all(store: &mut Store, extra: &[String]) -> usize {
                    let first = store.items.first().map(|item| item.len()).unwrap_or(0);
                    for item in extra {
                        store.items.push(item.clone());
                    }
                    first
                }
            """),
        },
        "mutation": {"file": "src/sync.rs", "find": "unwrap_or(0)",
                     "replace": "unwrap_or(1)"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0502"]},
    },
    {
        "id": "repo-15-caller-not-updated",
        "category": "repo-scale",
        "difficulty": "easy",
        "capability": "Update a caller after a helper's signature gained a parameter.",
        "prompt_intro": (
            "This crate does not compile. `format::label` now takes a width, but "
            "`report::line` still calls it with one argument. Pass the registered "
            "width of 6. Text shorter than the width is padded on the right; longer "
            "text is not truncated."),
        "observed_failure": (
            "error[E0061]: this function takes 2 arguments but 1 argument was supplied\n"
            " --> src/report.rs:4:5\n  |\n4 |     crate::format::label(text)\n"
            "  |     ^^^^^^^^^^^^^^^^^^^^ ---- argument #1 of type `&str` is supplied\n"
            "  |\n  = note: expected parameter `&str`, `usize`"),
        "files": {
            "src/lib.rs": code("""
                //! Labels and their reports.

                pub mod format;
                pub mod report;

                pub use report::line;
            """),
            "src/format.rs": code("""
                //! Label formatting.

                pub fn label(text: &str, width: usize) -> String {
                    let mut out = text.to_string();
                    while out.len() < width {
                        out.push(' ');
                    }
                    out
                }
            """),
            "src/report.rs": code("""
                //! A report line. This does not compile: the caller was not updated
                //! after the formatter gained a width parameter.

                pub fn line(text: &str) -> String {
                    crate::format::label(text)
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use repo_15_caller_not_updated::line;

                #[test]
                fn short_text_is_padded_to_six() {
                    assert_eq!(line("ab"), "ab    ");
                    assert_eq!(line("ab").len(), 6);
                }

                #[test]
                fn long_text_is_not_truncated() {
                    assert_eq!(line("abcdefg"), "abcdefg");
                }

                #[test]
                fn empty_text_becomes_six_spaces() {
                    assert_eq!(line(""), "      ");
                }
            """),
        },
        "reference": {
            "src/report.rs": code("""
                //! A report line.

                pub fn line(text: &str) -> String {
                    crate::format::label(text, 6)
                }
            """),
        },
        "mutation": {"file": "src/report.rs", "find": "label(text, 6)",
                     "replace": "label(text, 5)"},
        "naive": {"file": "src/report.rs", "find": "crate::format::label(text, 6)",
                  "replace": "crate::format::label(text, text.len())"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0061"]},
    },
    {
        "id": "repo-16-missing-associated-item",
        "category": "repo-scale",
        "difficulty": "medium",
        "capability": "Complete a trait implementation split across modules.",
        "prompt_intro": (
            "This crate does not compile. The `Sink` trait declared in `traits.rs` "
            "requires `accept` and `finish`; the `Memory` impl in `memory.rs` only "
            "provides `accept`. Add `finish` and keep the error type."),
        "observed_failure": (
            "error[E0046]: not all trait items implemented, missing: `finish`\n"
            " --> src/memory.rs:5:1\n  |\n5 | impl Sink for Memory {\n"
            "  | ^^^^^^^^^^^^^^^^^^^^ missing `finish` in implementation"),
        "files": {
            "src/lib.rs": code("""
                //! A sink and its in-memory implementation.

                pub mod memory;
                pub mod traits;

                pub use memory::Memory;
                pub use traits::Sink;
            """),
            "src/traits.rs": code("""
                //! The sink trait.

                pub trait Sink {
                    type Error;
                    fn accept(&mut self, value: u8) -> Result<(), Self::Error>;
                    fn finish(&mut self) -> Result<(), Self::Error>;
                }
            """),
            "src/memory.rs": code("""
                //! An in-memory sink. This does not compile: the impl omits `finish`.

                use crate::traits::Sink;

                #[derive(Default)]
                pub struct Memory {
                    pub bytes: Vec<u8>,
                }

                impl Sink for Memory {
                    type Error = ();

                    fn accept(&mut self, value: u8) -> Result<(), Self::Error> {
                        self.bytes.push(value);
                        Ok(())
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use repo_16_missing_associated_item::{Memory, Sink};

                #[test]
                fn accepted_bytes_are_recorded() {
                    let mut memory = Memory::default();
                    memory.accept(1).unwrap();
                    memory.accept(2).unwrap();
                    memory.finish().unwrap();
                    assert_eq!(memory.bytes, vec![1, 2]);
                }

                #[test]
                fn finish_on_an_empty_sink_succeeds() {
                    let mut memory = Memory::default();
                    assert!(memory.finish().is_ok());
                    assert!(memory.bytes.is_empty());
                }

                #[test]
                fn the_error_type_is_the_unit_type() {
                    let mut memory = Memory::default();
                    let result: Result<(), ()> = memory.finish();
                    assert!(result.is_ok());
                }
            """),
        },
        "reference": {
            "src/memory.rs": code("""
                //! An in-memory sink.

                use crate::traits::Sink;

                #[derive(Default)]
                pub struct Memory {
                    pub bytes: Vec<u8>,
                }

                impl Sink for Memory {
                    type Error = ();

                    fn accept(&mut self, value: u8) -> Result<(), Self::Error> {
                        self.bytes.push(value);
                        Ok(())
                    }

                    fn finish(&mut self) -> Result<(), Self::Error> {
                        Ok(())
                    }
                }
            """),
        },
        "mutation": {"file": "src/memory.rs", "find": "Ok(())",
                     "replace": "Err(())"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0046"]},
    },
    {
        "id": "repo-17-opaque-return-method",
        "category": "repo-scale",
        "difficulty": "medium",
        "capability": "Call a trait method on an opaque return type instead of an absent one.",
        "prompt_intro": (
            "This crate does not compile. `api::open` returns `impl Store`, so only "
            "the trait's methods are visible; `peek` calls `first`, which is not one "
            "of them. Return the value stored under key `0`."),
        "observed_failure": (
            "error[E0599]: no method named `first` found for opaque type `impl Store` "
            "in the current scope\n --> src/api.rs:6:36\n  |\n"
            "6 | pub fn peek() -> Option<u8> { open().first() }\n"
            "  |                                    ^^^^^ method not found in `impl Store`"),
        "files": {
            "src/lib.rs": code("""
                //! A keyed store and its API.

                pub mod api;
                pub mod store;

                pub use api::{open, peek};
                pub use store::Store;
            """),
            "src/store.rs": code("""
                //! The keyed store.

                pub trait Store {
                    fn get(&self, key: u8) -> Option<u8>;
                }

                pub struct Map {
                    pub value: u8,
                }

                impl Store for Map {
                    fn get(&self, key: u8) -> Option<u8> {
                        if key == 0 {
                            Some(self.value)
                        } else {
                            None
                        }
                    }
                }
            """),
            "src/api.rs": code("""
                //! The public store API. This does not compile: `first` is not a trait
                //! method, so it is not visible on the opaque return type.

                use crate::store::{Map, Store};

                pub fn open() -> impl Store {
                    Map { value: 7 }
                }

                pub fn peek() -> Option<u8> {
                    open().first()
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use repo_17_opaque_return_method::{open, peek, Store};

                #[test]
                fn peek_returns_the_value_at_key_zero() {
                    assert_eq!(peek(), Some(7));
                }

                #[test]
                fn the_opaque_value_exposes_the_trait_method() {
                    assert_eq!(open().get(0), Some(7));
                }

                #[test]
                fn other_keys_are_absent() {
                    assert_eq!(open().get(1), None);
                }
            """),
        },
        "reference": {
            "src/api.rs": code("""
                //! The public store API.

                use crate::store::{Map, Store};

                pub fn open() -> impl Store {
                    Map { value: 7 }
                }

                pub fn peek() -> Option<u8> {
                    open().get(0)
                }
            """),
        },
        "mutation": {"file": "src/api.rs", "find": "open().get(0)",
                     "replace": "open().get(1)"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0599"]},
    },
    {
        "id": "repo-18-moved-across-modules",
        "category": "repo-scale",
        "difficulty": "easy",
        "capability": "Measure a value before moving it across a module boundary.",
        "prompt_intro": (
            "This crate does not compile. `flow::register_and_count` moves the "
            "registry into a snapshot binding and then asks it for its count. Count "
            "before moving, and return that count."),
        "observed_failure": (
            "error[E0382]: borrow of moved value: `registry`\n --> src/flow.rs:10:17\n"
            "  |\n8 |     let snapshot = registry;\n  |         -------- value moved here\n"
            "10 |     let count = registry.count();\n"
            "  |                 ^^^^^^^^ value borrowed here after move"),
        "files": {
            "src/lib.rs": code("""
                //! A registry and its registration flow.

                pub mod flow;
                pub mod registry;

                pub use flow::register_and_count;
                pub use registry::Registry;
            """),
            "src/registry.rs": code("""
                //! The name registry.

                #[derive(Default)]
                pub struct Registry {
                    entries: Vec<String>,
                }

                impl Registry {
                    pub fn new() -> Self {
                        Self::default()
                    }

                    pub fn add(&mut self, name: &str) {
                        self.entries.push(name.to_string());
                    }

                    pub fn count(&self) -> usize {
                        self.entries.len()
                    }
                }
            """),
            "src/flow.rs": code("""
                //! The registration flow. This does not compile: the registry is used
                //! after it is moved.

                use crate::registry::Registry;

                pub fn register_and_count(mut registry: Registry, names: &[&str]) -> usize {
                    for name in names {
                        registry.add(name);
                    }
                    let snapshot = registry;
                    let count = registry.count();
                    count + snapshot.count()
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use repo_18_moved_across_modules::{register_and_count, Registry};

                #[test]
                fn two_names_are_counted() {
                    assert_eq!(register_and_count(Registry::new(), &["a", "b"]), 2);
                }

                #[test]
                fn no_names_are_counted() {
                    assert_eq!(register_and_count(Registry::new(), &[]), 0);
                }

                #[test]
                fn a_single_name_is_counted() {
                    assert_eq!(register_and_count(Registry::new(), &["a"]), 1);
                }
            """),
        },
        "reference": {
            "src/flow.rs": code("""
                //! The registration flow.

                use crate::registry::Registry;

                pub fn register_and_count(mut registry: Registry, names: &[&str]) -> usize {
                    for name in names {
                        registry.add(name);
                    }
                    registry.count()
                }
            """),
        },
        "mutation": {"file": "src/flow.rs", "find": "registry.count()",
                     "replace": "registry.count() + 1"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0382"]},
    },
]

# ---------------------------------------------------------------------------
# semantic-repair
#
# The initial state compiles and its visible test passes; only the hidden tests
# expose the behavioural defect.
# ---------------------------------------------------------------------------

SEMANTIC = [
    {
        "id": "semantic-10-round-half-even",
        "category": "semantic-repair",
        "difficulty": "medium",
        "capability": "Round ties to the nearest even integer rather than away from zero.",
        "prompt_intro": (
            "The helper builds and its visible test passes, but the contract rounds "
            "ties to the nearest **even** integer: 2.5 becomes 2 and 3.5 becomes 4 "
            "(and -2.5 becomes -2). Non-ties round to the nearest integer. Fix the "
            "helper; keep the signature."),
        "observed_failure": (
            "the crate builds and tests pass; round_half_even(2.5) is 3 where the "
            "contract requires 2"),
        "files": {
            "src/lib.rs": code("""
                //! Tie-to-even rounding with the standard rounding rule.

                pub fn round_half_even(value: f64) -> f64 {
                    value.round()
                }

                #[cfg(test)]
                mod tests {
                    use super::round_half_even;

                    #[test]
                    fn visible_not_a_tie() {
                        assert_eq!(round_half_even(2.4), 2.0);
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use semantic_10_round_half_even::round_half_even;

                #[test]
                fn positive_ties_go_to_even() {
                    assert_eq!(round_half_even(2.5), 2.0);
                    assert_eq!(round_half_even(3.5), 4.0);
                }

                #[test]
                fn negative_ties_go_to_even() {
                    assert_eq!(round_half_even(-2.5), -2.0);
                    assert_eq!(round_half_even(-3.5), -4.0);
                }

                #[test]
                fn non_ties_round_normally() {
                    assert_eq!(round_half_even(2.4), 2.0);
                    assert_eq!(round_half_even(2.6), 3.0);
                    assert_eq!(round_half_even(-2.6), -3.0);
                }

                #[test]
                fn integers_are_unchanged() {
                    assert_eq!(round_half_even(7.0), 7.0);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Tie-to-even rounding.

                pub fn round_half_even(value: f64) -> f64 {
                    let floor = value.floor();
                    let fraction = value - floor;
                    if (fraction - 0.5).abs() < 1e-12 {
                        if (floor as i64) % 2 == 0 {
                            floor
                        } else {
                            floor + 1.0
                        }
                    } else {
                        value.round()
                    }
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "(floor as i64) % 2 == 0",
                     "replace": "(floor as i64) % 2 != 0"},
        "initial_failure": {"kind": "hidden_test_failure",
                            "must_contain": ["test result: FAILED"]},
    },
    {
        "id": "semantic-11-leading-zero-id",
        "category": "semantic-repair",
        "difficulty": "easy",
        "capability": "Reject a numeric identifier that carries a leading zero.",
        "prompt_intro": (
            "The helper builds and its visible test passes, but identifiers with a "
            "leading zero are not canonical and must be rejected: `parse_id(\"012\")` "
            "is `None`, while `parse_id(\"0\")` and `parse_id(\"12\")` parse. Empty "
            "input and non-digits are `None`. Fix the helper; keep the signature."),
        "observed_failure": (
            "the crate builds and tests pass; parse_id(\"012\") is Some(12) where the "
            "contract requires None"),
        "files": {
            "src/lib.rs": code("""
                //! Identifier parsing with the standard integer parser.

                pub fn parse_id(text: &str) -> Option<u32> {
                    text.parse().ok()
                }

                #[cfg(test)]
                mod tests {
                    use super::parse_id;

                    #[test]
                    fn visible_plain_number() {
                        assert_eq!(parse_id("12"), Some(12));
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use semantic_11_leading_zero_id::parse_id;

                #[test]
                fn a_leading_zero_is_rejected() {
                    assert_eq!(parse_id("012"), None);
                    assert_eq!(parse_id("0012"), None);
                }

                #[test]
                fn zero_itself_is_allowed() {
                    assert_eq!(parse_id("0"), Some(0));
                }

                #[test]
                fn plain_numbers_parse() {
                    assert_eq!(parse_id("12"), Some(12));
                }

                #[test]
                fn empty_and_non_digits_are_rejected() {
                    assert_eq!(parse_id(""), None);
                    assert_eq!(parse_id("12a"), None);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Canonical identifier parsing.

                pub fn parse_id(text: &str) -> Option<u32> {
                    if text.len() > 1 && text.starts_with('0') {
                        return None;
                    }
                    text.parse().ok()
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "text.len() > 1 && text.starts_with('0')",
                     "replace": "text.len() > 3 && text.starts_with('0')"},
        "initial_failure": {"kind": "hidden_test_failure",
                            "must_contain": ["test result: FAILED"]},
    },
    {
        "id": "semantic-12-split-once",
        "category": "semantic-repair",
        "difficulty": "easy",
        "capability": "Split a key from a value at the first separator only.",
        "prompt_intro": (
            "The helper builds and its visible test passes, but the value may itself "
            "contain the separator: `split_key(\"a:b:c\")` is `(\"a\", \"b:c\")`. Split "
            "at the first `:` only. Empty key or value parts are allowed; a string "
            "with no separator is `None`."),
        "observed_failure": (
            "the crate builds and tests pass; split_key(\"a:b:c\") is (\"a\", \"b\") "
            "where the contract requires (\"a\", \"b:c\")"),
        "files": {
            "src/lib.rs": code("""
                //! Key/value splitting at the first two components.

                pub fn split_key(text: &str) -> Option<(String, String)> {
                    let mut parts = text.split(':');
                    let key = parts.next()?;
                    let value = parts.next()?;
                    Some((key.to_string(), value.to_string()))
                }

                #[cfg(test)]
                mod tests {
                    use super::split_key;

                    #[test]
                    fn visible_single_separator() {
                        assert_eq!(split_key("a:b"), Some(("a".to_string(), "b".to_string())));
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use semantic_12_split_once::split_key;

                #[test]
                fn the_value_keeps_later_separators() {
                    assert_eq!(split_key("a:b:c"),
                               Some(("a".to_string(), "b:c".to_string())));
                }

                #[test]
                fn empty_parts_are_allowed() {
                    assert_eq!(split_key("a:"), Some(("a".to_string(), String::new())));
                    assert_eq!(split_key(":b"), Some((String::new(), "b".to_string())));
                }

                #[test]
                fn no_separator_is_none() {
                    assert_eq!(split_key("abc"), None);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Key/value splitting at the first separator.

                pub fn split_key(text: &str) -> Option<(String, String)> {
                    text.split_once(':')
                        .map(|(key, value)| (key.to_string(), value.to_string()))
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "split_once(':')",
                     "replace": "split_once(';')"},
        "initial_failure": {"kind": "hidden_test_failure",
                            "must_contain": ["test result: FAILED"]},
    },
    {
        "id": "semantic-13-inverted-clamp",
        "category": "semantic-repair",
        "difficulty": "easy",
        "capability": "Honour the documented order when a clamp's bounds are inverted.",
        "prompt_intro": (
            "The helper builds and its visible test passes, but the contract says an "
            "inverted bound pair returns the low bound: `clamp(5, 10, 0)` is 10. With "
            "a well-ordered pair the normal clamp applies. Fix the helper; keep the "
            "signature."),
        "observed_failure": (
            "the crate builds and tests pass; clamp(5, 10, 0) is 0 where the contract "
            "requires 10"),
        "files": {
            "src/lib.rs": code("""
                //! Clamping with the standard two-step form.

                pub fn clamp(value: i64, low: i64, high: i64) -> i64 {
                    value.max(low).min(high)
                }

                #[cfg(test)]
                mod tests {
                    use super::clamp;

                    #[test]
                    fn visible_inside_the_bounds() {
                        assert_eq!(clamp(5, 0, 10), 5);
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use semantic_13_inverted_clamp::clamp;

                #[test]
                fn inverted_bounds_return_the_low_bound() {
                    assert_eq!(clamp(5, 10, 0), 10);
                    assert_eq!(clamp(-1, 3, 1), 3);
                }

                #[test]
                fn a_value_below_the_bounds_is_lifted() {
                    assert_eq!(clamp(-5, 0, 10), 0);
                }

                #[test]
                fn a_value_above_the_bounds_is_capped() {
                    assert_eq!(clamp(15, 0, 10), 10);
                }

                #[test]
                fn a_value_inside_is_unchanged() {
                    assert_eq!(clamp(5, 0, 10), 5);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Clamping with an inverted-bound rule.

                pub fn clamp(value: i64, low: i64, high: i64) -> i64 {
                    if low > high {
                        return low;
                    }
                    value.max(low).min(high)
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "return low;", "replace": "return high;"},
        "initial_failure": {"kind": "hidden_test_failure",
                            "must_contain": ["test result: FAILED"]},
    },
    {
        "id": "semantic-14-url-encode-plus",
        "category": "semantic-repair",
        "difficulty": "medium",
        "capability": "Encode a literal plus separately from a space in a query value.",
        "prompt_intro": (
            "The helper builds and its visible test passes, but a literal `+` in the "
            "input is data and must be percent-encoded: `encode(\"a+b\")` is "
            "`a%2Bb`, while a space becomes `+`. Fix the helper; keep the signature."),
        "observed_failure": (
            "the crate builds and tests pass; encode(\"a+b\") is \"a+b\" where the "
            "contract requires \"a%2Bb\""),
        "files": {
            "src/lib.rs": code("""
                //! Query-value encoding with the standard space replacement.

                pub fn encode(text: &str) -> String {
                    text.replace(' ', "+")
                }

                #[cfg(test)]
                mod tests {
                    use super::encode;

                    #[test]
                    fn visible_space_becomes_plus() {
                        assert_eq!(encode("a b"), "a+b");
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use semantic_14_url_encode_plus::encode;

                #[test]
                fn a_literal_plus_is_encoded() {
                    assert_eq!(encode("a+b"), "a%2Bb");
                }

                #[test]
                fn spaces_and_pluses_together() {
                    assert_eq!(encode("a b+c"), "a+b%2Bc");
                }

                #[test]
                fn plain_text_is_unchanged() {
                    assert_eq!(encode("abc"), "abc");
                }

                #[test]
                fn empty_input_is_empty() {
                    assert_eq!(encode(""), "");
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Query-value encoding.

                pub fn encode(text: &str) -> String {
                    text.replace('+', "%2B").replace(' ', "+")
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": 'text.replace(\'+\', "%2B")',
                     "replace": 'text.replace(\'+\', "%2b")'},
        "initial_failure": {"kind": "hidden_test_failure",
                            "must_contain": ["test result: FAILED"]},
    },
    {
        "id": "semantic-15-shorten-with-ellipsis",
        "category": "semantic-repair",
        "difficulty": "medium",
        "capability": "Include the ellipsis inside the requested character budget.",
        "prompt_intro": (
            "The helper builds and its visible test passes, but the result must be at "
            "most `limit` characters **including** the ellipsis: `shorten(\"abcdef\", 4)` "
            "is `abc…`. A limit of zero is the empty string. Inputs that already fit "
            "are returned unchanged."),
        "observed_failure": (
            "the crate builds and tests pass; shorten(\"abcdef\", 4) is \"abcd…\" (5 "
            "characters) where the contract requires \"abc…\""),
        "files": {
            "src/lib.rs": code("""
                //! Shortening with a trailing ellipsis.

                pub fn shorten(text: &str, limit: usize) -> String {
                    let characters: Vec<char> = text.chars().collect();
                    if characters.len() <= limit {
                        return text.to_string();
                    }
                    let mut out: String = characters.iter().take(limit).collect();
                    out.push('…');
                    out
                }

                #[cfg(test)]
                mod tests {
                    use super::shorten;

                    #[test]
                    fn visible_exact_fit() {
                        assert_eq!(shorten("abc", 3), "abc");
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use semantic_15_shorten_with_ellipsis::shorten;

                #[test]
                fn the_result_fits_the_budget() {
                    let short = shorten("abcdef", 4);
                    assert_eq!(short, "abc…");
                    assert_eq!(short.chars().count(), 4);
                }

                #[test]
                fn a_budget_of_one_is_just_the_ellipsis() {
                    assert_eq!(shorten("abcdef", 1), "…");
                }

                #[test]
                fn a_zero_budget_is_empty() {
                    assert_eq!(shorten("abcdef", 0), "");
                }

                #[test]
                fn text_that_fits_is_unchanged() {
                    assert_eq!(shorten("abc", 3), "abc");
                    assert_eq!(shorten("", 5), "");
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Shortening with a trailing ellipsis inside the budget.

                pub fn shorten(text: &str, limit: usize) -> String {
                    let characters: Vec<char> = text.chars().collect();
                    if characters.len() <= limit {
                        return text.to_string();
                    }
                    if limit == 0 {
                        return String::new();
                    }
                    let mut out: String = characters.iter().take(limit - 1).collect();
                    out.push('…');
                    out
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "take(limit - 1)",
                     "replace": "take(limit)"},
        "initial_failure": {"kind": "hidden_test_failure",
                            "must_contain": ["test result: FAILED"]},
    },
    {
        "id": "semantic-16-numeric-string-sort",
        "category": "semantic-repair",
        "difficulty": "easy",
        "capability": "Sort numbers numerically rather than by their decimal text.",
        "prompt_intro": (
            "The helper builds and its visible test passes, but the contract sorts "
            "numbers by value: `sorted(&[10, 9, 100])` is `[9, 10, 100]`. Fix the "
            "helper; keep the signature."),
        "observed_failure": (
            "the crate builds and tests pass; sorted(&[10, 9, 100]) is [10, 100, 9] "
            "where the contract requires [9, 10, 100]"),
        "files": {
            "src/lib.rs": code("""
                //! Sorting through decimal text.

                pub fn sorted(values: &[u32]) -> Vec<u32> {
                    let mut text: Vec<String> =
                        values.iter().map(|value| value.to_string()).collect();
                    text.sort();
                    text.iter().filter_map(|value| value.parse().ok()).collect()
                }

                #[cfg(test)]
                mod tests {
                    use super::sorted;

                    #[test]
                    fn visible_single_digits() {
                        assert_eq!(sorted(&[3, 1, 2]), vec![1, 2, 3]);
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use semantic_16_numeric_string_sort::sorted;

                #[test]
                fn wider_numbers_sort_by_value() {
                    assert_eq!(sorted(&[10, 9, 100]), vec![9, 10, 100]);
                }

                #[test]
                fn single_digits_sort_by_value() {
                    assert_eq!(sorted(&[3, 1, 2]), vec![1, 2, 3]);
                }

                #[test]
                fn duplicates_are_kept() {
                    assert_eq!(sorted(&[2, 2, 10]), vec![2, 2, 10]);
                }

                #[test]
                fn empty_input_is_empty() {
                    assert!(sorted(&[]).is_empty());
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Numeric sorting.

                pub fn sorted(values: &[u32]) -> Vec<u32> {
                    let mut out = values.to_vec();
                    out.sort_unstable();
                    out
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "out.sort_unstable();",
                     "replace": "out.sort_unstable_by(|a, b| b.cmp(a));"},
        "initial_failure": {"kind": "hidden_test_failure",
                            "must_contain": ["test result: FAILED"]},
    },
    {
        "id": "semantic-17-adjacent-dedup",
        "category": "semantic-repair",
        "difficulty": "easy",
        "capability": "Remove only adjacent duplicate values, preserving later repeats.",
        "prompt_intro": (
            "The helper builds and its visible test passes, but the contract removes "
            "only **adjacent** duplicates: `collapse(&[1, 2, 1])` is `[1, 2, 1]`. Fix "
            "the helper; keep the signature."),
        "observed_failure": (
            "the crate builds and tests pass; collapse(&[1, 2, 1]) is [1, 2] where "
            "the contract requires [1, 2, 1]"),
        "files": {
            "src/lib.rs": code("""
                //! Collapsing through a set of seen values.

                use std::collections::BTreeSet;

                pub fn collapse(values: &[u8]) -> Vec<u8> {
                    let mut seen = BTreeSet::new();
                    values
                        .iter()
                        .copied()
                        .filter(|value| seen.insert(*value))
                        .collect()
                }

                #[cfg(test)]
                mod tests {
                    use super::collapse;

                    #[test]
                    fn visible_adjacent_duplicates() {
                        assert_eq!(collapse(&[1, 1, 2]), vec![1, 2]);
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use semantic_17_adjacent_dedup::collapse;

                #[test]
                fn a_later_repeat_is_kept() {
                    assert_eq!(collapse(&[1, 2, 1]), vec![1, 2, 1]);
                }

                #[test]
                fn adjacent_duplicates_are_removed() {
                    assert_eq!(collapse(&[1, 1, 2]), vec![1, 2]);
                }

                #[test]
                fn long_runs_collapse_to_one() {
                    assert_eq!(collapse(&[3, 3, 3, 3]), vec![3]);
                }

                #[test]
                fn empty_input_is_empty() {
                    assert!(collapse(&[]).is_empty());
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Collapsing adjacent duplicates.

                pub fn collapse(values: &[u8]) -> Vec<u8> {
                    let mut out: Vec<u8> = Vec::new();
                    for value in values {
                        if out.last() != Some(value) {
                            out.push(*value);
                        }
                    }
                    out
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "out.last() != Some(value)",
                     "replace": "out.last() == Some(value)"},
        "initial_failure": {"kind": "hidden_test_failure",
                            "must_contain": ["test result: FAILED"]},
    },
    {
        "id": "semantic-18-email-domain-case",
        "category": "semantic-repair",
        "difficulty": "easy",
        "capability": "Lowercase only the domain of an address, preserving the local part.",
        "prompt_intro": (
            "The helper builds and its visible test passes, but the contract "
            "lowercases only the domain of an address: `normalise(\"User@Example.COM\")` "
            "is `User@example.com`. The local part keeps its case. Input without `@` "
            "is returned unchanged."),
        "observed_failure": (
            "the crate builds and tests pass; normalise(\"User@Example.COM\") is "
            "\"user@example.com\" where the contract requires \"User@example.com\""),
        "files": {
            "src/lib.rs": code("""
                //! Address normalisation with whole-string lowercasing.

                pub fn normalise(text: &str) -> String {
                    text.to_lowercase()
                }

                #[cfg(test)]
                mod tests {
                    use super::normalise;

                    #[test]
                    fn visible_already_lowercase() {
                        assert_eq!(normalise("user@example.com"), "user@example.com");
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use semantic_18_email_domain_case::normalise;

                #[test]
                fn the_local_part_keeps_its_case() {
                    assert_eq!(normalise("User@Example.COM"), "User@example.com");
                }

                #[test]
                fn an_already_normal_address_is_unchanged() {
                    assert_eq!(normalise("user@example.com"), "user@example.com");
                }

                #[test]
                fn text_without_a_domain_is_unchanged() {
                    assert_eq!(normalise("NoAtSign"), "NoAtSign");
                }

                #[test]
                fn empty_input_is_empty() {
                    assert_eq!(normalise(""), "");
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Address normalisation: only the domain is lowercased.

                pub fn normalise(text: &str) -> String {
                    match text.split_once('@') {
                        Some((local, domain)) => {
                            format!("{}@{}", local, domain.to_lowercase())
                        }
                        None => text.to_string(),
                    }
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "domain.to_lowercase()",
                     "replace": "domain.to_uppercase()"},
        "naive": {"file": "src/lib.rs", "find": "domain.to_lowercase()",
                  "replace": "domain.to_string()"},
        "initial_failure": {"kind": "hidden_test_failure",
                            "must_contain": ["test result: FAILED"]},
    },
]


TASKS = (LIFETIME + FAILING + MIGRATION + DEPENDENCY + CONCURRENCY
         + TRAITS + REPO + SEMANTIC)
