"""The sealed holdback (P14.7): 24 tasks, three per capability category.

These items are authored by the same process as the scored suite and are
**frozen** the moment their registration is written: the registration records
the item hashes and the sealed aggregate, and the harness refuses to score them
unless the unlock flag and the registered hash are supplied together
(`oracle/capability/sealed.py`).

They are never materialised under `fixtures/capability/tasks/`; they live in
`fixtures/capability/sealed/` so an ordinary run against the scored suite cannot
see them. Their diagnostics were observed by compiling each initial state; the
codes below are measured, not inferred.
"""

from .common import code

BORROW = [
    {
        "id": "sealed-borrow-01-sort-closure-mutation",
        "category": "lifetime-borrow",
        "difficulty": "medium",
        "capability": "Sort without mutating the collection inside the comparator.",
        "prompt_intro": (
            "This crate does not compile. The comparator passed to `sort_by` pushes "
            "into the vector being sorted, so the sort holds a mutable borrow while "
            "the closure asks for another. Sort the vector and return the number of "
            "adjacent duplicates (values equal to their predecessor)."),
        "observed_failure": (
            "error[E0500]: closure requires unique access to `*values` but it is already "
            "borrowed\n --> src/lib.rs:5:20\n  |\n"
            "4 |     values.sort_by(|left, right| {\n"
            "  |     ------ ------- first borrow later used by call\n"
            "5 |         values.push(*left);\n"
            "  |         ------ second borrow occurs due to use of `*values` in closure"),
        "files": {
            "src/lib.rs": code("""
                //! Duplicate counting before a sort. The crate does not build: the
                //! comparator mutates the vector being sorted.

                pub fn sort_and_count_duplicates(values: &mut Vec<i64>) -> usize {
                    values.sort_by(|left, right| {
                        values.push(*left);
                        left.cmp(right)
                    });
                    values.windows(2).filter(|pair| pair[0] == pair[1]).count()
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use sealed_borrow_01_sort_closure_mutation::sort_and_count_duplicates;

                #[test]
                fn counts_duplicates_and_sorts() {
                    let mut values = vec![4, 1, 4, 2, 4];
                    assert_eq!(sort_and_count_duplicates(&mut values), 2);
                    assert_eq!(values, vec![1, 2, 4, 4, 4]);
                }

                #[test]
                fn a_distinct_vector_has_no_duplicates() {
                    let mut values = vec![5, 2, 9];
                    assert_eq!(sort_and_count_duplicates(&mut values), 0);
                    assert_eq!(values, vec![2, 5, 9]);
                }

                #[test]
                fn an_empty_vector_is_unchanged() {
                    let mut values: Vec<i64> = Vec::new();
                    assert_eq!(sort_and_count_duplicates(&mut values), 0);
                    assert!(values.is_empty());
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Duplicate counting before a sort.

                pub fn sort_and_count_duplicates(values: &mut Vec<i64>) -> usize {
                    values.sort();
                    values.windows(2).filter(|pair| pair[0] == pair[1]).count()
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "values.sort();",
                     "replace": "values.sort_by(|a, b| b.cmp(a));"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0500"]},
    },
    {
        "id": "sealed-borrow-02-swap-two-elements",
        "category": "lifetime-borrow",
        "difficulty": "easy",
        "capability": "Swap two elements without holding two mutable references.",
        "prompt_intro": (
            "This crate does not compile. `swap_ends` takes a mutable reference to "
            "the first element and a second mutable reference to the last element at "
            "the same time. Swap them without overlapping borrows. An empty or "
            "single-element slice is unchanged."),
        "observed_failure": (
            "error[E0499]: cannot borrow `*values` as mutable more than once at a time\n"
            " --> src/lib.rs:6:16\n  |\n5 |     let first = &mut values[0];\n"
            "  |                      -------- first mutable borrow occurs here\n"
            "6 |     let last = &mut values[values.len() - 1];\n"
            "  |                 ^^^^^^ second mutable borrow occurs here"),
        "files": {
            "src/lib.rs": code("""
                //! Swap the ends of a vector. The crate does not build: both ends are
                //! borrowed mutably at once.

                pub fn swap_ends(values: &mut Vec<i64>) {
                    let first = &mut values[0];
                    let last = &mut values[values.len() - 1];
                    std::mem::swap(first, last);
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use sealed_borrow_02_swap_two_elements::swap_ends;

                #[test]
                fn swaps_the_ends() {
                    let mut values = vec![1, 2, 3, 4];
                    swap_ends(&mut values);
                    assert_eq!(values, vec![4, 2, 3, 1]);
                }

                #[test]
                fn a_single_element_is_unchanged() {
                    let mut values = vec![7];
                    swap_ends(&mut values);
                    assert_eq!(values, vec![7]);
                }

                #[test]
                fn an_empty_vector_is_unchanged() {
                    let mut values: Vec<i64> = Vec::new();
                    swap_ends(&mut values);
                    assert!(values.is_empty());
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Swap the ends of a vector.

                pub fn swap_ends(values: &mut Vec<i64>) {
                    let last = values.len().saturating_sub(1);
                    if last > 0 {
                        values.swap(0, last);
                    }
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "values.swap(0, last);",
                     "replace": "values.swap(1, last);"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0499"]},
    },
    {
        "id": "sealed-borrow-03-temporary-split-iterator",
        "category": "lifetime-borrow",
        "difficulty": "medium",
        "capability": "Bind the collection a long-lived iterator borrows.",
        "prompt_intro": (
            "This crate does not compile. `first_segment_length` stores an iterator "
            "borrowed from a temporary vector, and the temporary is dropped at the "
            "end of the statement. Bind the collection before iterating. Return the "
            "length of the first comma-separated segment."),
        "observed_failure": (
            "error[E0716]: temporary value dropped while borrowed\n --> src/lib.rs:4:26\n"
            "  |\n4 |     let mut segments = text.split(',').collect::<Vec<_>>().iter();\n"
            "  |                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ creates a "
            "temporary value which is freed while still in use\n5 |     segments.next()\n"
            "  |     -------- borrow later used here"),
        "files": {
            "src/lib.rs": code("""
                //! First segment length. The crate does not build: the iterator borrows
                //! a temporary.

                pub fn first_segment_length(text: &str) -> usize {
                    let mut segments = text.split(',').collect::<Vec<_>>().iter();
                    segments.next().map(|segment| segment.len()).unwrap_or(0)
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use sealed_borrow_03_temporary_split_iterator::first_segment_length;

                #[test]
                fn measures_the_first_segment() {
                    assert_eq!(first_segment_length("alpha,beta"), 5);
                }

                #[test]
                fn a_single_segment_is_measured() {
                    assert_eq!(first_segment_length("one"), 3);
                }

                #[test]
                fn an_empty_segment_is_zero() {
                    assert_eq!(first_segment_length(""), 0);
                    assert_eq!(first_segment_length(",tail"), 0);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! First segment length.

                pub fn first_segment_length(text: &str) -> usize {
                    let segments: Vec<&str> = text.split(',').collect();
                    segments.first().map(|segment| segment.len()).unwrap_or(0)
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "segments.first()",
                     "replace": "segments.last()"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0716"]},
    },
]

FAILING = [
    {
        "id": "sealed-failing-01-luhn-parity",
        "category": "failing-test",
        "difficulty": "medium",
        "capability": "Double every second digit from the right in a Luhn checksum.",
        "prompt_intro": (
            "The visible test fails. `luhn_valid` doubles every second digit counting "
            "from the wrong end. Digit `d` is doubled to `2 * d`; if the result is "
            "above nine, subtract nine. The sum must be divisible by ten. Non-digit "
            "characters make the input invalid, and fewer than two digits is invalid."),
        "observed_failure": (
            "thread 'tests::visible_known_valid_number' panicked at src/lib.rs:34:9:\n"
            "assertion failed: luhn_valid(\"49927398716\")"),
        "files": {
            "src/lib.rs": code("""
                //! Luhn validation. The visible test fails: the doubling starts from
                //! the wrong end.

                pub fn luhn_valid(text: &str) -> bool {
                    let digits: Vec<u32> = match text
                        .chars()
                        .map(|character| character.to_digit(10))
                        .collect::<Option<Vec<u32>>>()
                    {
                        Some(digits) => digits,
                        None => return false,
                    };
                    if digits.len() < 2 {
                        return false;
                    }
                    let mut sum = 0u32;
                    for (index, digit) in digits.iter().enumerate() {
                        if index % 2 == 0 {
                            let doubled = digit * 2;
                            sum += if doubled > 9 { doubled - 9 } else { doubled };
                        } else {
                            sum += digit;
                        }
                    }
                    sum % 10 == 0
                }

                #[cfg(test)]
                mod tests {
                    use super::luhn_valid;

                    #[test]
                    fn visible_known_valid_number() {
                        assert!(luhn_valid("49927398716"));
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use sealed_failing_01_luhn_parity::luhn_valid;

                #[test]
                fn accepts_a_known_valid_number() {
                    assert!(luhn_valid("49927398716"));
                }

                #[test]
                fn accepts_another_known_valid_number() {
                    assert!(luhn_valid("79927398713"));
                }

                #[test]
                fn rejects_a_single_digit_change() {
                    assert!(!luhn_valid("49927398717"));
                }

                #[test]
                fn rejects_non_digits_and_short_input() {
                    assert!(!luhn_valid("4992739871x"));
                    assert!(!luhn_valid("7"));
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Luhn validation.

                pub fn luhn_valid(text: &str) -> bool {
                    let digits: Vec<u32> = match text
                        .chars()
                        .map(|character| character.to_digit(10))
                        .collect::<Option<Vec<u32>>>()
                    {
                        Some(digits) => digits,
                        None => return false,
                    };
                    if digits.len() < 2 {
                        return false;
                    }
                    let mut sum = 0u32;
                    for (offset, digit) in digits.iter().rev().enumerate() {
                        if offset % 2 == 1 {
                            let doubled = digit * 2;
                            sum += if doubled > 9 { doubled - 9 } else { doubled };
                        } else {
                            sum += digit;
                        }
                    }
                    sum % 10 == 0
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "if offset % 2 == 1 {",
                     "replace": "if offset % 2 == 0 {"},
        "initial_failure": {"kind": "test_failure",
                            "must_contain": ["assertion failed"]},
    },
    {
        "id": "sealed-failing-02-base36-table",
        "category": "failing-test",
        "difficulty": "easy",
        "capability": "Encode an integer in base 36 with a complete digit table.",
        "prompt_intro": (
            "The visible test fails. `to_base36` encodes a non-negative integer in "
            "lowercase base 36, but its digit table is one character short. Zero "
            "encodes as `0`."),
        "observed_failure": (
            "thread 'tests::visible_thirty_five' panicked at src/lib.rs:19:9:\n"
            "index out of bounds: the len is 35 but the index is 35"),
        "files": {
            "src/lib.rs": code("""
                //! Base-36 encoding with an incomplete digit table.

                pub fn to_base36(mut value: u64) -> String {
                    const DIGITS: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxy";
                    if value == 0 {
                        return "0".to_string();
                    }
                    let mut out = Vec::new();
                    while value > 0 {
                        out.push(DIGITS[(value % 36) as usize]);
                        value /= 36;
                    }
                    out.reverse();
                    String::from_utf8(out).unwrap()
                }

                #[cfg(test)]
                mod tests {
                    use super::to_base36;

                    #[test]
                    fn visible_thirty_five() {
                        assert_eq!(to_base36(35), "z");
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use sealed_failing_02_base36_table::to_base36;

                #[test]
                fn the_last_digit_is_z() {
                    assert_eq!(to_base36(35), "z");
                }

                #[test]
                fn zero_and_small_values() {
                    assert_eq!(to_base36(0), "0");
                    assert_eq!(to_base36(10), "a");
                    assert_eq!(to_base36(34), "y");
                }

                #[test]
                fn multi_digit_values() {
                    assert_eq!(to_base36(36), "10");
                    assert_eq!(to_base36(1295), "zz");
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Base-36 encoding.

                pub fn to_base36(mut value: u64) -> String {
                    const DIGITS: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
                    if value == 0 {
                        return "0".to_string();
                    }
                    let mut out = Vec::new();
                    while value > 0 {
                        out.push(DIGITS[(value % 36) as usize]);
                        value /= 36;
                    }
                    out.reverse();
                    String::from_utf8(out).unwrap()
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "value % 36", "replace": "value % 35"},
        "initial_failure": {"kind": "test_failure",
                            "must_contain": ["index out of bounds"]},
    },
    {
        "id": "sealed-failing-03-rectangle-transpose",
        "category": "failing-test",
        "difficulty": "medium",
        "capability": "Transpose a rectangular matrix, not only a square one.",
        "prompt_intro": (
            "The visible test fails. `transpose` assumes the matrix is square and "
            "indexes the source by the output's row and column, so a 2x3 matrix is "
            "transposed into the wrong shape. Transpose any rows x columns matrix; "
            "an empty matrix stays empty."),
        "observed_failure": (
            "thread 'tests::visible_rectangular' panicked at src/lib.rs:17:9:\n"
            "assertion `left == right` failed\n"
            "  left: [[1, 4], [2, 5], [3, 6]]\n right: [[1, 3, 5], [2, 4, 6]]"),
        "files": {
            "src/lib.rs": code("""
                //! Matrix transpose assuming a square input.

                pub fn transpose(matrix: &[Vec<i32>]) -> Vec<Vec<i32>> {
                    let size = matrix.len();
                    let mut out = vec![vec![0i32; size]; size];
                    for row in 0..size {
                        for column in 0..size {
                            out[column][row] = matrix[row][column];
                        }
                    }
                    out
                }

                #[cfg(test)]
                mod tests {
                    use super::transpose;

                    #[test]
                    fn visible_rectangular() {
                        let matrix = vec![vec![1, 2, 3], vec![4, 5, 6]];
                        assert_eq!(transpose(&matrix),
                                   vec![vec![1, 4], vec![2, 5], vec![3, 6]]);
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use sealed_failing_03_rectangle_transpose::transpose;

                #[test]
                fn a_two_by_three_matrix_transposes() {
                    let matrix = vec![vec![1, 2, 3], vec![4, 5, 6]];
                    assert_eq!(transpose(&matrix),
                               vec![vec![1, 4], vec![2, 5], vec![3, 6]]);
                }

                #[test]
                fn a_square_matrix_transposes() {
                    let matrix = vec![vec![1, 2], vec![3, 4]];
                    assert_eq!(transpose(&matrix), vec![vec![1, 3], vec![2, 4]]);
                }

                #[test]
                fn an_empty_matrix_stays_empty() {
                    assert!(transpose(&[]).is_empty());
                }

                #[test]
                fn a_single_row_transposes_to_a_column() {
                    let matrix = vec![vec![1, 2, 3]];
                    assert_eq!(transpose(&matrix), vec![vec![1], vec![2], vec![3]]);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Matrix transpose.

                pub fn transpose(matrix: &[Vec<i32>]) -> Vec<Vec<i32>> {
                    let rows = matrix.len();
                    let columns = matrix.first().map(|row| row.len()).unwrap_or(0);
                    let mut out = vec![Vec::with_capacity(rows); columns];
                    for row in matrix {
                        for (column, value) in row.iter().enumerate() {
                            out[column].push(*value);
                        }
                    }
                    out
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "matrix.first().map(|row| row.len())",
                     "replace": "matrix.first().map(|row| row.len() - 1)"},
        "initial_failure": {"kind": "test_failure",
                            "must_contain": ["assertion `left == right` failed"]},
    },
]

MIGRATION = [
    {
        "id": "sealed-migrate-01-option-argument",
        "category": "api-migration",
        "difficulty": "easy",
        "capability": "Wrap a value in the Option that version 2 requires.",
        "prompt_intro": (
            "This crate does not build against the vendored `store` 2.0.0. Version 2 "
            "made the stored value an `Option<i64>`, so `Some(value)` is required and "
            "a key can be cleared with `None`. Wrap the value. Do not modify `dep/`."),
        "observed_failure": (
            "error[E0308]: mismatched types\n --> src/lib.rs:6:16\n  |\n"
            "6 |     store.set(key, value);\n"
            "  |           --- ^^^^^ expected `Option<i64>`, found `i64`"),
        "files": {
            "dep/Cargo.toml": (
                "[package]\n"
                'name = "store"\n'
                'version = "2.0.0"\n'
                'edition = "2021"\n'
                "\n"
                "[lib]\n"
                'path = "src/lib.rs"\n'),
            "dep/src/lib.rs": code("""
                //! Store, version 2. Values are optional so a key can be cleared.

                use std::collections::BTreeMap;

                pub struct Store {
                    values: BTreeMap<String, Option<i64>>,
                }

                impl Store {
                    pub fn new() -> Self {
                        Self { values: BTreeMap::new() }
                    }

                    pub fn set(&mut self, key: &str, value: Option<i64>) {
                        self.values.insert(key.to_string(), value);
                    }

                    pub fn get(&self, key: &str) -> Option<i64> {
                        self.values.get(key).copied().flatten()
                    }
                }
            """),
            "src/lib.rs": code("""
                //! Record a value. This does not build: version 2 takes an Option.

                pub use store::Store;

                pub fn record(store: &mut Store, key: &str, value: i64) {
                    store.set(key, value);
                }

                pub fn read(store: &Store, key: &str) -> Option<i64> {
                    store.get(key)
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use sealed_migrate_01_option_argument::{read, record, Store};

                #[test]
                fn a_recorded_value_can_be_read() {
                    let mut store = Store::new();
                    record(&mut store, "a", 7);
                    assert_eq!(read(&store, "a"), Some(7));
                }

                #[test]
                fn reading_a_missing_key_is_none() {
                    let store = Store::new();
                    assert_eq!(read(&store, "missing"), None);
                }

                #[test]
                fn a_later_record_overwrites() {
                    let mut store = Store::new();
                    record(&mut store, "a", 1);
                    record(&mut store, "a", 2);
                    assert_eq!(read(&store, "a"), Some(2));
                }
            """),
        },
        "readonly": ["dep/"],
        "reference": {
            "src/lib.rs": code("""
                //! Record a value.

                pub use store::Store;

                pub fn record(store: &mut Store, key: &str, value: i64) {
                    store.set(key, Some(value));
                }

                pub fn read(store: &Store, key: &str) -> Option<i64> {
                    store.get(key)
                }
            """),
        },
        "manifest": (
            "[package]\n"
            'name = "sealed_migrate_01_option_argument"\n'
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
            'store = { path = "dep" }\n'),
        "mutation": {"file": "src/lib.rs", "find": "Some(value)", "replace": "None"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0308"]},
    },
    {
        "id": "sealed-migrate-02-iterator-return",
        "category": "api-migration",
        "difficulty": "medium",
        "capability": "Iterate an owned iterator instead of indexing a borrowed slice.",
        "prompt_intro": (
            "This crate does not build against the vendored `catalog` 2.0.0. Version "
            "2 returns an owning iterator instead of a borrowed slice, so the caller "
            "cannot index it. Return the number of entries and the first entry, if "
            "any. Do not modify `dep/`."),
        "observed_failure": (
            "error[E0608]: cannot index into a value of type "
            "`impl Iterator<Item = String>`\n --> src/lib.rs:6:28\n  |\n"
            "6 |     let first = catalog.items()[0].clone();\n"
            "  |                            ^^^^^ cannot be indexed"),
        "files": {
            "dep/Cargo.toml": (
                "[package]\n"
                'name = "catalog"\n'
                'version = "2.0.0"\n'
                'edition = "2021"\n'
                "\n"
                "[lib]\n"
                'path = "src/lib.rs"\n'),
            "dep/src/lib.rs": code("""
                //! Catalog, version 2. `items` owns its result instead of borrowing the
                //! catalog.

                pub struct Catalog {
                    entries: Vec<String>,
                }

                impl Catalog {
                    pub fn new(entries: Vec<String>) -> Self {
                        Self { entries }
                    }

                    pub fn items(&self) -> impl Iterator<Item = String> + '_ {
                        self.entries.iter().cloned()
                    }
                }
            """),
            "src/lib.rs": code("""
                //! Catalog summary. This does not build: version 2 returns an iterator.

                pub use catalog::Catalog;

                pub fn summarise(catalog: &Catalog) -> (usize, Option<String>) {
                    let first = catalog.items()[0].clone();
                    (catalog.items().count(), Some(first))
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use sealed_migrate_02_iterator_return::{summarise, Catalog};

                fn catalog(names: &[&str]) -> Catalog {
                    Catalog::new(names.iter().map(|name| name.to_string()).collect())
                }

                #[test]
                fn summarises_a_non_empty_catalog() {
                    assert_eq!(summarise(&catalog(&["a", "b"])),
                               (2, Some("a".to_string())));
                }

                #[test]
                fn summarises_an_empty_catalog() {
                    assert_eq!(summarise(&catalog(&[])), (0, None));
                }

                #[test]
                fn a_single_entry_is_summarised() {
                    assert_eq!(summarise(&catalog(&["only"])),
                               (1, Some("only".to_string())));
                }
            """),
        },
        "readonly": ["dep/"],
        "reference": {
            "src/lib.rs": code("""
                //! Catalog summary.

                pub use catalog::Catalog;

                pub fn summarise(catalog: &Catalog) -> (usize, Option<String>) {
                    let entries: Vec<String> = catalog.items().collect();
                    let first = entries.first().cloned();
                    (entries.len(), first)
                }
            """),
        },
        "manifest": (
            "[package]\n"
            'name = "sealed_migrate_02_iterator_return"\n'
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
            'catalog = { path = "dep" }\n'),
        "mutation": {"file": "src/lib.rs", "find": "entries.len()", "replace": "entries.len() + 1"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0608"]},
    },
    {
        "id": "sealed-migrate-03-error-not-std-error",
        "category": "api-migration",
        "difficulty": "hard",
        "capability": "Convert an error type that version 2 stopped deriving std::error::Error.",
        "prompt_intro": (
            "This crate does not build against the vendored `parse` 2.0.0. Version 2 "
            "keeps its error type but no longer implements `std::error::Error`, so it "
            "cannot be converted into `Box<dyn Error>` with `?`. Convert the error to "
            "its rendered message instead. Do not modify `dep/`."),
        "observed_failure": (
            "error[E0277]: `parse::Error` doesn't implement `std::error::Error`\n"
            " --> src/lib.rs:5:37\n  |\n5 |     let value = parse::parse(text)?;\n"
            "  |                                     ^ the trait `std::error::Error` is not "
            "implemented for `parse::Error`"),
        "files": {
            "dep/Cargo.toml": (
                "[package]\n"
                'name = "parse"\n'
                'version = "2.0.0"\n'
                'edition = "2021"\n'
                "\n"
                "[lib]\n"
                'path = "src/lib.rs"\n'),
            "dep/src/lib.rs": code("""
                //! Parse, version 2. The error type no longer implements
                //! `std::error::Error`; it renders itself only through `Display`.

                pub struct Error {
                    pub reason: &'static str,
                }

                impl core::fmt::Display for Error {
                    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                        write!(formatter, "{}", self.reason)
                    }
                }

                pub fn parse(text: &str) -> Result<i64, Error> {
                    text.trim()
                        .parse()
                        .map_err(|_| Error { reason: "not a number" })
                }
            """),
            "src/lib.rs": code("""
                //! Parsing through the vendored crate. This does not build: version 2's
                //! error is not a `std::error::Error`.

                pub fn parse_or_message(text: &str) -> Result<i64, Box<dyn std::error::Error>> {
                    let value = parse::parse(text)?;
                    Ok(value * 2)
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use sealed_migrate_03_error_not_std_error::parse_or_message;

                #[test]
                fn a_number_is_doubled() {
                    assert_eq!(parse_or_message("21").unwrap(), 42);
                }

                #[test]
                fn surrounding_space_is_ignored() {
                    assert_eq!(parse_or_message(" 3 ").unwrap(), 6);
                }

                #[test]
                fn text_is_an_error_with_the_vendored_message() {
                    let error = parse_or_message("nope").unwrap_err();
                    assert_eq!(error.to_string(), "not a number");
                }
            """),
        },
        "readonly": ["dep/"],
        "reference": {
            "src/lib.rs": code("""
                //! Parsing through the vendored crate.

                pub fn parse_or_message(text: &str) -> Result<i64, Box<dyn std::error::Error>> {
                    let value = parse::parse(text).map_err(|error| error.to_string())?;
                    Ok(value * 2)
                }
            """),
        },
        "manifest": (
            "[package]\n"
            'name = "sealed_migrate_03_error_not_std_error"\n'
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
            'parse = { path = "dep" }\n'),
        "mutation": {"file": "src/lib.rs", "find": "Ok(value * 2)", "replace": "Ok(value + 2)"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0277"]},
    },
]

DEPENDENCY = [
    {
        "id": "sealed-deps-01-missing-workspace-key",
        "category": "dependency-compat",
        "difficulty": "easy",
        "capability": "Isolate a nested package from the workspace that contains its directory.",
        "prompt_intro": (
            "This crate does not build. It sits below a workspace directory but "
            "declares no workspace of its own, so Cargo believes it belongs to the "
            "outer workspace and is not a member. Make it its own workspace. Keep "
            "the behaviour."),
        "observed_failure": (
            "error: current package believes it's in a workspace when it's not:\n"
            "current:   /work/.../Cargo.toml\nworkspace: /work/Cargo.toml\n"
            "this may be fixable by adding `...` to the `workspace.members` array"),
        "files": {
            "Cargo.toml": (
                "[package]\n"
                'name = "sealed_deps_01_missing_workspace_key"\n'
                'version = "0.1.0"\n'
                'edition = "2021"\n'
                "\n"
                "[lib]\n"
                'path = "src/lib.rs"\n'),
            "src/lib.rs": code("""
                //! A nested package. This does not build: it declares no workspace.

                pub fn max_gap(values: &[u32]) -> u32 {
                    values
                        .windows(2)
                        .map(|pair| pair[1].saturating_sub(pair[0]))
                        .max()
                        .unwrap_or(0)
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use sealed_deps_01_missing_workspace_key::max_gap;

                #[test]
                fn the_largest_rise_wins() {
                    assert_eq!(max_gap(&[1, 4, 2]), 3);
                }

                #[test]
                fn a_single_value_has_no_gap() {
                    assert_eq!(max_gap(&[5]), 0);
                }

                #[test]
                fn a_falling_pair_has_no_gap() {
                    assert_eq!(max_gap(&[9, 1]), 0);
                }
            """),
        },
        "manifest_policy": "arm",
        "reference": {
            "Cargo.toml": (
                "[package]\n"
                'name = "sealed_deps_01_missing_workspace_key"\n'
                'version = "0.1.0"\n'
                'edition = "2021"\n'
                "\n"
                "[workspace]\n"
                "\n"
                "[lib]\n"
                'path = "src/lib.rs"\n'),
        },
        "mutation": {"file": "Cargo.toml", "find": "\n[workspace]\n", "replace": "\n"},
        "initial_failure": {"kind": "compile_error",
                            "must_contain": ["believes it's in a workspace"]},
    },
    {
        "id": "sealed-deps-02-resolver-type",
        "category": "dependency-compat",
        "difficulty": "easy",
        "capability": "Correct the TOML type of a workspace resolver field.",
        "prompt_intro": (
            "This crate does not build. Its workspace declares `resolver = 2` as an "
            "integer; Cargo requires a string. Correct the type. Keep the behaviour."),
        "observed_failure": (
            "error: invalid type: integer `2`, expected a string\n --> Cargo.toml:7:12\n"
            "  |\n7 | resolver = 2\n  |            ^"),
        "files": {
            "Cargo.toml": (
                "[package]\n"
                'name = "sealed_deps_02_resolver_type"\n'
                'version = "0.1.0"\n'
                'edition = "2021"\n'
                "\n"
                "[workspace]\n"
                "resolver = 2\n"
                "\n"
                "[lib]\n"
                'path = "src/lib.rs"\n'),
            "src/lib.rs": code("""
                //! A workspace resolver. This does not build: the resolver is an integer.

                pub fn reverse_words(text: &str) -> String {
                    text.split_whitespace()
                        .rev()
                        .collect::<Vec<_>>()
                        .join(" ")
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use sealed_deps_02_resolver_type::reverse_words;

                #[test]
                fn three_words_are_reversed() {
                    assert_eq!(reverse_words("a b c"), "c b a");
                }

                #[test]
                fn surrounding_space_is_ignored() {
                    assert_eq!(reverse_words(" single "), "single");
                }

                #[test]
                fn empty_input_is_empty() {
                    assert_eq!(reverse_words(""), "");
                }
            """),
        },
        "manifest_policy": "arm",
        "reference": {
            "Cargo.toml": (
                "[package]\n"
                'name = "sealed_deps_02_resolver_type"\n'
                'version = "0.1.0"\n'
                'edition = "2021"\n'
                "\n"
                "[workspace]\n"
                'resolver = "2"\n'
                "\n"
                "[lib]\n"
                'path = "src/lib.rs"\n'),
        },
        "mutation": {"file": "Cargo.toml", "find": 'resolver = "2"',
                     "replace": "resolver = 2"},
        "initial_failure": {"kind": "compile_error",
                            "must_contain": ["invalid type: integer"]},
    },
    {
        "id": "sealed-deps-03-static-mut-edition",
        "category": "dependency-compat",
        "difficulty": "medium",
        "capability": "Replace an edition-2024-rejected mutable static with an atomic.",
        "prompt_intro": (
            "This crate does not build under edition 2024: taking a shared reference "
            "to a `static mut` is an error. Replace the mutable static with an "
            "atomic counter and keep `bump()` returning the new value, starting at "
            "one. You may edit `src/lib.rs` or `Cargo.toml`."),
        "observed_failure": (
            "error: creating a shared reference to mutable static\n --> src/lib.rs:5:26\n"
            "  |\n5 |     let reference = unsafe { &COUNTER };\n"
            "  |                          ^^^^^^^^ shared reference to mutable static"),
        "files": {
            "Cargo.toml": (
                "[package]\n"
                'name = "sealed_deps_03_static_mut_edition"\n'
                'version = "0.1.0"\n'
                'edition = "2024"\n'
                "\n"
                "[workspace]\n"
                "\n"
                "[lib]\n"
                'path = "src/lib.rs"\n'),
            "src/lib.rs": code("""
                //! A mutable static counter. This does not build under edition 2024.

                static mut COUNTER: i64 = 0;

                pub fn bump() -> i64 {
                    let reference = unsafe { &COUNTER };
                    unsafe { COUNTER += 1; }
                    *reference
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use sealed_deps_03_static_mut_edition::bump;

                #[test]
                fn a_bump_returns_a_positive_value() {
                    assert!(bump() >= 1);
                }

                #[test]
                fn successive_bumps_increase() {
                    let before = bump();
                    let after = bump();
                    assert!(after > before);
                }

                #[test]
                fn bumps_are_unique() {
                    assert_ne!(bump(), bump());
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! An atomic counter.

                use std::sync::atomic::{AtomicI64, Ordering};

                static COUNTER: AtomicI64 = AtomicI64::new(0);

                pub fn bump() -> i64 {
                    COUNTER.fetch_add(1, Ordering::SeqCst) + 1
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "fetch_add(1, Ordering::SeqCst) + 1",
                     "replace": "fetch_add(1, Ordering::SeqCst)"},
        "initial_failure": {"kind": "compile_error",
                            "must_contain": ["creating a shared reference to mutable static"]},
    },
]

CONCURRENCY = [
    {
        "id": "sealed-concurrency-01-bounded-channel-join",
        "category": "concurrency",
        "difficulty": "medium",
        "capability": "Drain a bounded channel while its producer is still running.",
        "prompt_intro": (
            "The crate builds, but `collect_after_join` never returns: the channel "
            "holds one message, the producer sends several, and the producer is joined "
            "before the receiver is drained. Drain while the producer runs. Return "
            "the messages in order."),
        "observed_failure": (
            "running 1 test\ntest tests::visible_three_messages ... "),
        "files": {
            "src/lib.rs": code("""
                //! A bounded channel drained too late. The crate builds, but the
                //! producer is joined before the receiver drains.

                use std::sync::mpsc::sync_channel;

                pub fn collect_after_join(count: usize) -> Vec<usize> {
                    let (sender, receiver) = sync_channel(1);
                    std::thread::scope(|scope| {
                        scope.spawn(|| {
                            for value in 0..count {
                                sender.send(value).unwrap();
                            }
                        });
                    });
                    receiver.iter().collect()
                }

                #[cfg(test)]
                mod tests {
                    #[test]
                    fn visible_three_messages() {
                        assert_eq!(super::collect_after_join(3), vec![0, 1, 2]);
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use sealed_concurrency_01_bounded_channel_join::collect_after_join;

                #[test]
                fn three_messages_are_collected_in_order() {
                    assert_eq!(collect_after_join(3), vec![0, 1, 2]);
                }

                #[test]
                fn one_message_is_collected() {
                    assert_eq!(collect_after_join(1), vec![0]);
                }

                #[test]
                fn no_messages_is_empty() {
                    assert!(collect_after_join(0).is_empty());
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! A bounded channel drained while the producer runs.

                use std::sync::mpsc::sync_channel;

                pub fn collect_after_join(count: usize) -> Vec<usize> {
                    let (sender, receiver) = sync_channel(1);
                    let mut out = Vec::new();
                    std::thread::scope(|scope| {
                        scope.spawn(move || {
                            for value in 0..count {
                                sender.send(value).unwrap();
                            }
                        });
                        out.extend(receiver.iter());
                    });
                    out
                }
            """),
        },
        "limits": {"wall_seconds": 25.0, "cpu_seconds": 40},
        "mutation": {"file": "src/lib.rs", "find": "for value in 0..count {",
                     "replace": "for value in 0..count.min(1) {"},
        "initial_failure": {"kind": "timeout", "must_contain": ["running 1 test"]},
    },
    {
        "id": "sealed-concurrency-02-atomic-take-pair",
        "category": "concurrency",
        "difficulty": "hard",
        "capability": "Drain a counter atomically instead of reading and storing it.",
        "prompt_intro": (
            "The visible test fails. Two threads call `take_pair` at the same time; "
            "each reads the counter and then stores zero, so both report the same "
            "value and the total is counted twice. Drain the counter atomically. The "
            "two results must sum to the value that was in the counter, and the "
            "counter must end at zero."),
        "observed_failure": (
            "thread 'tests::visible_pair' panicked at src/lib.rs:30:9:\n"
            "assertion `left == right` failed\n  left: 10\n right: 5"),
        "files": {
            "src/lib.rs": code("""
                //! Two concurrent drains. The visible test fails: the read and the store
                //! are separate operations.

                use std::sync::atomic::{AtomicUsize, Ordering};
                use std::sync::Barrier;

                pub fn take_pair(counter: &AtomicUsize) -> usize {
                    let barrier = Barrier::new(2);
                    std::thread::scope(|scope| {
                        let first = scope.spawn(|| {
                            let value = counter.load(Ordering::SeqCst);
                            barrier.wait();
                            counter.store(0, Ordering::SeqCst);
                            value
                        });
                        let second = scope.spawn(|| {
                            let value = counter.load(Ordering::SeqCst);
                            barrier.wait();
                            counter.store(0, Ordering::SeqCst);
                            value
                        });
                        first.join().unwrap() + second.join().unwrap()
                    })
                }

                #[cfg(test)]
                mod tests {
                    use super::take_pair;
                    use std::sync::atomic::AtomicUsize;

                    #[test]
                    fn visible_pair() {
                        let counter = AtomicUsize::new(5);
                        assert_eq!(take_pair(&counter), 5);
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use std::sync::atomic::{AtomicUsize, Ordering};

                use sealed_concurrency_02_atomic_take_pair::take_pair;

                #[test]
                fn the_pair_sums_to_the_stored_value() {
                    let counter = AtomicUsize::new(5);
                    assert_eq!(take_pair(&counter), 5);
                    assert_eq!(counter.load(Ordering::SeqCst), 0);
                }

                #[test]
                fn an_empty_counter_yields_zero() {
                    let counter = AtomicUsize::new(0);
                    assert_eq!(take_pair(&counter), 0);
                    assert_eq!(counter.load(Ordering::SeqCst), 0);
                }

                #[test]
                fn the_pair_runs_again_on_a_refilled_counter() {
                    let counter = AtomicUsize::new(9);
                    assert_eq!(take_pair(&counter), 9);
                    counter.store(4, Ordering::SeqCst);
                    assert_eq!(take_pair(&counter), 4);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Two concurrent atomic drains.

                use std::sync::atomic::{AtomicUsize, Ordering};
                use std::sync::Barrier;

                pub fn take_pair(counter: &AtomicUsize) -> usize {
                    let barrier = Barrier::new(2);
                    std::thread::scope(|scope| {
                        let first = scope.spawn(|| {
                            let value = counter.swap(0, Ordering::SeqCst);
                            barrier.wait();
                            value
                        });
                        let second = scope.spawn(|| {
                            let value = counter.swap(0, Ordering::SeqCst);
                            barrier.wait();
                            value
                        });
                        first.join().unwrap() + second.join().unwrap()
                    })
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "counter.swap(0, Ordering::SeqCst)",
                     "replace": "counter.load(Ordering::SeqCst)"},
        "initial_failure": {"kind": "test_failure",
                            "must_contain": ["assertion `left == right` failed"]},
    },
    {
        "id": "sealed-concurrency-03-two-closures-unique-access",
        "category": "concurrency",
        "difficulty": "medium",
        "capability": "Share a counter between scoped threads with an atomic instead of &mut.",
        "prompt_intro": (
            "This crate does not compile. Both scoped closures try to hold the same "
            "mutable reference to the counter, which Rust forbids. Take an `AtomicU32` "
            "instead, add one per closure, and return the final value."),
        "observed_failure": (
            "error[E0499]: cannot borrow `*counter` as mutable more than once at a time\n"
            " --> src/lib.rs:4:21\n  |\n3 |     std::thread::scope(|scope| {\n"
            "  |                        ------- first mutable borrow occurs here\n"
            "4 |         scope.spawn(|| { *counter += 1; });\n"
            "  |                     ^^ second mutable borrow occurs here"),
        "files": {
            "src/lib.rs": code("""
                //! Two scoped increments. The crate does not compile: both closures hold
                //! the same mutable reference.

                pub fn bump_both(counter: &mut u32) {
                    std::thread::scope(|scope| {
                        scope.spawn(|| {
                            *counter += 1;
                        });
                        scope.spawn(|| {
                            *counter += 1;
                        });
                    });
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use std::sync::atomic::AtomicU32;

                use sealed_concurrency_03_two_closures_unique_access::bump_both;

                #[test]
                fn both_closures_add_one() {
                    assert_eq!(bump_both(&AtomicU32::new(0)), 2);
                }

                #[test]
                fn the_starting_value_is_kept() {
                    assert_eq!(bump_both(&AtomicU32::new(10)), 12);
                }

                #[test]
                fn repeated_calls_advance() {
                    let counter = AtomicU32::new(0);
                    assert_eq!(bump_both(&counter), 2);
                    assert_eq!(bump_both(&counter), 4);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Two scoped atomic increments.

                use std::sync::atomic::{AtomicU32, Ordering};

                pub fn bump_both(counter: &AtomicU32) -> u32 {
                    std::thread::scope(|scope| {
                        scope.spawn(|| {
                            counter.fetch_add(1, Ordering::SeqCst);
                        });
                        scope.spawn(|| {
                            counter.fetch_add(1, Ordering::SeqCst);
                        });
                    });
                    counter.load(Ordering::SeqCst)
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "counter.fetch_add(1, Ordering::SeqCst);",
                     "replace": "counter.fetch_add(2, Ordering::SeqCst);"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0499"]},
    },
]

TRAITS = [
    {
        "id": "sealed-traits-01-unconstrained-impl-parameter",
        "category": "trait-impl",
        "difficulty": "medium",
        "capability": "Remove a type parameter that the impl does not constrain.",
        "prompt_intro": (
            "This crate does not compile. The inherent impl for `Registry` declares a "
            "type parameter that appears nowhere in the self type or the methods, so "
            "the compiler cannot infer it. Remove the parameter and keep the "
            "constructor."),
        "observed_failure": (
            "error[E0207]: the type parameter `T` is not constrained by the impl trait, "
            "self type, or predicates\n --> src/lib.rs:5:6\n  |\n"
            "5 | impl<T> Registry {\n  |      ^ unconstrained type parameter"),
        "files": {
            "src/lib.rs": code("""
                //! A registry with an unconstrained impl parameter. The crate does not
                //! build.

                #[derive(Default)]
                pub struct Registry {
                    pub entries: Vec<String>,
                }

                impl<T> Registry {
                    pub fn new() -> Self {
                        Self { entries: Vec::new() }
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use sealed_traits_01_unconstrained_impl_parameter::Registry;

                #[test]
                fn the_constructor_is_empty() {
                    assert!(Registry::new().entries.is_empty());
                }

                #[test]
                fn a_constructed_registry_can_be_used() {
                    let mut registry = Registry::new();
                    registry.entries.push("a".to_string());
                    assert_eq!(registry.entries.len(), 1);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! A registry.

                #[derive(Default)]
                pub struct Registry {
                    pub entries: Vec<String>,
                }

                impl Registry {
                    pub fn new() -> Self {
                        Self { entries: Vec::new() }
                    }
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "Self { entries: Vec::new() }",
                     "replace": "Self { entries: vec![String::new()] }"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0207"]},
    },
    {
        "id": "sealed-traits-02-missing-associated-item",
        "category": "trait-impl",
        "difficulty": "medium",
        "capability": "Name an associated type that the generic helper requires.",
        "prompt_intro": (
            "This crate does not compile. `clone_value` asks for `E::Number`, but the "
            "trait declares `Value`. Use the declared associated type. Do not change "
            "the trait."),
        "observed_failure": (
            "error[E0220]: associated type `Number` not found for `E`\n"
            " --> src/lib.rs:10:40\n  |\n10 | pub fn clone_value<E: Element>(element: &E) -> E::Number {\n"
            "  |                                        ^^^^^^^^^ help: there is an associated "
            "type with a similar name: `Value`"),
        "files": {
            "src/lib.rs": code("""
                //! An element trait. The crate does not build: the helper names an
                //! associated type the trait does not declare.

                pub trait Element {
                    type Value;
                    fn value(&self) -> Self::Value;
                }

                pub struct Cell {
                    pub item: u8,
                }

                impl Element for Cell {
                    type Value = u8;

                    fn value(&self) -> u8 {
                        self.item
                    }
                }

                pub fn clone_value<E: Element>(element: &E) -> E::Number {
                    element.value()
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use sealed_traits_02_missing_associated_item::{clone_value, Cell, Element};

                #[test]
                fn the_helper_returns_the_value() {
                    let cell = Cell { item: 7 };
                    assert_eq!(clone_value(&cell), 7);
                }

                #[test]
                fn the_trait_method_agrees() {
                    let cell = Cell { item: 3 };
                    assert_eq!(cell.value(), clone_value(&cell));
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! An element trait.

                pub trait Element {
                    type Value;
                    fn value(&self) -> Self::Value;
                }

                pub struct Cell {
                    pub item: u8,
                }

                impl Element for Cell {
                    type Value = u8;

                    fn value(&self) -> u8 {
                        self.item
                    }
                }

                pub fn clone_value<E: Element>(element: &E) -> E::Value {
                    element.value()
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "element.value()",
                     "replace": "element.value() + 1"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0220"]},
    },
    {
        "id": "sealed-traits-03-dyn-associated-type",
        "category": "trait-impl",
        "difficulty": "medium",
        "capability": "Specify an associated type when using a trait object.",
        "prompt_intro": (
            "This crate does not compile. `describe` takes `&dyn Store`, but `Store` "
            "has an associated type, so the object type must name it. Specify "
            "`Output = u8` and keep the behaviour."),
        "observed_failure": (
            "error[E0191]: the value of the associated type `Output` (from trait `Store`) "
            "must be specified\n --> src/lib.rs:9:30\n  |\n"
            "9 | pub fn describe(store: &dyn Store) -> u8 {\n"
            "  |                              ^^^^^ help: specify the associated type: "
            "`Store<Output = Type>`"),
        "files": {
            "src/lib.rs": code("""
                //! A trait object with an unspecified associated type. The crate does
                //! not build.

                pub trait Store {
                    type Output;
                    fn get(&self, key: u8) -> Self::Output;
                }

                pub struct Map;

                impl Store for Map {
                    type Output = u8;

                    fn get(&self, key: u8) -> u8 {
                        key
                    }
                }

                pub fn describe(store: &dyn Store) -> u8 {
                    store.get(0)
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use sealed_traits_03_dyn_associated_type::{describe, Map};

                #[test]
                fn the_trait_object_returns_the_value() {
                    assert_eq!(describe(&Map), 0);
                }

                #[test]
                fn the_concrete_type_agrees() {
                    use sealed_traits_03_dyn_associated_type::Store;
                    assert_eq!(Map.get(0), describe(&Map));
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! A trait object with its associated type specified.

                pub trait Store {
                    type Output;
                    fn get(&self, key: u8) -> Self::Output;
                }

                pub struct Map;

                impl Store for Map {
                    type Output = u8;

                    fn get(&self, key: u8) -> u8 {
                        key
                    }
                }

                pub fn describe(store: &dyn Store<Output = u8>) -> u8 {
                    store.get(0)
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "store.get(0)", "replace": "store.get(1)"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0191"]},
    },
]

REPO = [
    {
        "id": "sealed-repo-01-private-method",
        "category": "repo-scale",
        "difficulty": "easy",
        "capability": "Widen the visibility of a method a sibling module calls.",
        "prompt_intro": (
            "This crate does not compile. `model::Config` keeps `secret` private, and "
            "`report` calls it from another module. Make the method visible to the "
            "crate. Keep the value and the rendering."),
        "observed_failure": (
            "error[E0624]: method `secret` is private\n --> src/report.rs:4:48\n  |\n"
            "4 | pub fn describe(config: &Config) -> u8 { config.secret() }\n"
            "  |                                                ^^^^^^^^ private method"),
        "files": {
            "src/lib.rs": code("""
                //! Configuration and its report.

                pub mod model;
                pub mod report;

                pub use model::Config;
                pub use report::describe;
            """),
            "src/model.rs": code("""
                //! The configuration type.

                pub struct Config {
                    pub value: u8,
                }

                impl Config {
                    pub fn new(value: u8) -> Self {
                        Self { value }
                    }

                    fn secret(&self) -> u8 {
                        self.value
                    }
                }
            """),
            "src/report.rs": code("""
                //! A report over the configuration. This does not compile: the method it
                //! calls is private to `model`.

                use crate::model::Config;

                pub fn describe(config: &Config) -> u8 {
                    config.secret()
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use sealed_repo_01_private_method::{describe, Config};

                #[test]
                fn the_report_reads_the_value() {
                    assert_eq!(describe(&Config::new(7)), 7);
                }

                #[test]
                fn the_constructor_still_works() {
                    assert_eq!(Config::new(3).value, 3);
                }
            """),
        },
        "reference": {
            "src/model.rs": code("""
                //! The configuration type.

                pub struct Config {
                    pub value: u8,
                }

                impl Config {
                    pub fn new(value: u8) -> Self {
                        Self { value }
                    }

                    pub(crate) fn secret(&self) -> u8 {
                        self.value
                    }
                }
            """),
        },
        "mutation": {"file": "src/model.rs", "find": "self.value",
                     "replace": "self.value + 1"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0624"]},
    },
    {
        "id": "sealed-repo-02-duplicate-reexport",
        "category": "repo-scale",
        "difficulty": "medium",
        "capability": "Disambiguate two modules that export the same name.",
        "prompt_intro": (
            "This crate does not compile. Both `left` and `right` export an `Item`, "
            "and the crate root re-exports both under the same name. Re-export the "
            "second one under a different name, and keep both behaviours."),
        "observed_failure": (
            "error[E0252]: the name `Item` is defined multiple times\n --> src/lib.rs:6:9\n"
            "  |\n5 | pub use left::Item;\n  |         --------- previous import of the type "
            "`Item` here\n6 | pub use right::Item;\n  |         ^^^^^^^^^^^ `Item` reimported here"),
        "files": {
            "src/lib.rs": code("""
                //! Two item types re-exported at the root. This does not compile: both
                //! re-exports use the same name.

                pub mod left;
                pub mod right;

                pub use left::Item;
                pub use right::Item;
            """),
            "src/left.rs": code("""
                //! The left item.

                pub struct Item {
                    pub value: u8,
                }

                impl Item {
                    pub fn left_value(&self) -> u8 {
                        self.value
                    }
                }
            """),
            "src/right.rs": code("""
                //! The right item.

                pub struct Item {
                    pub value: u8,
                }

                impl Item {
                    pub fn right_value(&self) -> u8 {
                        self.value * 2
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use sealed_repo_02_duplicate_reexport::{Item, RightItem};

                #[test]
                fn the_left_item_keeps_its_name() {
                    let item = Item { value: 4 };
                    assert_eq!(item.left_value(), 4);
                }

                #[test]
                fn the_right_item_is_aliased() {
                    let item = RightItem { value: 4 };
                    assert_eq!(item.right_value(), 8);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Two item types re-exported at the root.

                pub mod left;
                pub mod right;

                pub use left::Item;
                pub use right::Item as RightItem;
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "pub use right::Item as RightItem;",
                     "replace": "pub use right::Item as RightItem;\npub use left::Item as RightItem;"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0252"]},
    },
    {
        "id": "sealed-repo-03-missing-field",
        "category": "repo-scale",
        "difficulty": "easy",
        "capability": "Update a struct initializer after the model gained a field.",
        "prompt_intro": (
            "This crate does not compile. `model::Config` gained a `label` field, and "
            "`build::make` still constructs it with only `retries`. Supply the label "
            "`\"default\"` and keep the retry count at three."),
        "observed_failure": (
            "error[E0063]: missing field `label` in initializer of `Config`\n"
            " --> src/build.rs:4:24\n  |\n4 |     Config { retries: 3 }\n"
            "  |            ^^^^^^ missing `label`"),
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
                    pub retries: u32,
                    pub label: String,
                }
            """),
            "src/build.rs": code("""
                //! Build a configuration. This does not compile: the model gained a field.

                use crate::model::Config;

                pub fn make() -> Config {
                    Config { retries: 3 }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use sealed_repo_03_missing_field::make;

                #[test]
                fn the_default_configuration_is_complete() {
                    let config = make();
                    assert_eq!(config.retries, 3);
                    assert_eq!(config.label, "default");
                }
            """),
        },
        "reference": {
            "src/build.rs": code("""
                //! Build a configuration.

                use crate::model::Config;

                pub fn make() -> Config {
                    Config { retries: 3, label: "default".to_string() }
                }
            """),
        },
        "mutation": {"file": "src/build.rs", "find": '"default"',
                     "replace": '"other"'},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0063"]},
    },
]

SEMANTIC = [
    {
        "id": "sealed-semantic-01-floor-division",
        "category": "semantic-repair",
        "difficulty": "easy",
        "capability": "Average integers with a floor division for negative sums.",
        "prompt_intro": (
            "The helper builds and its visible test passes, but the contract floors a "
            "negative average: `mean_floor(&[-3, -2])` is -3, because -5/2 is -2.5 "
            "floored. An empty slice averages to 0. Fix the helper; keep the "
            "signature."),
        "observed_failure": (
            "the crate builds and tests pass; mean_floor(&[-3, -2]) is -2 where the "
            "contract requires -3"),
        "files": {
            "src/lib.rs": code("""
                //! Integer average with the standard division.

                pub fn mean_floor(values: &[i64]) -> i64 {
                    if values.is_empty() {
                        return 0;
                    }
                    let sum: i64 = values.iter().sum();
                    sum / values.len() as i64
                }

                #[cfg(test)]
                mod tests {
                    use super::mean_floor;

                    #[test]
                    fn visible_positive_average() {
                        assert_eq!(mean_floor(&[2, 4]), 3);
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use sealed_semantic_01_floor_division::mean_floor;

                #[test]
                fn a_negative_half_floors_down() {
                    assert_eq!(mean_floor(&[-3, -2]), -3);
                }

                #[test]
                fn a_negative_exact_average_is_unchanged() {
                    assert_eq!(mean_floor(&[-4, -2]), -3);
                }

                #[test]
                fn positive_values_behave_as_before() {
                    assert_eq!(mean_floor(&[2, 4]), 3);
                }

                #[test]
                fn an_empty_slice_is_zero() {
                    assert_eq!(mean_floor(&[]), 0);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Integer average with floor division.

                pub fn mean_floor(values: &[i64]) -> i64 {
                    if values.is_empty() {
                        return 0;
                    }
                    let sum: i64 = values.iter().sum();
                    sum.div_euclid(values.len() as i64)
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "sum.div_euclid(values.len() as i64)",
                     "replace": "sum / values.len() as i64"},
        "initial_failure": {"kind": "hidden_test_failure",
                            "must_contain": ["test result: FAILED"]},
    },
    {
        "id": "sealed-semantic-02-trim-both-ends",
        "category": "semantic-repair",
        "difficulty": "easy",
        "capability": "Trim both ends of a field, not only the leading whitespace.",
        "prompt_intro": (
            "The helper builds and its visible test passes, but the contract trims "
            "both ends: `cleaned(\" x \")` is `\"x\"`. Internal whitespace is kept. Fix "
            "the helper; keep the signature."),
        "observed_failure": (
            "the crate builds and tests pass; cleaned(\" x \") is \"x \" where the "
            "contract requires \"x\""),
        "files": {
            "src/lib.rs": code("""
                //! Field cleaning with a leading trim only.

                pub fn cleaned(text: &str) -> String {
                    text.trim_start().to_string()
                }

                #[cfg(test)]
                mod tests {
                    use super::cleaned;

                    #[test]
                    fn visible_leading_whitespace() {
                        assert_eq!(cleaned(" x"), "x");
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use sealed_semantic_02_trim_both_ends::cleaned;

                #[test]
                fn trailing_whitespace_is_trimmed() {
                    assert_eq!(cleaned(" x "), "x");
                }

                #[test]
                fn internal_whitespace_is_kept() {
                    assert_eq!(cleaned(" a b "), "a b");
                }

                #[test]
                fn tabs_and_newlines_are_trimmed() {
                    assert_eq!(cleaned("\\tx\\n"), "x");
                }

                #[test]
                fn empty_and_blank_inputs_are_empty() {
                    assert_eq!(cleaned(""), "");
                    assert_eq!(cleaned("   "), "");
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Field cleaning.

                pub fn cleaned(text: &str) -> String {
                    text.trim().to_string()
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "text.trim()", "replace": "text.trim_end()"},
        "initial_failure": {"kind": "hidden_test_failure",
                            "must_contain": ["test result: FAILED"]},
    },
    {
        "id": "sealed-semantic-03-percent-rounding",
        "category": "semantic-repair",
        "difficulty": "medium",
        "capability": "Round a percentage to the nearest integer, ties upward.",
        "prompt_intro": (
            "The helper builds and its visible test passes, but the contract rounds "
            "to the nearest integer with ties going up: `percent(2, 3)` is 67, not "
            "66. A total of zero is 0. Fix the helper; keep the signature."),
        "observed_failure": (
            "the crate builds and tests pass; percent(2, 3) is 66 where the contract "
            "requires 67"),
        "files": {
            "src/lib.rs": code("""
                //! Whole-number percentages with truncating division.

                pub fn percent(part: u64, total: u64) -> u64 {
                    if total == 0 {
                        return 0;
                    }
                    part * 100 / total
                }

                #[cfg(test)]
                mod tests {
                    use super::percent;

                    #[test]
                    fn visible_exact_half() {
                        assert_eq!(percent(1, 2), 50);
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use sealed_semantic_03_percent_rounding::percent;

                #[test]
                fn two_thirds_rounds_up() {
                    assert_eq!(percent(2, 3), 67);
                }

                #[test]
                fn one_third_rounds_down() {
                    assert_eq!(percent(1, 3), 33);
                }

                #[test]
                fn an_exact_half_rounds_up() {
                    assert_eq!(percent(1, 8), 13);
                }

                #[test]
                fn zero_total_and_zero_part() {
                    assert_eq!(percent(0, 5), 0);
                    assert_eq!(percent(5, 0), 0);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Whole-number percentages, rounded with ties upward.

                pub fn percent(part: u64, total: u64) -> u64 {
                    if total == 0 {
                        return 0;
                    }
                    (part * 100 + total / 2) / total
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "(part * 100 + total / 2) / total",
                     "replace": "part * 100 / total"},
        "initial_failure": {"kind": "hidden_test_failure",
                            "must_contain": ["test result: FAILED"]},
    },
]


SEALED_TASKS = (BORROW + FAILING + MIGRATION + DEPENDENCY + CONCURRENCY
                + TRAITS + REPO + SEMANTIC)
