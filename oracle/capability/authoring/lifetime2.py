"""Borrow-checker repairs, batch two (five tasks).

Each initial state is a real compile failure with the diagnostic named in
`initial_failure`; each reference changes ownership or lifetime structure rather
than silencing the compiler, and each has a declared mutation the hidden tests
must reject.
"""

from .common import code

TASKS = [
    {
        "id": "borrow-05-return-normalized",
        "category": "lifetime-borrow",
        "difficulty": "easy",
        "capability": "Return owned data for a string that is normalised locally.",
        "prompt_intro": (
            "This crate does not compile. `normalize` builds a normalised `String` "
            "and returns a reference to it. Change the signature to "
            "`pub fn normalize(text: &str) -> String`. Collapse every run of "
            "whitespace to one space and trim the ends."),
        "observed_failure": (
            "error[E0515]: cannot return value referencing local variable `normalized`\n"
            " --> src/lib.rs:3:5\n"
            "  |\n"
            "2 |     let normalized = text.split_whitespace().collect::<Vec<_>>().join(\" \");\n"
            "  |         ---------- `normalized` is borrowed here\n"
            "3 |     &normalized\n"
            "  |     ^^^^^^^^^^^ returns a value referencing data owned by the current function"),
        "files": {
            "src/lib.rs": code("""
                //! Whitespace normalisation. The crate does not build: it returns a
                //! reference to a `String` it creates locally.

                pub fn normalize(text: &str) -> &str {
                    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
                    &normalized
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use borrow_05_return_normalized::normalize;

                #[test]
                fn collapses_internal_runs() {
                    assert_eq!(normalize("  alpha   beta  "), "alpha beta");
                }

                #[test]
                fn handles_tabs_and_newlines() {
                    assert_eq!(normalize("a\\tb\\nc"), "a b c");
                }

                #[test]
                fn empty_and_blank_inputs_are_empty() {
                    assert_eq!(normalize(""), "");
                    assert_eq!(normalize("   \\t\\n"), "");
                }

                #[test]
                fn successive_calls_are_independent() {
                    let first = normalize("  one  ");
                    let second = normalize("two");
                    assert_eq!(first, "one");
                    assert_eq!(second, "two");
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Whitespace normalisation.

                pub fn normalize(text: &str) -> String {
                    text.split_whitespace().collect::<Vec<_>>().join(" ")
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": '.join(" ")', "replace": '.join("-")'},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0515"]},
    },
    {
        "id": "borrow-06-move-while-borrowed",
        "category": "lifetime-borrow",
        "difficulty": "easy",
        "capability": "Compute from a borrow before moving the owned value.",
        "prompt_intro": (
            "This crate does not compile. `split_first` moves the owned `String` "
            "while a slice of it is still borrowed in the same expression. Return "
            "the original string and the byte length of its first whitespace-"
            "delimited token. A string with no token reports length 0."),
        "observed_failure": (
            "error[E0505]: cannot move out of `text` because it is borrowed\n"
            " --> src/lib.rs:4:6\n"
            "  |\n"
            "3 |     let first = text.split_whitespace().next().unwrap_or(\"\");\n"
            "  |                 ---- borrow of `text` occurs here\n"
            "4 |     (text, first.len())\n"
            "  |      ^^^^  ----------- borrow later used here\n"
            "  |      move out of `text` occurs here"),
        "files": {
            "src/lib.rs": code("""
                //! Split the first token length from an owned string.

                pub fn split_first(text: String) -> (String, usize) {
                    let first = text.split_whitespace().next().unwrap_or("");
                    (text, first.len())
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use borrow_06_move_while_borrowed::split_first;

                #[test]
                fn returns_the_original_and_the_token_length() {
                    assert_eq!(split_first("hello world".to_string()), ("hello world".to_string(), 5));
                }

                #[test]
                fn leading_whitespace_is_not_part_of_the_token() {
                    assert_eq!(split_first("   hi there".to_string()), ("   hi there".to_string(), 2));
                }

                #[test]
                fn blank_input_has_length_zero() {
                    assert_eq!(split_first("   ".to_string()), ("   ".to_string(), 0));
                }

                #[test]
                fn empty_input_has_length_zero() {
                    assert_eq!(split_first(String::new()), (String::new(), 0));
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Split the first token length from an owned string.

                pub fn split_first(text: String) -> (String, usize) {
                    let len = text.split_whitespace().next().unwrap_or("").len();
                    (text, len)
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": 'unwrap_or("").len()',
                     "replace": 'unwrap_or("x").len()'},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0505"]},
    },
    {
        "id": "borrow-07-extend-while-iterating",
        "category": "lifetime-borrow",
        "difficulty": "hard",
        "capability": "Collect the additions before mutating the vector under iteration.",
        "prompt_intro": (
            "This crate does not compile. `double_evens` pushes into the vector "
            "while iterating it. Append the double of every even value that was "
            "present before the call; values appended by the call are not "
            "themselves doubled. Keep the signature and the original order."),
        "observed_failure": (
            "error[E0502]: cannot borrow `*values` as mutable because it is also\n"
            "borrowed as immutable\n"
            " --> src/lib.rs:4:13\n"
            "  |\n"
            "3 |     for value in values.iter() {\n"
            "  |                  ------ immutable borrow occurs here\n"
            "4 |         if value % 2 == 0 {\n"
            "  |             ^^^^^^^^^^^^ mutable borrow occurs here"),
        "files": {
            "src/lib.rs": code("""
                //! Double every even value, appending the doubles.

                pub fn double_evens(values: &mut Vec<i32>) {
                    for value in values.iter() {
                        if value % 2 == 0 {
                            values.push(*value * 2);
                        }
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use borrow_07_extend_while_iterating::double_evens;

                #[test]
                fn appends_doubles_in_order() {
                    let mut values = vec![1, 2, 3, 4];
                    double_evens(&mut values);
                    assert_eq!(values, vec![1, 2, 3, 4, 4, 8]);
                }

                #[test]
                fn appends_nothing_without_evens() {
                    let mut values = vec![1, 3, 5];
                    double_evens(&mut values);
                    assert_eq!(values, vec![1, 3, 5]);
                }

                #[test]
                fn negative_evens_count() {
                    let mut values = vec![-2, 1];
                    double_evens(&mut values);
                    assert_eq!(values, vec![-2, 1, -4]);
                }

                #[test]
                fn empty_stays_empty() {
                    let mut values: Vec<i32> = Vec::new();
                    double_evens(&mut values);
                    assert!(values.is_empty());
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Double every even value, appending the doubles.

                pub fn double_evens(values: &mut Vec<i32>) {
                    let additions: Vec<i32> = values
                        .iter()
                        .filter(|value| *value % 2 == 0)
                        .map(|value| value * 2)
                        .collect();
                    values.extend(additions);
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": ".filter(|value| *value % 2 == 0)",
                     "replace": ".filter(|value| *value % 2 != 0)"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0502"]},
    },
    {
        "id": "borrow-08-iterator-outlives-local",
        "category": "lifetime-borrow",
        "difficulty": "medium",
        "capability": "Recognise that a returned iterator cannot borrow a local collection.",
        "prompt_intro": (
            "This crate does not compile. `evens` creates a local `Vec` and returns "
            "an iterator borrowing it. Return an iterator over the even numbers in "
            "`0..limit` without allocating the collection. Keep the signature "
            "`pub fn evens(limit: i32) -> impl Iterator<Item = i32>`."),
        "observed_failure": (
            "error[E0597]: `values` does not live long enough\n"
            " --> src/lib.rs:5:5\n"
            "  |\n"
            "4 |     let values: Vec<i32> = (0..limit).collect();\n"
            "  |         ------ binding `values` declared here\n"
            "5 |     values.iter().copied().filter(|value| *value % 2 == 0)\n"
            "  |     ^^^^^^ borrowed value does not live long enough\n"
            "  |     opaque type requires that `values` is borrowed for `'static`"),
        "files": {
            "src/lib.rs": code("""
                //! Even numbers below a limit.

                pub fn evens(limit: i32) -> impl Iterator<Item = i32> {
                    let values: Vec<i32> = (0..limit).collect();
                    values.iter().copied().filter(|value| *value % 2 == 0)
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use borrow_08_iterator_outlives_local::evens;

                #[test]
                fn yields_the_even_numbers() {
                    assert_eq!(evens(10).collect::<Vec<_>>(), vec![0, 2, 4, 6, 8]);
                }

                #[test]
                fn zero_limit_is_empty() {
                    assert_eq!(evens(0).collect::<Vec<_>>(), Vec::<i32>::new());
                }

                #[test]
                fn one_limit_is_zero() {
                    assert_eq!(evens(1).collect::<Vec<_>>(), vec![0]);
                }

                #[test]
                fn negative_limit_is_empty() {
                    assert_eq!(evens(-5).collect::<Vec<_>>(), Vec::<i32>::new());
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Even numbers below a limit.

                pub fn evens(limit: i32) -> impl Iterator<Item = i32> {
                    (0..limit).filter(|value| value % 2 == 0)
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "value % 2 == 0", "replace": "value % 2 == 1"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0597"]},
    },
    {
        "id": "borrow-09-owned-excerpt",
        "category": "lifetime-borrow",
        "difficulty": "hard",
        "capability": "Replace a struct field that points at a local with an owned field.",
        "prompt_intro": (
            "This crate does not compile. `Excerpt` stores a `&'static str` that "
            "points at a `String` created inside `excerpt`. Make the field owned "
            "(`pub text: String`) and keep the public functions: "
            "`excerpt(source: &str) -> Excerpt` and `Excerpt::text(&self) -> &str`. "
            "The excerpt is the trimmed, upper-cased source."),
        "observed_failure": (
            "error[E0597]: `owned` does not live long enough\n"
            "  --> src/lib.rs:15:21\n"
            "   |\n"
            "14 |     let owned = source.trim().to_uppercase();\n"
            "   |         ----- binding `owned` declared here\n"
            "15 |     Excerpt { text: &owned }\n"
            "   |                     ^^^^^^ borrowed value does not live long enough\n"
            "   |     this usage requires that `owned` is borrowed for `'static`"),
        "files": {
            "src/lib.rs": code("""
                //! An excerpt that must own its text.

                pub struct Excerpt {
                    pub text: &'static str,
                }

                impl Excerpt {
                    pub fn text(&self) -> &str {
                        self.text
                    }
                }

                pub fn excerpt(source: &str) -> Excerpt {
                    let owned = source.trim().to_uppercase();
                    Excerpt { text: &owned }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use borrow_09_owned_excerpt::excerpt;

                #[test]
                fn trims_and_upper_cases() {
                    assert_eq!(excerpt("  hello  ").text(), "HELLO");
                }

                #[test]
                fn empty_source_is_empty() {
                    assert_eq!(excerpt("   ").text(), "");
                }

                #[test]
                fn excerpts_are_independent() {
                    let first = excerpt("one");
                    let second = excerpt("two");
                    assert_eq!(first.text(), "ONE");
                    assert_eq!(second.text(), "TWO");
                }

                #[test]
                fn the_field_is_owned_and_readable() {
                    let value = excerpt("mixed Case");
                    assert_eq!(value.text, "MIXED CASE");
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! An excerpt that owns its text.

                pub struct Excerpt {
                    pub text: String,
                }

                impl Excerpt {
                    pub fn text(&self) -> &str {
                        &self.text
                    }
                }

                pub fn excerpt(source: &str) -> Excerpt {
                    Excerpt { text: source.trim().to_uppercase() }
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "source.trim().to_uppercase()",
                     "replace": "source.trim().to_lowercase()"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0597"]},
    },
]
