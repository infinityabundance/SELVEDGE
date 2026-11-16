"""Replacements for tasks that duplicated an existing capability.

The first expansion pass knowingly produced pairs that shared the same fix
(e.g. two "return an owned String" lifetime tasks). The quality bar rejects
near-duplicates, so these sixteen declarations supersede the listed ids in
`authoring/__init__.py`. The superseded declarations are not materialised and do
not enter `SUITE_TASKS`; the replacement is a different diagnostic, different
code, and a different reference.
"""

from .common import code


def _app(name: str, dependency: str) -> str:
    return code(f"""
        [package]
        name = "{name}"
        version = "0.1.0"
        edition = "2021"

        [workspace]
        members = [".", "dep"]

        [lib]
        path = "src/lib.rs"

        [dependencies]
        {dependency}
    """)


def _dep(name: str, features: str = "") -> str:
    body = f"""
        [package]
        name = "{name}"
        version = "2.0.0"
        edition = "2021"
    """
    if features:
        body += features
    body += """
        [lib]
        path = "src/lib.rs"
    """
    return code(body)


TASKS = [
    {
        "id": "borrow-05-temporary-lifetime",
        "category": "lifetime-borrow",
        "difficulty": "medium",
        "capability": "Bind a temporary before borrowing it so the borrow outlives the statement.",
        "prompt_intro": (
            "This crate does not compile. `label` borrows a `format!` temporary "
            "through `first_word` and then uses the result on the next line. Bind "
            "the formatted string first so its borrow is valid. Keep the signatures "
            "and the `item <number>` wording."),
        "observed_failure": (
            "error[E0716]: temporary value dropped while borrowed\n"
            " --> src/lib.rs:8:23\n"
            "  |\n"
            "8 |     let word = first_word(&format!(\"item {number}\"));\n"
            "  |                       ^^^^^^^^^^^^^^^^^^^^^^^ creates a temporary value which is freed while still in use\n"
            "9 |     word.to_string()\n"
            "  |     ---- borrow later used here"),
        "files": {
            "src/lib.rs": code("""
                //! A label built from a borrowed temporary.

                pub fn first_word(text: &str) -> &str {
                    text.split_whitespace().next().unwrap_or("")
                }

                pub fn label(number: u32) -> String {
                    let word = first_word(&format!("item {number}"));
                    word.to_string()
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use borrow_05_temporary_lifetime::label;

                #[test]
                fn labels_the_first_word() {
                    assert_eq!(label(3), "item");
                }

                #[test]
                fn zero_is_still_labelled() {
                    assert_eq!(label(0), "item");
                }

                #[test]
                fn the_number_only_changes_the_temporary() {
                    assert_eq!(label(42), "item");
                }

                #[test]
                fn successive_labels_are_independent() {
                    assert_eq!(label(1), "item");
                    assert_eq!(label(2), "item");
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! A label built from a borrowed temporary.

                pub fn first_word(text: &str) -> &str {
                    text.split_whitespace().next().unwrap_or("")
                }

                pub fn label(number: u32) -> String {
                    let formatted = format!("item {number}");
                    first_word(&formatted).to_string()
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": 'format!("item {number}")',
                     "replace": 'format!("item-{number}")'},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0716"]},
    },
    {
        "id": "borrow-07-overlapping-mut-borrows",
        "category": "lifetime-borrow",
        "difficulty": "medium",
        "capability": "End one mutable borrow before starting the next one.",
        "prompt_intro": (
            "This crate does not compile. `bump_twice` holds two mutable borrows of "
            "the same field at once. Use each borrow to completion before taking the "
            "next. Keep the public API and make `bump_twice` add two."),
        "observed_failure": (
            "error[E0499]: cannot borrow `self.value` as mutable more than once at a time\n"
            "  --> src/lib.rs:12:22\n"
            "   |\n"
            "11 |         let first = self.value_mut();\n"
            "   |                     ---------------- first mutable borrow occurs here\n"
            "12 |         let second = self.value_mut();\n"
            "   |                      ^^^^^^^^^^^^^^^^ second mutable borrow occurs here"),
        "files": {
            "src/lib.rs": code("""
                //! A counter with two overlapping mutable borrows.

                pub struct Counter {
                    value: i32,
                }

                impl Counter {
                    pub fn new(value: i32) -> Self {
                        Counter { value }
                    }

                    pub fn value(&self) -> i32 {
                        self.value
                    }

                    pub fn value_mut(&mut self) -> &mut i32 {
                        &mut self.value
                    }

                    pub fn bump_twice(&mut self) {
                        let first = self.value_mut();
                        let second = self.value_mut();
                        *first += 1;
                        *second += 1;
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use borrow_07_overlapping_mut_borrows::Counter;

                #[test]
                fn bumps_a_fresh_counter_twice() {
                    let mut counter = Counter::new(0);
                    counter.bump_twice();
                    assert_eq!(counter.value(), 2);
                }

                #[test]
                fn bumps_from_the_current_value() {
                    let mut counter = Counter::new(5);
                    counter.bump_twice();
                    assert_eq!(counter.value(), 7);
                }

                #[test]
                fn repeated_calls_accumulate() {
                    let mut counter = Counter::new(0);
                    counter.bump_twice();
                    counter.bump_twice();
                    counter.bump_twice();
                    assert_eq!(counter.value(), 6);
                }

                #[test]
                fn the_mutable_accessor_still_works() {
                    let mut counter = Counter::new(1);
                    *counter.value_mut() = 9;
                    assert_eq!(counter.value(), 9);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! A counter with sequential mutable borrows.

                pub struct Counter {
                    value: i32,
                }

                impl Counter {
                    pub fn new(value: i32) -> Self {
                        Counter { value }
                    }

                    pub fn value(&self) -> i32 {
                        self.value
                    }

                    pub fn value_mut(&mut self) -> &mut i32 {
                        &mut self.value
                    }

                    pub fn bump_twice(&mut self) {
                        let first = self.value_mut();
                        *first += 1;
                        let second = self.value_mut();
                        *second += 1;
                    }
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "*first += 1;", "replace": "*first += 2;"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0499"]},
    },
    {
        "id": "borrow-09-struct-lifetime",
        "category": "lifetime-borrow",
        "difficulty": "medium",
        "capability": "Add a lifetime parameter to a borrowing struct and its impl.",
        "prompt_intro": (
            "This crate does not compile. `View` stores a `&str` without a lifetime "
            "parameter. Add the lifetime to the struct, the impl and the constructor "
            "so a `View` can borrow from a caller's `String`. Keep the API "
            "`View::new(&str)`, `text(&self) -> &str`."),
        "observed_failure": (
            "error[E0106]: missing lifetime specifier\n"
            " --> src/lib.rs:3:24\n"
            "  |\n"
            "3 |     pub text: &str,\n"
            "  |              ^ expected named lifetime parameter"),
        "files": {
            "src/lib.rs": code("""
                //! A view that borrows text but has no lifetime parameter.

                pub struct View {
                    pub text: &str,
                }

                impl View {
                    pub fn new(text: &str) -> View {
                        View { text }
                    }

                    pub fn text(&self) -> &str {
                        self.text
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use borrow_09_struct_lifetime::View;

                #[test]
                fn borrows_a_static_literal() {
                    assert_eq!(View::new("alpha").text(), "alpha");
                }

                #[test]
                fn borrows_an_owned_string() {
                    let owned = String::from("beta");
                    let view = View::new(&owned);
                    assert_eq!(view.text(), "beta");
                }

                #[test]
                fn empty_text_is_empty() {
                    assert_eq!(View::new("").text(), "");
                }

                #[test]
                fn two_views_are_independent() {
                    let first = String::from("one");
                    let second = String::from("two");
                    assert_eq!(View::new(&first).text(), "one");
                    assert_eq!(View::new(&second).text(), "two");
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! A view with an explicit lifetime.

                pub struct View<'a> {
                    pub text: &'a str,
                }

                impl<'a> View<'a> {
                    pub fn new(text: &'a str) -> View<'a> {
                        View { text }
                    }

                    pub fn text(&self) -> &str {
                        self.text
                    }
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "self.text", "replace": "\"x\""},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0106"]},
    },
    {
        "id": "failing-07-common-prefix",
        "category": "failing-test",
        "difficulty": "medium",
        "capability": "Replace a shortest-length shortcut with a real common-prefix scan.",
        "prompt_intro": (
            "The visible unit test fails. `common_prefix` must return the longest "
            "string that is a prefix of every input. An empty slice, or inputs with "
            "no common first character, return an empty string. Keep the signature "
            "`pub fn common_prefix(values: &[&str]) -> String`."),
        "observed_failure": (
            "thread 'tests::prefix_of_three_words' panicked at src/lib.rs:16:9:\n"
            "assertion `left == right` failed\n"
            "  left: \"flow\"\n"
            " right: \"fl\"\n"
            "\n"
            "test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out"),
        "files": {
            "src/lib.rs": code("""
                //! A common-prefix helper that only truncates the first string.

                pub fn common_prefix(values: &[&str]) -> String {
                    if values.is_empty() {
                        return String::new();
                    }
                    let shortest = values.iter().map(|value| value.len()).min().unwrap_or(0);
                    values[0][..shortest].to_string()
                }

                #[cfg(test)]
                mod tests {
                    use super::common_prefix;

                    #[test]
                    fn prefix_of_three_words() {
                        assert_eq!(common_prefix(&["flower", "flow", "flight"]), "fl");
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use failing_07_common_prefix::common_prefix;

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
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Longest common prefix.

                pub fn common_prefix(values: &[&str]) -> String {
                    let Some(first) = values.first() else {
                        return String::new();
                    };
                    let mut prefix = first.to_string();
                    for value in &values[1..] {
                        while !value.starts_with(&prefix) {
                            prefix.pop();
                            if prefix.is_empty() {
                                return prefix;
                            }
                        }
                    }
                    prefix
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "let mut prefix = first.to_string();",
                     "replace": "let mut prefix = first.to_uppercase();"},
        "initial_failure": {"kind": "test_failure", "must_contain": ["test result: FAILED"]},
    },
    {
        "id": "migrate-08-path-api",
        "category": "api-migration",
        "difficulty": "medium",
        "capability": "Migrate a call site to a dependency that now takes a path type.",
        "prompt_intro": (
            "The crate does not build against the vendored `storage` 2.0.0. Version 1 "
            "took `&str`; version 2 takes `&std::path::Path`. Do not modify `dep/`. "
            "Keep `pub fn stem_of(name: &str) -> Option<String>`; the returned stem "
            "is the file name without its final extension."),
        "observed_failure": (
            "error[E0308]: mismatched types\n"
            " --> src/lib.rs:3:19\n"
            "  |\n"
            "3 |     storage::stem(name)\n"
            "  |                  ^^^^ expected `&Path`, found `&str`"),
        "manifest": _app("migrate_08_path_api", 'storage = { path = "dep" }'),
        "files": {
            "dep/Cargo.toml": _dep("storage"),
            "dep/src/lib.rs": code("""
                //! Path stems, version 2. The API takes `&Path`.

                use std::path::Path;

                pub fn stem(path: &Path) -> Option<String> {
                    path.file_stem().map(|stem| stem.to_string_lossy().into_owned())
                }
            """),
            "src/lib.rs": code("""
                //! File stems through the vendored crate. This does not build: the
                //! API now takes `&Path`.

                pub fn stem_of(name: &str) -> Option<String> {
                    storage::stem(name)
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use migrate_08_path_api::stem_of;

                #[test]
                fn takes_the_stem_of_a_path() {
                    assert_eq!(stem_of("a/b.txt"), Some("b".to_string()));
                }

                #[test]
                fn a_bare_name_is_its_own_stem() {
                    assert_eq!(stem_of("file"), Some("file".to_string()));
                }

                #[test]
                fn only_the_last_extension_is_removed() {
                    assert_eq!(stem_of("a/b.tar.gz"), Some("b.tar".to_string()));
                }

                #[test]
                fn a_root_has_no_stem() {
                    assert_eq!(stem_of("/"), None);
                }
            """),
        },
        "readonly": ["dep/"],
        "reference": {
            "src/lib.rs": code("""
                //! File stems through the vendored crate.

                use std::path::Path;

                pub fn stem_of(name: &str) -> Option<String> {
                    storage::stem(Path::new(name))
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "Path::new(name)",
                     "replace": "Path::new(\"fixed\")"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0308"]},
    },
    {
        "id": "deps-04-unknown-feature",
        "category": "dependency-compat",
        "difficulty": "easy",
        "capability": "Diagnose a misspelled feature name that the dependency does not declare.",
        "prompt_intro": (
            "This crate does not build. The manifest asks the vendored `codec` 2.0.0 "
            "for feature `jason`, but the crate declares `json`. Correct the feature "
            "name; do not modify `dep/`. Keep `pub fn encode(value: &str) -> String`."),
        "observed_failure": (
            "error: failed to select a version for `codec`.\n"
            "the package depends on `codec`, with features: `jason` but `codec` does\n"
            "not have these features."),
        "manifest_policy": "arm",
        "manifest": _app("deps_04_unknown_feature",
                         'codec = { path = "dep", features = ["jason"] }'),
        "files": {
            "dep/Cargo.toml": _dep("codec", """
                [features]
                default = []
                json = []
            """),
            "dep/src/lib.rs": code("""
                //! Encoding, version 2. JSON support is behind the `json` feature.

                #[cfg(feature = "json")]
                pub mod json {
                    pub fn to_json(value: &str) -> String {
                        format!("\\"{value}\\"")
                    }
                }
            """),
            "src/lib.rs": code("""
                //! Encode through the vendored crate.

                pub fn encode(value: &str) -> String {
                    codec::json::to_json(value)
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use deps_04_unknown_feature::encode;

                #[test]
                fn wraps_a_value_in_quotes() {
                    assert_eq!(encode("alpha"), "\\"alpha\\"");
                }

                #[test]
                fn empty_value_gets_quotes() {
                    assert_eq!(encode(""), "\\"\\"");
                }

                #[test]
                fn spacing_is_preserved() {
                    assert_eq!(encode("a b"), "\\"a b\\"");
                }
            """),
        },
        "readonly": ["dep/"],
        "reference": {
            "Cargo.toml": _app("deps_04_unknown_feature",
                               'codec = { path = "dep", features = ["json"] }'),
            "src/lib.rs": code("""
                //! Encode through the vendored crate.

                pub fn encode(value: &str) -> String {
                    codec::json::to_json(value)
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "codec::json::to_json(value)",
                     "replace": "codec::json::to_json(value).to_uppercase()"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["jason"]},
    },
    {
        "id": "deps-05-workspace-member",
        "category": "dependency-compat",
        "difficulty": "easy",
        "capability": "Remove a workspace member that has no directory, without touching the dependency.",
        "prompt_intro": (
            "This crate does not build. The workspace lists a member `missing-tool` "
            "that has no directory. Remove that member from the workspace; keep the "
            "`codec` dependency and do not modify `dep/`. Keep "
            "`pub fn encode(value: &str) -> String`."),
        "observed_failure": (
            "error: failed to load manifest for workspace member `/work/.../missing-tool`\n"
            "Caused by:\n"
            "  failed to read `/work/.../missing-tool/Cargo.toml`\n"
            "Caused by:\n"
            "  No such file or directory (os error 2)"),
        "manifest_policy": "arm",
        "manifest": code("""
            [package]
            name = "deps_05_workspace_member"
            version = "0.1.0"
            edition = "2021"

            [workspace]
            members = [".", "dep", "missing-tool"]

            [lib]
            path = "src/lib.rs"

            [dependencies]
            codec = { path = "dep" }
        """),
        "files": {
            "dep/Cargo.toml": _dep("codec"),
            "dep/src/lib.rs": code("""
                //! Encoding, version 2.

                pub fn to_json(value: &str) -> String {
                    format!("\\"{value}\\"")
                }
            """),
            "src/lib.rs": code("""
                //! Encode through the vendored crate.

                pub fn encode(value: &str) -> String {
                    codec::to_json(value)
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use deps_05_workspace_member::encode;

                #[test]
                fn wraps_a_value_in_quotes() {
                    assert_eq!(encode("alpha"), "\\"alpha\\"");
                }

                #[test]
                fn empty_value_gets_quotes() {
                    assert_eq!(encode(""), "\\"\\"");
                }

                #[test]
                fn spacing_is_preserved() {
                    assert_eq!(encode("a b"), "\\"a b\\"");
                }
            """),
        },
        "readonly": ["dep/"],
        "reference": {
            "Cargo.toml": code("""
                [package]
                name = "deps_05_workspace_member"
                version = "0.1.0"
                edition = "2021"

                [workspace]
                members = [".", "dep"]

                [lib]
                path = "src/lib.rs"

                [dependencies]
                codec = { path = "dep" }
            """),
            "src/lib.rs": code("""
                //! Encode through the vendored crate.

                pub fn encode(value: &str) -> String {
                    codec::to_json(value)
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "codec::to_json(value)",
                     "replace": "codec::to_json(value).to_uppercase()"},
        "initial_failure": {"kind": "compile_error",
                            "must_contain": ["missing-tool"]},
    },
    {
        "id": "deps-06-edition-unsafe-block",
        "category": "dependency-compat",
        "difficulty": "medium",
        "capability": "Wrap an unsafe operation in an unsafe block under edition 2024.",
        "prompt_intro": (
            "This crate does not build under edition 2024. It denies "
            "`unsafe_op_in_unsafe_fn`, and edition 2024 requires an explicit `unsafe` "
            "block inside an `unsafe fn` for each unsafe operation. Wrap the "
            "dereference; keep the signature, the safety comment, the lint and the "
            "edition."),
        "observed_failure": (
            "error[E0133]: dereference of raw pointer is unsafe and requires unsafe block\n"
            " --> src/lib.rs:5:5\n"
            "  |\n"
            "5 |     *pointer\n"
            "  |     ^^^^^^^^ dereference of raw pointer\n"
            "  |\n"
            "  = note: unsafe function bodies are now required to use an explicit unsafe block"),
        "manifest": code("""
            [package]
            name = "deps_06_edition_unsafe_block"
            version = "0.1.0"
            edition = "2024"

            [workspace]

            [lib]
            path = "src/lib.rs"
        """),
        "files": {
            "src/lib.rs": code("""
                #![deny(unsafe_op_in_unsafe_fn)]

                //! Raw-pointer access under edition 2024.

                /// # Safety
                ///
                /// `pointer` must be valid for one read.
                pub unsafe fn read_byte(pointer: *const u8) -> u8 {
                    *pointer
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use deps_06_edition_unsafe_block::read_byte;

                #[test]
                fn reads_a_value() {
                    let value = 42u8;
                    assert_eq!(unsafe { read_byte(&value as *const u8) }, 42);
                }

                #[test]
                fn reads_zero() {
                    let value = 0u8;
                    assert_eq!(unsafe { read_byte(&value as *const u8) }, 0);
                }

                #[test]
                fn reads_the_maximum_byte() {
                    let value = 255u8;
                    assert_eq!(unsafe { read_byte(&value as *const u8) }, 255);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                #![deny(unsafe_op_in_unsafe_fn)]

                //! Raw-pointer access under edition 2024.

                /// # Safety
                ///
                /// `pointer` must be valid for one read.
                pub unsafe fn read_byte(pointer: *const u8) -> u8 {
                    unsafe { *pointer }
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "unsafe { *pointer }",
                     "replace": "unsafe { *pointer + 1 }"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0133"]},
    },
    {
        "id": "concurrency-04-mutex-critical-section",
        "category": "concurrency",
        "difficulty": "medium",
        "capability": "Hold a mutex across the whole read-modify-write, not across a barrier.",
        "prompt_intro": (
            "The hidden test fails. Two threads read the mutex-protected counter, meet "
            "at a barrier, and then each writes `read + 1`, so one update is lost. "
            "Move the barrier before the critical section and perform the increment "
            "under a single lock. Keep the signature and the barrier."),
        "observed_failure": (
            "thread 'both_threads_increment' panicked at tests/check.rs:\n"
            "assertion `left == right` failed\n"
            "  left: 1\n"
            " right: 2"),
        "files": {
            "src/lib.rs": code("""
                //! Two synchronized increments with a mutex, and a lost update.

                use std::sync::{Arc, Barrier, Mutex};
                use std::thread;

                pub fn increment_pair(counter: &Arc<Mutex<usize>>) {
                    let barrier = Arc::new(Barrier::new(2));
                    thread::scope(|scope| {
                        for _ in 0..2 {
                            let counter = Arc::clone(counter);
                            let barrier = Arc::clone(&barrier);
                            scope.spawn(move || {
                                let current = *counter.lock().unwrap();
                                barrier.wait();
                                *counter.lock().unwrap() = current + 1;
                            });
                        }
                    });
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use concurrency_04_mutex_critical_section::increment_pair;
                use std::sync::{Arc, Mutex};

                #[test]
                fn both_threads_increment() {
                    let counter = Arc::new(Mutex::new(0));
                    increment_pair(&counter);
                    assert_eq!(*counter.lock().unwrap(), 2);
                }

                #[test]
                fn the_result_is_stable_across_runs() {
                    for _ in 0..20 {
                        let counter = Arc::new(Mutex::new(0));
                        increment_pair(&counter);
                        assert_eq!(*counter.lock().unwrap(), 2);
                    }
                }

                #[test]
                fn increments_from_the_current_value() {
                    let counter = Arc::new(Mutex::new(41));
                    increment_pair(&counter);
                    assert_eq!(*counter.lock().unwrap(), 43);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Two synchronized increments under one lock.

                use std::sync::{Arc, Barrier, Mutex};
                use std::thread;

                pub fn increment_pair(counter: &Arc<Mutex<usize>>) {
                    let barrier = Arc::new(Barrier::new(2));
                    thread::scope(|scope| {
                        for _ in 0..2 {
                            let counter = Arc::clone(counter);
                            let barrier = Arc::clone(&barrier);
                            scope.spawn(move || {
                                barrier.wait();
                                *counter.lock().unwrap() += 1;
                            });
                        }
                    });
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "*counter.lock().unwrap() += 1;",
                     "replace": "*counter.lock().unwrap() += 2;"},
        "initial_failure": {"kind": "test_failure", "must_contain": ["test result: FAILED"]},
    },
    {
        "id": "traits-06-method-ambiguity",
        "category": "trait-impl",
        "difficulty": "medium",
        "capability": "Disambiguate a method provided by two traits.",
        "prompt_intro": (
            "This crate does not compile. `Item` implements two traits that both "
            "provide `value`, so `item.value()` is ambiguous. Use fully-qualified "
            "trait syntax in `left`; keep `right` and the trait definitions."),
        "observed_failure": (
            "error[E0034]: multiple applicable items in scope\n"
            " --> src/lib.rs:15:25\n"
            "  |\n"
            "15 | pub fn left(item: &Item) -> u32 { item.value() }\n"
            "  |                                       ^^^^^ multiple `value` found"),
        "files": {
            "src/lib.rs": code("""
                //! Two traits with the same method.

                pub trait Left {
                    fn value(&self) -> u32;
                }

                pub trait Right {
                    fn value(&self) -> u32;
                }

                pub struct Item(pub u32);

                impl Left for Item {
                    fn value(&self) -> u32 {
                        self.0
                    }
                }

                impl Right for Item {
                    fn value(&self) -> u32 {
                        self.0 + 100
                    }
                }

                pub fn left(item: &Item) -> u32 {
                    item.value()
                }

                pub fn right(item: &Item) -> u32 {
                    Right::value(item)
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use traits_06_method_ambiguity::{left, right, Item};

                #[test]
                fn left_uses_the_left_trait() {
                    assert_eq!(left(&Item(1)), 1);
                }

                #[test]
                fn right_uses_the_right_trait() {
                    assert_eq!(right(&Item(1)), 101);
                }

                #[test]
                fn the_traits_are_independent() {
                    let item = Item(7);
                    assert_eq!(left(&item), 7);
                    assert_eq!(right(&item), 107);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Two traits with the same method.

                pub trait Left {
                    fn value(&self) -> u32;
                }

                pub trait Right {
                    fn value(&self) -> u32;
                }

                pub struct Item(pub u32);

                impl Left for Item {
                    fn value(&self) -> u32 {
                        self.0
                    }
                }

                impl Right for Item {
                    fn value(&self) -> u32 {
                        self.0 + 100
                    }
                }

                pub fn left(item: &Item) -> u32 {
                    Left::value(item)
                }

                pub fn right(item: &Item) -> u32 {
                    Right::value(item)
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "Left::value(item)",
                     "replace": "Right::value(item)"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0034"]},
    },
    {
        "id": "traits-07-add-impl",
        "category": "trait-impl",
        "difficulty": "easy",
        "capability": "Implement the operator trait a newtype needs to participate in arithmetic.",
        "prompt_intro": (
            "This crate does not compile. `Money` cannot be added because it does not "
            "implement `std::ops::Add`. Implement it for `Money + Money`; keep the "
            "newtype and `total`."),
        "observed_failure": (
            "error[E0369]: cannot add `Money` to `Money`\n"
            " --> src/lib.rs:5:44\n"
            "  |\n"
            "5 | pub fn total(left: Money, right: Money) -> Money { left + right }\n"
            "  |                                                    ^^^^ ------- Money\n"
            "  |                                                    |\n"
            "  |                                                    Money"),
        "files": {
            "src/lib.rs": code("""
                //! Money arithmetic without the Add impl.

                #[derive(Debug, PartialEq)]
                pub struct Money(pub u64);

                pub fn total(left: Money, right: Money) -> Money {
                    left + right
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use traits_07_add_impl::{Money, total};

                #[test]
                fn adds_two_values() {
                    assert_eq!(total(Money(2), Money(3)), Money(5));
                }

                #[test]
                fn zero_is_the_identity() {
                    assert_eq!(total(Money(0), Money(7)), Money(7));
                    assert_eq!(total(Money(7), Money(0)), Money(7));
                }

                #[test]
                fn large_values_are_added() {
                    assert_eq!(total(Money(1_000_000), Money(2_000_000)), Money(3_000_000));
                }

                #[test]
                fn the_operator_is_available_directly() {
                    assert_eq!(Money(1) + Money(2), Money(3));
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Money arithmetic.

                use std::ops::Add;

                #[derive(Debug, PartialEq)]
                pub struct Money(pub u64);

                impl Add for Money {
                    type Output = Money;

                    fn add(self, right: Money) -> Money {
                        Money(self.0 + right.0)
                    }
                }

                pub fn total(left: Money, right: Money) -> Money {
                    left + right
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "Money(self.0 + right.0)",
                     "replace": "Money(self.0 + right.0 + 1)"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0369"]},
    },
    {
        "id": "traits-08-method-signature",
        "category": "trait-impl",
        "difficulty": "medium",
        "capability": "Match a trait method's parameter type in its implementation.",
        "prompt_intro": (
            "This crate does not compile. `Hex::encode` takes `u64`, but the trait "
            "declares `u32`. Correct the implementation's signature; keep `Hex`, the "
            "trait and the lowercase hex output."),
        "observed_failure": (
            "error[E0053]: method `encode` has an incompatible type for trait\n"
            " --> src/lib.rs:9:26\n"
            "  |\n"
            "9 |     fn encode(&self, value: u64) -> String {\n"
            "  |                          ^^^ expected `u32`, found `u64`"),
        "files": {
            "src/lib.rs": code("""
                //! A hex codec whose impl signature does not match the trait.

                pub trait Codec {
                    fn encode(&self, value: u32) -> String;
                }

                pub struct Hex;

                impl Codec for Hex {
                    fn encode(&self, value: u64) -> String {
                        format!("{value:x}")
                    }
                }

                pub fn encode_all<T: Codec>(codec: &T, values: &[u32]) -> Vec<String> {
                    values.iter().map(|value| codec.encode(*value)).collect()
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use traits_08_method_signature::{encode_all, Codec, Hex};

                #[test]
                fn encodes_the_maximum_byte() {
                    assert_eq!(Hex.encode(255), "ff");
                }

                #[test]
                fn encodes_through_the_generic_helper() {
                    assert_eq!(encode_all(&Hex, &[1, 15, 255]), vec!["1", "f", "ff"]);
                }

                #[test]
                fn zero_encodes_to_one_zero() {
                    assert_eq!(Hex.encode(0), "0");
                }

                #[test]
                fn an_empty_slice_is_empty() {
                    assert!(encode_all(&Hex, &[]).is_empty());
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! A hex codec.

                pub trait Codec {
                    fn encode(&self, value: u32) -> String;
                }

                pub struct Hex;

                impl Codec for Hex {
                    fn encode(&self, value: u32) -> String {
                        format!("{value:x}")
                    }
                }

                pub fn encode_all<T: Codec>(codec: &T, values: &[u32]) -> Vec<String> {
                    values.iter().map(|value| codec.encode(*value)).collect()
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": 'format!("{value:x}")',
                     "replace": 'format!("{value:X}")'},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0053"]},
    },
    {
        "id": "semantic-01-overlapping-count",
        "category": "semantic-repair",
        "difficulty": "medium",
        "capability": "Count overlapping occurrences where a non-overlapping matcher passes the visible case.",
        "prompt_intro": (
            "The helper builds and its visible test passes, but the contract counts "
            "**overlapping** occurrences: `count_occurrences(\"aaaa\", \"aa\")` is 3, "
            "not 2. An empty needle counts zero. Inputs are ASCII. Fix the helper; "
            "keep the signature."),
        "observed_failure": (
            "the crate builds and tests pass; count_occurrences(\"aaaa\", \"aa\") is 2\n"
            "where the contract requires 3 overlapping matches"),
        "files": {
            "src/lib.rs": code("""
                //! Occurrence counting with the standard non-overlapping matcher.

                pub fn count_occurrences(haystack: &str, needle: &str) -> usize {
                    haystack.matches(needle).count()
                }

                #[cfg(test)]
                mod tests {
                    use super::count_occurrences;

                    #[test]
                    fn visible_example_is_non_overlapping() {
                        assert_eq!(count_occurrences("aaaa", "aa"), 2);
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use semantic_01_overlapping_count::count_occurrences;

                #[test]
                fn counts_overlapping_matches() {
                    assert_eq!(count_occurrences("aaaa", "aa"), 3);
                }

                #[test]
                fn counts_every_aba_in_ababa() {
                    assert_eq!(count_occurrences("ababa", "aba"), 2);
                }

                #[test]
                fn counts_single_characters() {
                    assert_eq!(count_occurrences("aaa", "a"), 3);
                }

                #[test]
                fn no_match_is_zero() {
                    assert_eq!(count_occurrences("abc", "d"), 0);
                }

                #[test]
                fn an_empty_needle_is_zero() {
                    assert_eq!(count_occurrences("abc", ""), 0);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Overlapping occurrence counting.

                pub fn count_occurrences(haystack: &str, needle: &str) -> usize {
                    if needle.is_empty() {
                        return 0;
                    }
                    haystack
                        .as_bytes()
                        .windows(needle.len())
                        .filter(|window| *window == needle.as_bytes())
                        .count()
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "*window == needle.as_bytes()",
                     "replace": "*window != needle.as_bytes()"},
        "initial_failure": {"kind": "hidden_test_failure",
                            "must_contain": ["test result: FAILED"]},
    },
    {
        "id": "semantic-05-lowercase-hex",
        "category": "semantic-repair",
        "difficulty": "easy",
        "capability": "Use the documented lowercase hex alphabet where digits-only examples pass.",
        "prompt_intro": (
            "The hex encoder builds and its visible test passes because it uses only "
            "digits. The contract is lowercase hexadecimal with two digits per byte: "
            "`hex(&[0xab, 0x0f])` is `ab0f`. Fix the encoder; keep the signature."),
        "observed_failure": (
            "the crate builds and tests pass; hex(&[0xab, 0x0f]) is \"AB0F\" where the\n"
            "contract requires lowercase \"ab0f\""),
        "files": {
            "src/lib.rs": code("""
                //! Hex encoding with an upper-case format specifier.

                pub fn hex(bytes: &[u8]) -> String {
                    bytes.iter().map(|byte| format!("{byte:02X}")).collect()
                }

                #[cfg(test)]
                mod tests {
                    use super::hex;

                    #[test]
                    fn digits_only_are_the_same_in_either_case() {
                        assert_eq!(hex(&[1, 2]), "0102");
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use semantic_05_lowercase_hex::hex;

                #[test]
                fn uses_lowercase_letters() {
                    assert_eq!(hex(&[0xab, 0x0f]), "ab0f");
                }

                #[test]
                fn encodes_a_word() {
                    assert_eq!(hex(&[0xde, 0xad]), "dead");
                }

                #[test]
                fn empty_is_empty() {
                    assert_eq!(hex(&[]), "");
                }

                #[test]
                fn pads_every_byte_to_two_digits() {
                    assert_eq!(hex(&[0, 1, 255]), "0001ff");
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Lowercase hex encoding.

                pub fn hex(bytes: &[u8]) -> String {
                    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": 'format!("{byte:02x}")',
                     "replace": 'format!("{byte:02X}")'},
        "initial_failure": {"kind": "hidden_test_failure",
                            "must_contain": ["test result: FAILED"]},
    },
    {
        "id": "repo-06-moved-module-import",
        "category": "repo-scale",
        "difficulty": "medium",
        "capability": "Propagate a module move through every importing module.",
        "prompt_intro": (
            "This crate does not compile. `load` moved from a removed `storage` "
            "module to `store`, but `service` and `report` still import the old path. "
            "Update both imports; do not add a compatibility module and do not change "
            "the behaviour."),
        "observed_failure": (
            "error[E0432]: unresolved import `crate::storage`\n"
            " --> src/service.rs:1:12\n"
            "  |\n"
            "1 | use crate::storage::load;\n"
            "  |            ^^^^^^^ could not find `storage` in the crate root"),
        "files": {
            "src/lib.rs": code("""
                //! Loading through a store.

                pub mod report;
                pub mod service;
                pub mod store;

                pub use service::fetch;
            """),
            "src/store.rs": code("""
                //! Store reads.

                pub fn load(key: &str, data: &[(&str, i32)]) -> Option<i32> {
                    data.iter()
                        .find(|(name, _value)| *name == key)
                        .map(|(_name, value)| *value)
                }
            """),
            "src/service.rs": code("""
                //! Service reads.

                use crate::storage::load;

                pub fn fetch(key: &str, data: &[(&str, i32)]) -> Option<i32> {
                    load(key, data).filter(|value| *value > 0)
                }
            """),
            "src/report.rs": code("""
                //! Report formatting.

                use crate::storage::load;

                pub fn describe(key: &str, data: &[(&str, i32)]) -> String {
                    match load(key, data) {
                        Some(value) => format!("{key}={value}"),
                        None => format!("{key}=missing"),
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use repo_06_moved_module_import::{fetch, report::describe};

                fn data() -> Vec<(&'static str, i32)> {
                    vec![("a", 2), ("b", -1), ("zero", 0)]
                }

                #[test]
                fn fetches_a_positive_value() {
                    assert_eq!(fetch("a", &data()), Some(2));
                }

                #[test]
                fn filters_a_non_positive_value() {
                    assert_eq!(fetch("b", &data()), None);
                    assert_eq!(fetch("zero", &data()), None);
                    assert_eq!(fetch("missing", &data()), None);
                }

                #[test]
                fn describes_a_found_value() {
                    assert_eq!(describe("a", &data()), "a=2");
                }

                #[test]
                fn describes_a_missing_value() {
                    assert_eq!(describe("z", &data()), "z=missing");
                }
            """),
        },
        "reference": {
            "src/service.rs": code("""
                //! Service reads.

                use crate::store::load;

                pub fn fetch(key: &str, data: &[(&str, i32)]) -> Option<i32> {
                    load(key, data).filter(|value| *value > 0)
                }
            """),
            "src/report.rs": code("""
                //! Report formatting.

                use crate::store::load;

                pub fn describe(key: &str, data: &[(&str, i32)]) -> String {
                    match load(key, data) {
                        Some(value) => format!("{key}={value}"),
                        None => format!("{key}=missing"),
                    }
                }
            """),
        },
        "mutation": {"file": "src/service.rs", "find": "load(key, data).filter(|value| *value > 0)",
                     "replace": "load(key, data).filter(|value| *value >= 0)"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0432"]},
    },
    {
        "id": "repo-09-result-propagation",
        "category": "repo-scale",
        "difficulty": "medium",
        "capability": "Propagate a return-type change from an API module to its consumers.",
        "prompt_intro": (
            "This crate does not compile. `api::fetch` changed from `Option<i32>` to "
            "`Result<i32, String>`, but `cache` and `report` still match `Some`/`None`. "
            "Update both consumers: a cache miss and an error both fall back to 0, and "
            "a report error renders `missing`. Keep `run` and the signatures."),
        "observed_failure": (
            "error[E0308]: mismatched types\n"
            " --> src/cache.rs:5:11\n"
            "  |\n"
            "4 |     match api::fetch(key) {\n"
            "  |           --------------- this expression has type `Result<i32, String>`\n"
            "5 |         Some(value) => value,\n"
            "  |         ^^^^^^^^^^^ expected `Result<i32, String>`, found `Option<_>`"),
        "files": {
            "src/lib.rs": code("""
                //! A fetch pipeline.

                pub mod api;
                pub mod cache;
                pub mod report;

                pub fn run(key: &str) -> String {
                    format!("{}:{}", cache::cached(key), report::describe(key))
                }
            """),
            "src/api.rs": code("""
                //! The new fallible fetch API.

                pub fn fetch(key: &str) -> Result<i32, String> {
                    if key.is_empty() {
                        return Err("empty key".to_string());
                    }
                    match key {
                        "ok" => Ok(1),
                        "neg" => Ok(-2),
                        _ => Err("missing".to_string()),
                    }
                }
            """),
            "src/cache.rs": code("""
                //! Cache reads, still matching the old Option API.

                use crate::api;

                pub fn cached(key: &str) -> i32 {
                    match api::fetch(key) {
                        Some(value) => value,
                        None => 0,
                    }
                }
            """),
            "src/report.rs": code("""
                //! Report formatting, still matching the old Option API.

                use crate::api;

                pub fn describe(key: &str) -> String {
                    match api::fetch(key) {
                        Some(value) => format!("{key}={value}"),
                        None => format!("{key}=missing"),
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use repo_09_result_propagation::{cache::cached, report::describe, run};

                #[test]
                fn caches_a_found_value() {
                    assert_eq!(cached("ok"), 1);
                }

                #[test]
                fn a_miss_falls_back_to_zero() {
                    assert_eq!(cached("missing"), 0);
                    assert_eq!(cached(""), 0);
                }

                #[test]
                fn describes_a_found_value() {
                    assert_eq!(describe("ok"), "ok=1");
                    assert_eq!(describe("neg"), "neg=-2");
                }

                #[test]
                fn describes_an_error_as_missing() {
                    assert_eq!(describe("missing"), "missing=missing");
                    assert_eq!(run("missing"), "0:missing=missing");
                }
            """),
        },
        "reference": {
            "src/cache.rs": code("""
                //! Cache reads.

                use crate::api;

                pub fn cached(key: &str) -> i32 {
                    match api::fetch(key) {
                        Ok(value) => value,
                        Err(_) => 0,
                    }
                }
            """),
            "src/report.rs": code("""
                //! Report formatting.

                use crate::api;

                pub fn describe(key: &str) -> String {
                    match api::fetch(key) {
                        Ok(value) => format!("{key}={value}"),
                        Err(_) => format!("{key}=missing"),
                    }
                }
            """),
        },
        "mutation": {"file": "src/cache.rs", "find": "Ok(value) => value",
                     "replace": "Ok(value) => value + 1"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0308"]},
    },
]
