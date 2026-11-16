"""Borrow-checker and lifetime repairs (four tasks).

Each task fails to build in its initial state, names the diagnostic the
self-test must observe, and has a reference solution that changes ownership or
lifetime structure rather than silencing the compiler.
"""

TASKS = [
    {
        "id": "borrow-01-local-reference",
        "category": "lifetime-borrow",
        "capability": "Recognise a reference to a local value and return owned data instead.",
        "prompt_intro": (
            "This crate does not compile. `label` creates a `String` locally and "
            "returns a reference to it. Change the signature to "
            "`pub fn label(prefix: &str) -> String` and keep the formatting "
            "identical."),
        "observed_failure": (
            "error[E0515]: cannot return value referencing local variable `text`\n"
            " --> src/lib.rs:4:5\n"
            "  |\n"
            "3 |     let text = format!(\"{prefix}:ready\");\n"
            "  |         ---- `text` is borrowed here\n"
            "4 |     &text\n"
            "  |     ^^^^^ returns a value referencing data owned by the current function"),
        "files": {
            "src/lib.rs": (
                "//! Label construction. The crate does not build: `label` returns a\n"
                "//! reference to a `String` it creates locally.\n"
                "\n"
                "pub fn label(prefix: &str) -> &str {\n"
                "    let text = format!(\"{prefix}:ready\");\n"
                "    &text\n"
                "}\n"),
        },
        "hidden": {
            "tests/check.rs": (
                "use borrow_01_local_reference::label;\n"
                "\n"
                "#[test]\n"
                "fn label_owns_its_result() {\n"
                "    assert_eq!(label(\"job\"), \"job:ready\".to_string());\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn successive_labels_are_independent() {\n"
                "    let first = label(\"a\");\n"
                "    let second = label(\"b\");\n"
                "    assert_eq!(first, \"a:ready\");\n"
                "    assert_eq!(second, \"b:ready\");\n"
                "}\n"),
        },
        "reference": {
            "src/lib.rs": (
                "//! Label construction.\n"
                "\n"
                "pub fn label(prefix: &str) -> String {\n"
                "    format!(\"{prefix}:ready\")\n"
                "}\n"),
        },
        "mutation": {"file": "src/lib.rs", "find": ":ready", "replace": ":READY"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0515"]},
    },
    {
        "id": "borrow-02-mutate-while-borrowed",
        "category": "lifetime-borrow",
        "capability": "Resolve a shared/mutable borrow conflict by copying the value out.",
        "prompt_intro": (
            "This crate does not compile. `push_and_report` keeps a borrow of the "
            "vector's last element across a push. Return the previous last value "
            "(or 0 for an empty vector) without holding the borrow across the "
            "mutation. Keep the signature."),
        "observed_failure": (
            "error[E0502]: cannot borrow `*values` as mutable because it is also\n"
            "borrowed as immutable\n"
            " --> src/lib.rs:4:5\n"
            "  |\n"
            "2 |     let last = values.last().unwrap_or(&0);\n"
            "  |                ------ immutable borrow occurs here\n"
            "3 |     values.push(value);\n"
            "  |     ^^^^^^^^^^^^^^^^^^ mutable borrow occurs here\n"
            "4 |     *last\n"
            "  |     ----- immutable borrow later used here"),
        "files": {
            "src/lib.rs": (
                "//! Push a value and report the value that was last before it.\n"
                "\n"
                "pub fn push_and_report(values: &mut Vec<i32>, value: i32) -> i32 {\n"
                "    let last = values.last().unwrap_or(&0);\n"
                "    values.push(value);\n"
                "    *last\n"
                "}\n"),
        },
        "hidden": {
            "tests/check.rs": (
                "use borrow_02_mutate_while_borrowed::push_and_report;\n"
                "\n"
                "#[test]\n"
                "fn reports_zero_for_an_empty_vector() {\n"
                "    let mut values = vec![];\n"
                "    assert_eq!(push_and_report(&mut values, 7), 0);\n"
                "    assert_eq!(values, vec![7]);\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn reports_the_previous_last() {\n"
                "    let mut values = vec![1, 2];\n"
                "    assert_eq!(push_and_report(&mut values, 9), 2);\n"
                "    assert_eq!(values, vec![1, 2, 9]);\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn repeated_pushes_chain() {\n"
                "    let mut values = vec![5];\n"
                "    assert_eq!(push_and_report(&mut values, 6), 5);\n"
                "    assert_eq!(push_and_report(&mut values, 8), 6);\n"
                "    assert_eq!(values, vec![5, 6, 8]);\n"
                "}\n"),
        },
        "reference": {
            "src/lib.rs": (
                "//! Push a value and report the value that was last before it.\n"
                "\n"
                "pub fn push_and_report(values: &mut Vec<i32>, value: i32) -> i32 {\n"
                "    let last = values.last().copied().unwrap_or(0);\n"
                "    values.push(value);\n"
                "    last\n"
                "}\n"),
        },
        "mutation": {"file": "src/lib.rs", "find": "unwrap_or(0)", "replace": "unwrap_or(1)"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0502"]},
    },
    {
        "id": "borrow-03-self-referential-window",
        "category": "lifetime-borrow",
        "capability": "Replace a self-referential borrow with on-demand slicing (indices, not references).",
        "prompt_intro": (
            "This crate does not compile. `Window` stores a `&'static str` that "
            "points into a `String` the struct itself owns. Remove the stored "
            "reference and compute the first word from `self.text` when asked, so "
            "`first()` stays valid after `replace()`. Keep the public API: "
            "`Window::new(String)`, `first(&self) -> &str`, `replace(&mut self, String)`."),
        "observed_failure": (
            "error[E0597]: `text` does not live long enough\n"
            "  --> src/lib.rs:10:21\n"
            "   |\n"
            "10 |         let first = text.split_whitespace().next().unwrap_or(\"\");\n"
            "   |                     ^^^^ borrowed value does not live long enough\n"
            "11 |         Self { text, first }\n"
            "   |                      ----- this usage requires that `text` is borrowed for `'static`\n"
            "12 |     }\n"
            "   | - `text` dropped here while still borrowed\n"
            "\n"
            "error[E0505]: cannot move out of `text` because it is borrowed"),
        "files": {
            "src/lib.rs": (
                "//! A window over a text, caching its first word as a slice.\n"
                "\n"
                "pub struct Window {\n"
                "    text: String,\n"
                "    first: &'static str,\n"
                "}\n"
                "\n"
                "impl Window {\n"
                "    pub fn new(text: String) -> Self {\n"
                "        let first = text.split_whitespace().next().unwrap_or(\"\");\n"
                "        Self { text, first }\n"
                "    }\n"
                "\n"
                "    pub fn first(&self) -> &str {\n"
                "        self.first\n"
                "    }\n"
                "\n"
                "    pub fn replace(&mut self, text: String) {\n"
                "        self.text = text;\n"
                "    }\n"
                "}\n"),
        },
        "hidden": {
            "tests/check.rs": (
                "use borrow_03_self_referential_window::Window;\n"
                "\n"
                "#[test]\n"
                "fn first_word() {\n"
                "    assert_eq!(Window::new(\"alpha beta\".to_string()).first(), \"alpha\");\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn empty_text_has_no_first_word() {\n"
                "    assert_eq!(Window::new(String::new()).first(), \"\");\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn first_word_survives_replace() {\n"
                "    let mut window = Window::new(\"alpha beta\".to_string());\n"
                "    window.replace(\"gamma delta\".to_string());\n"
                "    assert_eq!(window.first(), \"gamma\");\n"
                "}\n"),
        },
        "reference": {
            "src/lib.rs": (
                "//! A window over a text, computing its first word on demand.\n"
                "\n"
                "pub struct Window {\n"
                "    text: String,\n"
                "}\n"
                "\n"
                "impl Window {\n"
                "    pub fn new(text: String) -> Self {\n"
                "        Self { text }\n"
                "    }\n"
                "\n"
                "    pub fn first(&self) -> &str {\n"
                "        self.text.split_whitespace().next().unwrap_or(\"\")\n"
                "    }\n"
                "\n"
                "    pub fn replace(&mut self, text: String) {\n"
                "        self.text = text;\n"
                "    }\n"
                "}\n"),
        },
        "mutation": {"file": "src/lib.rs", "find": "unwrap_or(\"\")", "replace": "unwrap_or(\"x\")"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0597"]},
    },
    {
        "id": "borrow-04-iterator-invalidation",
        "category": "lifetime-borrow",
        "capability": "Collect the keys before mutating the map (iterator invalidation).",
        "prompt_intro": (
            "This crate does not compile. The loop removes entries while iterating "
            "the map. Collect the keys that must go first, then remove them. Return "
            "the number removed. Keep the signature."),
        "observed_failure": (
            "error[E0502]: cannot borrow `*map` as mutable because it is also\n"
            "borrowed as immutable\n"
            " --> src/lib.rs:6:13\n"
            "  |\n"
            "4 |     for (key, _value) in map.iter() {\n"
            "  |                           ------- immutable borrow occurs here\n"
            "6 |             map.remove(key);\n"
            "  |             ^^^^^^^^^^^^^^^ mutable borrow occurs here"),
        "files": {
            "src/lib.rs": (
                "//! Remove every key longer than a bound.\n"
                "\n"
                "use std::collections::BTreeMap;\n"
                "\n"
                "pub fn drop_long_keys(map: &mut BTreeMap<String, i32>, max_len: usize) -> usize {\n"
                "    let mut removed = 0;\n"
                "    for (key, _value) in map.iter() {\n"
                "        if key.len() > max_len {\n"
                "            map.remove(key);\n"
                "            removed += 1;\n"
                "        }\n"
                "    }\n"
                "    removed\n"
                "}\n"),
        },
        "hidden": {
            "tests/check.rs": (
                "use borrow_04_iterator_invalidation::drop_long_keys;\n"
                "use std::collections::BTreeMap;\n"
                "\n"
                "fn map() -> BTreeMap<String, i32> {\n"
                "    let mut map = BTreeMap::new();\n"
                "    map.insert(\"a\".to_string(), 1);\n"
                "    map.insert(\"bb\".to_string(), 2);\n"
                "    map.insert(\"ccc\".to_string(), 3);\n"
                "    map\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn removes_only_the_long_keys() {\n"
                "    let mut map = map();\n"
                "    assert_eq!(drop_long_keys(&mut map, 1), 2);\n"
                "    assert_eq!(map.keys().cloned().collect::<Vec<_>>(), vec![\"a\".to_string()]);\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn keeps_everything_when_the_bound_is_large() {\n"
                "    let mut map = map();\n"
                "    assert_eq!(drop_long_keys(&mut map, 10), 0);\n"
                "    assert_eq!(map.len(), 3);\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn empty_map_is_unchanged() {\n"
                "    let mut map = BTreeMap::new();\n"
                "    assert_eq!(drop_long_keys(&mut map, 0), 0);\n"
                "}\n"),
        },
        "reference": {
            "src/lib.rs": (
                "//! Remove every key longer than a bound.\n"
                "\n"
                "use std::collections::BTreeMap;\n"
                "\n"
                "pub fn drop_long_keys(map: &mut BTreeMap<String, i32>, max_len: usize) -> usize {\n"
                "    let doomed: Vec<String> = map\n"
                "        .keys()\n"
                "        .filter(|key| key.len() > max_len)\n"
                "        .cloned()\n"
                "        .collect();\n"
                "    for key in &doomed {\n"
                "        map.remove(key);\n"
                "    }\n"
                "    doomed.len()\n"
                "}\n"),
        },
        "mutation": {"file": "src/lib.rs", "find": "key.len() > max_len",
                     "replace": "key.len() >= max_len"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0502"]},
    },
]
