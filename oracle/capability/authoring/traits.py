"""Trait and impl errors (three tasks).

An incomplete impl, an overlapping impl, and a trait that is not object-safe.
Each failing state is a compiler diagnostic; each reference changes the trait
design rather than working around it at the call site.
"""

TASKS = [
    {
        "id": "traits-01-associated-type",
        "category": "trait-impl",
        "capability": "Complete an impl that omits an associated type and keep the generic helper valid.",
        "prompt_intro": (
            "This crate does not build: the `Repo` impl for `MemoryRepo` omits the\n"
            "associated type `Id`. Add it (`type Id = u32;`) so the impl matches the\n"
            "trait and the generic helper compiles. Do not change the trait."),
        "observed_failure": (
            "error[E0046]: not all trait items implemented, missing: `Id`\n"
            " --> src/lib.rs:11:1\n"
            "  |\n"
            "11| impl Repo for MemoryRepo {\n"
            "  | ^^^^^^^^^^^^^^^^^^^^^^^^ missing `Id` in implementation"),
        "files": {
            "src/lib.rs": (
                "//! A repository trait with an associated identifier type.\n"
                "\n"
                "use std::collections::BTreeMap;\n"
                "\n"
                "pub trait Repo {\n"
                "    type Id;\n"
                "    fn get(&self, id: Self::Id) -> Option<i64>;\n"
                "    fn put(&mut self, id: Self::Id, value: i64);\n"
                "}\n"
                "\n"
                "#[derive(Default)]\n"
                "pub struct MemoryRepo {\n"
                "    items: BTreeMap<u32, i64>,\n"
                "}\n"
                "\n"
                "impl Repo for MemoryRepo {\n"
                "    fn get(&self, id: u32) -> Option<i64> {\n"
                "        self.items.get(&id).copied()\n"
                "    }\n"
                "\n"
                "    fn put(&mut self, id: u32, value: i64) {\n"
                "        self.items.insert(id, value);\n"
                "    }\n"
                "}\n"
                "\n"
                "pub fn first_value<R: Repo<Id = u32>>(repo: &R) -> i64 {\n"
                "    repo.get(0).unwrap_or(0)\n"
                "}\n"),
        },
        "hidden": {
            "tests/check.rs": (
                "use traits_01_associated_type::{first_value, MemoryRepo, Repo};\n"
                "\n"
                "#[test]\n"
                "fn put_then_get() {\n"
                "    let mut repo = MemoryRepo::default();\n"
                "    repo.put(7, 11);\n"
                "    assert_eq!(repo.get(7), Some(11));\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn the_generic_helper_uses_the_associated_type() {\n"
                "    let mut repo = MemoryRepo::default();\n"
                "    repo.put(0, 5);\n"
                "    assert_eq!(first_value(&repo), 5);\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn a_missing_id_is_none() {\n"
                "    let repo = MemoryRepo::default();\n"
                "    assert_eq!(repo.get(3), None);\n"
                "}\n"),
        },
        "reference": {
            "src/lib.rs": (
                "//! A repository trait with an associated identifier type.\n"
                "\n"
                "use std::collections::BTreeMap;\n"
                "\n"
                "pub trait Repo {\n"
                "    type Id;\n"
                "    fn get(&self, id: Self::Id) -> Option<i64>;\n"
                "    fn put(&mut self, id: Self::Id, value: i64);\n"
                "}\n"
                "\n"
                "#[derive(Default)]\n"
                "pub struct MemoryRepo {\n"
                "    items: BTreeMap<u32, i64>,\n"
                "}\n"
                "\n"
                "impl Repo for MemoryRepo {\n"
                "    type Id = u32;\n"
                "\n"
                "    fn get(&self, id: u32) -> Option<i64> {\n"
                "        self.items.get(&id).copied()\n"
                "    }\n"
                "\n"
                "    fn put(&mut self, id: u32, value: i64) {\n"
                "        self.items.insert(id, value);\n"
                "    }\n"
                "}\n"
                "\n"
                "pub fn first_value<R: Repo<Id = u32>>(repo: &R) -> i64 {\n"
                "    repo.get(0).unwrap_or(0)\n"
                "}\n"),
        },
        "mutation": {"file": "src/lib.rs", "find": "repo.get(0).unwrap_or(0)",
                     "replace": "repo.get(1).unwrap_or(0)"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0046"]},
    },
    {
        "id": "traits-02-coherence",
        "category": "trait-impl",
        "capability": "Resolve an overlapping blanket impl by implementing the concrete types only.",
        "prompt_intro": (
            "This crate does not build: `impl<T: Display> Summary for T` overlaps the\n"
            "impl for `Report`. Remove the blanket impl and implement `Summary` for the\n"
            "two concrete types the application needs: `Report` (prefix `report: `) and\n"
            "`&str` (prefix `text: `). Keep `summary_of_text`."),
        "observed_failure": (
            "error[E0119]: conflicting implementations of trait `Summary` for type `Report`\n"
            " --> src/lib.rs:15:1\n"
            "  |\n"
            "15| impl<T: Display> Summary for T {\n"
            "  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ conflicting implementation for `Report`"),
        "files": {
            "src/lib.rs": (
                "//! Summaries for a report and for text.\n"
                "\n"
                "use core::fmt::Display;\n"
                "\n"
                "pub trait Summary {\n"
                "    fn summary(&self) -> String;\n"
                "}\n"
                "\n"
                "#[derive(Debug)]\n"
                "pub struct Report(pub String);\n"
                "\n"
                "impl Display for Report {\n"
                "    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {\n"
                "        write!(formatter, \"{}\", self.0)\n"
                "    }\n"
                "}\n"
                "\n"
                "impl Summary for Report {\n"
                "    fn summary(&self) -> String {\n"
                "        format!(\"report: {}\", self.0)\n"
                "    }\n"
                "}\n"
                "\n"
                "impl<T: Display> Summary for T {\n"
                "    fn summary(&self) -> String {\n"
                "        format!(\"value: {self}\")\n"
                "    }\n"
                "}\n"
                "\n"
                "pub fn summary_of_text(text: &str) -> String {\n"
                "    text.summary()\n"
                "}\n"),
        },
        "hidden": {
            "tests/check.rs": (
                "use traits_02_coherence::{summary_of_text, Report, Summary};\n"
                "\n"
                "#[test]\n"
                "fn report_summary() {\n"
                "    assert_eq!(Report(\"q3\".to_string()).summary(), \"report: q3\");\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn text_summary() {\n"
                "    assert_eq!(summary_of_text(\"hello\"), \"text: hello\");\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn empty_text_summary() {\n"
                "    assert_eq!(summary_of_text(\"\"), \"text: \");\n"
                "}\n"),
        },
        "reference": {
            "src/lib.rs": (
                "//! Summaries for a report and for text.\n"
                "\n"
                "use core::fmt::Display;\n"
                "\n"
                "pub trait Summary {\n"
                "    fn summary(&self) -> String;\n"
                "}\n"
                "\n"
                "#[derive(Debug)]\n"
                "pub struct Report(pub String);\n"
                "\n"
                "impl Display for Report {\n"
                "    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {\n"
                "        write!(formatter, \"{}\", self.0)\n"
                "    }\n"
                "}\n"
                "\n"
                "impl Summary for Report {\n"
                "    fn summary(&self) -> String {\n"
                "        format!(\"report: {}\", self.0)\n"
                "    }\n"
                "}\n"
                "\n"
                "impl Summary for &str {\n"
                "    fn summary(&self) -> String {\n"
                "        format!(\"text: {self}\")\n"
                "    }\n"
                "}\n"
                "\n"
                "pub fn summary_of_text(text: &str) -> String {\n"
                "    text.summary()\n"
                "}\n"),
        },
        "mutation": {"file": "src/lib.rs", "find": "\"text: {self}\"", "replace": "\"text:{self}\""},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0119"]},
    },
    {
        "id": "traits-03-object-safety",
        "category": "trait-impl",
        "capability": "Make a trait object-safe by removing a generic method, keeping dynamic dispatch.",
        "prompt_intro": (
            "This crate does not build: `Store::put` is generic, so `Store` cannot be\n"
            "used as `dyn Store`. Make `put` take an `i64` directly so the trait is\n"
            "object-safe, and keep `count(&dyn Store)` working. Keep the method names and\n"
            "the `len` semantics."),
        "observed_failure": (
            "error[E0038]: the trait `Store` cannot be made into an object\n"
            " --> src/lib.rs:18:19\n"
            "  |\n"
            "18| pub fn count(store: &dyn Store) -> usize {\n"
            "  |                   ^^^^^^^^^ `Store` cannot be made into an object\n"
            "  |\n"
            "  = note: method `put` has generic type parameters"),
        "files": {
            "src/lib.rs": (
                "//! A small key-value store used through dynamic dispatch.\n"
                "\n"
                "use std::collections::BTreeMap;\n"
                "\n"
                "pub trait Store {\n"
                "    fn get(&self, key: &str) -> Option<i64>;\n"
                "    fn put<T: Into<i64>>(&mut self, key: &str, value: T);\n"
                "    fn len(&self) -> usize;\n"
                "}\n"
                "\n"
                "#[derive(Default)]\n"
                "pub struct MemoryStore {\n"
                "    items: BTreeMap<String, i64>,\n"
                "}\n"
                "\n"
                "impl Store for MemoryStore {\n"
                "    fn get(&self, key: &str) -> Option<i64> {\n"
                "        self.items.get(key).copied()\n"
                "    }\n"
                "\n"
                "    fn put<T: Into<i64>>(&mut self, key: &str, value: T) {\n"
                "        self.items.insert(key.to_string(), value.into());\n"
                "    }\n"
                "\n"
                "    fn len(&self) -> usize {\n"
                "        self.items.len()\n"
                "    }\n"
                "}\n"
                "\n"
                "pub fn count(store: &dyn Store) -> usize {\n"
                "    store.len()\n"
                "}\n"),
        },
        "hidden": {
            "tests/check.rs": (
                "use traits_03_object_safety::{count, MemoryStore, Store};\n"
                "\n"
                "#[test]\n"
                "fn put_and_get() {\n"
                "    let mut store = MemoryStore::default();\n"
                "    store.put(\"a\", 3i64);\n"
                "    assert_eq!(store.get(\"a\"), Some(3));\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn dynamic_dispatch_works() {\n"
                "    let mut store = MemoryStore::default();\n"
                "    store.put(\"a\", 1i64);\n"
                "    store.put(\"b\", 2i64);\n"
                "    let dynamic: &dyn Store = &store;\n"
                "    assert_eq!(count(dynamic), 2);\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn empty_store_counts_zero() {\n"
                "    let store = MemoryStore::default();\n"
                "    assert_eq!(count(&store), 0);\n"
                "}\n"),
        },
        "reference": {
            "src/lib.rs": (
                "//! A small key-value store used through dynamic dispatch.\n"
                "\n"
                "use std::collections::BTreeMap;\n"
                "\n"
                "pub trait Store {\n"
                "    fn get(&self, key: &str) -> Option<i64>;\n"
                "    fn put(&mut self, key: &str, value: i64);\n"
                "    fn len(&self) -> usize;\n"
                "}\n"
                "\n"
                "#[derive(Default)]\n"
                "pub struct MemoryStore {\n"
                "    items: BTreeMap<String, i64>,\n"
                "}\n"
                "\n"
                "impl Store for MemoryStore {\n"
                "    fn get(&self, key: &str) -> Option<i64> {\n"
                "        self.items.get(key).copied()\n"
                "    }\n"
                "\n"
                "    fn put(&mut self, key: &str, value: i64) {\n"
                "        self.items.insert(key.to_string(), value);\n"
                "    }\n"
                "\n"
                "    fn len(&self) -> usize {\n"
                "        self.items.len()\n"
                "    }\n"
                "}\n"
                "\n"
                "pub fn count(store: &dyn Store) -> usize {\n"
                "    store.len()\n"
                "}\n"),
        },
        "mutation": {"file": "src/lib.rs", "find": "store.len()", "replace": "store.len() + 1"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0038"]},
    },
]
