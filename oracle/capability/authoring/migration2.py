"""API migrations against a vendored dependency, batch two (five tasks).

`dep/` is harness-owned and restored after the arm writes, so each task is a
change at the call sites. Every initial state fails to build against the
vendored 2.0 API; every reference changes only the application code.
"""

from .common import code


def _app_manifest(name: str, dependency: str) -> str:
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
        {dependency} = {{ path = "dep" }}
    """)


def _dep_manifest(name: str) -> str:
    return code(f"""
        [package]
        name = "{name}"
        version = "2.0.0"
        edition = "2021"

        [lib]
        path = "src/lib.rs"
    """)


TASKS = [
    {
        "id": "migrate-05-iterator-api",
        "category": "api-migration",
        "difficulty": "hard",
        "capability": "Replace a removed collection-returning API with the callback API.",
        "prompt_intro": (
            "The crate does not build against the vendored `items` 2.0.0. Version 1 "
            "returned a `Vec<Item>`; version 2 removed that function and provides "
            "`items::scan(input, &mut |item| ...)`, which visits each item. Rewrite "
            "`names` using `scan`. Do not modify `dep/`. Keep "
            "`pub fn names(input: &str) -> Vec<String>`. Empty comma-separated "
            "segments are skipped."),
        "observed_failure": (
            "error[E0425]: cannot find function `collect` in crate `items`\n"
            " --> src/lib.rs:3:28\n"
            "  |\n"
            "3 |     items::collect(input).into_iter().map(|item| item.name.to_string()).collect()\n"
            "  |                            ^^^^^^^ not found in `items`"),
        "manifest": _app_manifest("migrate_05_iterator_api", "items"),
        "files": {
            "dep/Cargo.toml": _dep_manifest("items"),
            "dep/src/lib.rs": code("""
                //! Item scanning, version 2. The collection-returning API is gone.

                pub struct Item<'a> {
                    pub name: &'a str,
                }

                pub fn scan<'a>(input: &'a str, visit: &mut dyn FnMut(Item<'a>)) {
                    for part in input.split(',') {
                        if !part.is_empty() {
                            visit(Item { name: part });
                        }
                    }
                }
            """),
            "src/lib.rs": code("""
                //! Collect item names through the item crate. This does not build:
                //! version 2 replaced `collect` with the callback `scan`.

                pub fn names(input: &str) -> Vec<String> {
                    items::collect(input).into_iter().map(|item| item.name.to_string()).collect()
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use migrate_05_iterator_api::names;

                #[test]
                fn visits_every_segment() {
                    assert_eq!(names("a,b"), vec!["a".to_string(), "b".to_string()]);
                }

                #[test]
                fn empty_input_is_empty() {
                    assert!(names("").is_empty());
                }

                #[test]
                fn empty_segments_are_skipped() {
                    assert_eq!(names("a,,b"), vec!["a".to_string(), "b".to_string()]);
                }

                #[test]
                fn segments_are_not_trimmed() {
                    assert_eq!(names("a, b"), vec!["a".to_string(), " b".to_string()]);
                }
            """),
        },
        "readonly": ["dep/"],
        "reference": {
            "src/lib.rs": code("""
                //! Collect item names through the item crate.

                pub fn names(input: &str) -> Vec<String> {
                    let mut names = Vec::new();
                    items::scan(input, &mut |item| names.push(item.name.to_string()));
                    names
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "item.name.to_string()",
                     "replace": "item.name.to_uppercase()"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0425"]},
    },
    {
        "id": "migrate-06-error-variant",
        "category": "api-migration",
        "difficulty": "hard",
        "capability": "Migrate a match arm to a structured error variant and handle the new variant.",
        "prompt_intro": (
            "The crate does not build against the vendored `table` 2.0.0. Version 1 "
            "had `Error::NotFound`; version 2 has "
            "`Error::Missing { key: String }` and `Error::Empty`. Rewrite `read` to "
            "return `missing <key>` for a missing key and `empty key` for an empty "
            "key. Do not modify `dep/`. Keep "
            "`pub fn read(table: &[(&'static str, i32)], key: &str) -> Result<i32, String>`."),
        "observed_failure": (
            "error[E0599]: no variant or associated item named `NotFound` found for enum `Error`\n"
            " --> src/lib.rs:6:23\n"
            "  |\n"
            "6 |         Err(table::Error::NotFound) => Err(format!(\"missing {key}\")),\n"
            "  |                       ^^^^^^^^ variant or associated item not found in `Error`"),
        "manifest": _app_manifest("migrate_06_error_variant", "table"),
        "files": {
            "dep/Cargo.toml": _dep_manifest("table"),
            "dep/src/lib.rs": code("""
                //! Table lookup, version 2. The error variants are structured.

                #[derive(Debug, PartialEq)]
                pub enum Error {
                    Missing { key: String },
                    Empty,
                }

                pub fn lookup(
                    table: &[(&'static str, i32)],
                    key: &str,
                ) -> Result<i32, Error> {
                    if key.is_empty() {
                        return Err(Error::Empty);
                    }
                    table
                        .iter()
                        .find(|(name, _value)| *name == key)
                        .map(|(_name, value)| *value)
                        .ok_or_else(|| Error::Missing { key: key.to_string() })
                }
            """),
            "src/lib.rs": code("""
                //! Read a value through the table crate. This does not build: the
                //! `NotFound` variant was replaced by `Missing { key }` and `Empty`.

                pub fn read(table: &[(&'static str, i32)], key: &str) -> Result<i32, String> {
                    match table::lookup(table, key) {
                        Ok(value) => Ok(value),
                        Err(table::Error::Missing { .. }) => Err(format!("missing {key}")),
                        Err(table::Error::NotFound) => Err(format!("missing {key}")),
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use migrate_06_error_variant::read;

                fn table() -> Vec<(&'static str, i32)> {
                    vec![("alpha", 1), ("beta", 2)]
                }

                #[test]
                fn returns_a_present_value() {
                    assert_eq!(read(&table(), "beta"), Ok(2));
                }

                #[test]
                fn reports_a_missing_key_with_its_name() {
                    assert_eq!(read(&table(), "gamma"), Err("missing gamma".to_string()));
                }

                #[test]
                fn reports_the_empty_variant_distinctly() {
                    assert_eq!(read(&table(), ""), Err("empty key".to_string()));
                }
            """),
        },
        "readonly": ["dep/"],
        "reference": {
            "src/lib.rs": code("""
                //! Read a value through the table crate.

                pub fn read(table: &[(&'static str, i32)], key: &str) -> Result<i32, String> {
                    match table::lookup(table, key) {
                        Ok(value) => Ok(value),
                        Err(table::Error::Missing { .. }) => Err(format!("missing {key}")),
                        Err(table::Error::Empty) => Err("empty key".to_string()),
                    }
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": '"empty key"', "replace": '"empty"'},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0599"]},
    },
    {
        "id": "migrate-07-display-method",
        "category": "api-migration",
        "difficulty": "easy",
        "capability": "Use a Display implementation after an inherent accessor is removed.",
        "prompt_intro": (
            "The crate does not build against the vendored `labels` 2.0.0. Version 1 "
            "had `Label::text`; version 2 removes it and implements `Display` "
            "instead. Rewrite `render` to use the `Display` rendering. Do not modify "
            "`dep/`. Keep `pub fn render(value: &str) -> String`."),
        "observed_failure": (
            "error[E0599]: no method named `text` found for struct `Label` in the current scope\n"
            " --> src/lib.rs:4:11\n"
            "  |\n"
            "4 |     label.text().to_string()\n"
            "  |           ^^^^ method not found in `Label`"),
        "manifest": _app_manifest("migrate_07_display_method", "labels"),
        "files": {
            "dep/Cargo.toml": _dep_manifest("labels"),
            "dep/src/lib.rs": code("""
                //! Labels, version 2. Rendering is through `Display`; the inherent
                //! `text` accessor was removed.

                pub struct Label(String);

                impl Label {
                    pub fn new(value: &str) -> Self {
                        Label(value.to_string())
                    }
                }

                impl core::fmt::Display for Label {
                    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                        write!(formatter, "{}", self.0)
                    }
                }
            """),
            "src/lib.rs": code("""
                //! Render a label. This does not build: version 2 removed the
                //! `text` accessor and renders through `Display`.

                pub fn render(value: &str) -> String {
                    let label = labels::Label::new(value);
                    label.text().to_string()
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use migrate_07_display_method::render;

                #[test]
                fn renders_a_value() {
                    assert_eq!(render("alpha"), "alpha");
                }

                #[test]
                fn preserves_spacing() {
                    assert_eq!(render("a b"), "a b");
                }

                #[test]
                fn renders_unicode() {
                    assert_eq!(render("hé"), "hé");
                }

                #[test]
                fn empty_renders_empty() {
                    assert_eq!(render(""), "");
                }
            """),
        },
        "readonly": ["dep/"],
        "reference": {
            "src/lib.rs": code("""
                //! Render a label.

                pub fn render(value: &str) -> String {
                    let label = labels::Label::new(value);
                    label.to_string()
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "label.to_string()",
                     "replace": 'format!("{label}!")'},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0599"]},
    },
    {
        "id": "migrate-08-builder-result",
        "category": "api-migration",
        "difficulty": "hard",
        "capability": "Migrate a direct constructor to a fallible builder chain.",
        "prompt_intro": (
            "The crate does not build against the vendored `settings` 2.0.0. Version "
            "1 had `Config::new(level)`; version 2 uses "
            "`Config::builder().level(level).build()`, and `build` fails for level "
            "0. Rewrite `configured` to use the builder and map a build failure to "
            "the string `invalid level`. Do not modify `dep/`. Keep "
            "`pub fn configured(level: u8) -> Result<u8, String>`."),
        "observed_failure": (
            "error[E0599]: no function or associated item named `new` found for struct `Config`\n"
            " --> src/lib.rs:3:20\n"
            "  |\n"
            "3 |     let config = settings::Config::new(level);\n"
            "  |                    ^^^^^^^^^^^^^^^^^^^ function or associated item not found in `Config`"),
        "manifest": _app_manifest("migrate_08_builder_result", "settings"),
        "files": {
            "dep/Cargo.toml": _dep_manifest("settings"),
            "dep/src/lib.rs": code("""
                //! Settings, version 2. Construction goes through a fallible builder.

                #[derive(Debug)]
                pub struct Config {
                    pub level: u8,
                }

                #[derive(Debug)]
                pub struct BuildError;

                impl core::fmt::Display for BuildError {
                    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                        write!(formatter, "level must be non-zero")
                    }
                }

                pub struct Builder {
                    level: u8,
                }

                impl Builder {
                    pub fn level(mut self, level: u8) -> Self {
                        self.level = level;
                        self
                    }

                    pub fn build(self) -> Result<Config, BuildError> {
                        if self.level == 0 {
                            Err(BuildError)
                        } else {
                            Ok(Config { level: self.level })
                        }
                    }
                }

                impl Config {
                    pub fn builder() -> Builder {
                        Builder { level: 1 }
                    }
                }
            """),
            "src/lib.rs": code("""
                //! Configure a value. This does not build: version 2 removed the
                //! direct constructor in favour of a fallible builder.

                pub fn configured(level: u8) -> Result<u8, String> {
                    let config = settings::Config::new(level);
                    Ok(config.level)
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use migrate_08_builder_result::configured;

                #[test]
                fn accepts_a_positive_level() {
                    assert_eq!(configured(3), Ok(3));
                }

                #[test]
                fn accepts_the_maximum_level() {
                    assert_eq!(configured(255), Ok(255));
                }

                #[test]
                fn rejects_zero_with_the_registered_text() {
                    assert_eq!(configured(0), Err("invalid level".to_string()));
                }

                #[test]
                fn accepts_level_one() {
                    assert_eq!(configured(1), Ok(1));
                }
            """),
        },
        "readonly": ["dep/"],
        "reference": {
            "src/lib.rs": code("""
                //! Configure a value.

                pub fn configured(level: u8) -> Result<u8, String> {
                    settings::Config::builder()
                        .level(level)
                        .build()
                        .map(|config| config.level)
                        .map_err(|_| "invalid level".to_string())
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": '"invalid level"', "replace": '"invalid"'},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0599"]},
    },
    {
        "id": "migrate-09-mutable-receiver",
        "category": "api-migration",
        "difficulty": "easy",
        "capability": "Bind a value mutably when the new API requires a mutable receiver.",
        "prompt_intro": (
            "The crate does not build against the vendored `counter` 2.0.0. Version 1 "
            "used interior mutability and `tick(&self)`; version 2 uses "
            "`tick(&mut self)`. Fix the call site so `count_to(limit)` ticks `limit` "
            "times and returns the value. Do not modify `dep/`. Keep "
            "`pub fn count_to(limit: u32) -> u32`."),
        "observed_failure": (
            "error[E0596]: cannot borrow `counter` as mutable, as it is not declared as mutable\n"
            " --> src/lib.rs:4:9\n"
            "  |\n"
            "3 |     let counter = counter::Counter::new();\n"
            "  |         ------- help: consider changing this to be mutable: `mut counter`\n"
            "4 |         counter.tick();\n"
            "  |         ^^^^^^^^^^^^^^ cannot borrow as mutable"),
        "manifest": _app_manifest("migrate_09_mutable_receiver", "counter"),
        "files": {
            "dep/Cargo.toml": _dep_manifest("counter"),
            "dep/src/lib.rs": code("""
                //! Counter, version 2. `tick` now takes `&mut self`.

                pub struct Counter {
                    value: u32,
                }

                impl Counter {
                    pub fn new() -> Self {
                        Counter { value: 0 }
                    }

                    pub fn tick(&mut self) {
                        self.value += 1;
                    }

                    pub fn value(&self) -> u32 {
                        self.value
                    }
                }
            """),
            "src/lib.rs": code("""
                //! Count through the counter crate. This does not build: version 2
                //! requires a mutable receiver for `tick`.

                pub fn count_to(limit: u32) -> u32 {
                    let counter = counter::Counter::new();
                    for _ in 0..limit {
                        counter.tick();
                    }
                    counter.value()
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use migrate_09_mutable_receiver::count_to;

                #[test]
                fn ticks_the_exact_number_of_times() {
                    assert_eq!(count_to(3), 3);
                }

                #[test]
                fn zero_ticks_is_zero() {
                    assert_eq!(count_to(0), 0);
                }

                #[test]
                fn larger_counts_still_work() {
                    assert_eq!(count_to(10), 10);
                }
            """),
        },
        "readonly": ["dep/"],
        "reference": {
            "src/lib.rs": code("""
                //! Count through the counter crate.

                pub fn count_to(limit: u32) -> u32 {
                    let mut counter = counter::Counter::new();
                    for _ in 0..limit {
                        counter.tick();
                    }
                    counter.value()
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "for _ in 0..limit",
                     "replace": "for _ in 0..=limit"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0596"]},
    },
]
