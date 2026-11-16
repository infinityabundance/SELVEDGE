"""Repo-scale multi-file repairs, batch two (six tasks).

Every crate has three or four source files and the repair must propagate across
them: a visibility, an enum variant, an error conversion, a renamed field, a
mutability change, and a trait method.
"""

from .common import code

TASKS = [
    {
        "id": "repo-04-module-visibility",
        "category": "repo-scale",
        "difficulty": "easy",
        "capability": "Widen a module-private helper to crate visibility for a sibling module.",
        "prompt_intro": (
            "This crate does not compile. `service::run` calls `model::load`, which is "
            "private to the model module. Make `load` visible within the crate; do not "
            "make it part of the public API and do not change its behaviour. Invalid "
            "lines count as 0."),
        "observed_failure": (
            "error[E0603]: function `load` is private\n"
            " --> src/service.rs:1:19\n"
            "  |\n"
            "1 | use crate::model::load;\n"
            "  |                   ^^^^ private function"),
        "files": {
            "src/lib.rs": code("""
                //! Line totals.

                pub mod model;
                pub mod service;

                pub use service::run;
            """),
            "src/model.rs": code("""
                //! Record parsing.

                pub struct Record {
                    pub value: i32,
                }

                fn load(lines: &[&str]) -> Vec<Record> {
                    lines
                        .iter()
                        .map(|line| Record { value: line.trim().parse().unwrap_or(0) })
                        .collect()
                }
            """),
            "src/service.rs": code("""
                //! Totals over parsed lines.

                use crate::model::load;

                pub fn run(lines: &[&str]) -> i32 {
                    load(lines).iter().map(|record| record.value).sum()
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use repo_04_module_visibility::run;

                #[test]
                fn totals_numbers() {
                    assert_eq!(run(&["1", "2", "3"]), 6);
                }

                #[test]
                fn trims_whitespace() {
                    assert_eq!(run(&[" 4 ", "5"]), 9);
                }

                #[test]
                fn invalid_lines_count_as_zero() {
                    assert_eq!(run(&["x", "2"]), 2);
                }

                #[test]
                fn empty_input_is_zero() {
                    assert_eq!(run(&[]), 0);
                }
            """),
        },
        "reference": {
            "src/model.rs": code("""
                //! Record parsing.

                pub struct Record {
                    pub value: i32,
                }

                pub(crate) fn load(lines: &[&str]) -> Vec<Record> {
                    lines
                        .iter()
                        .map(|line| Record { value: line.trim().parse().unwrap_or(0) })
                        .collect()
                }
            """),
        },
        "mutation": {"file": "src/model.rs", "find": "unwrap_or(0)", "replace": "unwrap_or(1)"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0603"]},
    },
    {
        "id": "repo-05-variant-propagation",
        "category": "repo-scale",
        "difficulty": "hard",
        "capability": "Propagate a new enum variant through every exhaustive match.",
        "prompt_intro": (
            "This crate does not compile. A `Pause` variant was added to `Event`, but "
            "`handler` and `audit` do not handle it. Add the arm to both: `handle` "
            "returns `\"pause\"` and `audit::describe` returns `\"P\"`. Keep every "
            "existing arm unchanged."),
        "observed_failure": (
            "error[E0004]: non-exhaustive patterns: `&Event::Pause` not covered\n"
            " --> src/handler.rs:4:11\n"
            "  |\n"
            "4 |     match event {\n"
            "  |           ^^^^^ pattern `&Event::Pause` not covered"),
        "files": {
            "src/lib.rs": code("""
                //! Event handling.

                pub mod audit;
                pub mod event;
                pub mod handler;

                pub use handler::handle;
            """),
            "src/event.rs": code("""
                //! Event kinds.

                #[derive(Clone, Copy, Debug, PartialEq)]
                pub enum Event {
                    Start,
                    Stop,
                    Pause,
                }
            """),
            "src/handler.rs": code("""
                //! Event handling.

                use crate::event::Event;

                pub fn handle(event: &Event) -> &'static str {
                    match event {
                        Event::Start => "start",
                        Event::Stop => "stop",
                    }
                }
            """),
            "src/audit.rs": code("""
                //! Audit descriptions.

                use crate::event::Event;

                pub fn describe(event: &Event) -> String {
                    match event {
                        Event::Start => "S".to_string(),
                        Event::Stop => "T".to_string(),
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use repo_05_variant_propagation::{audit::describe, event::Event, handle};

                #[test]
                fn handles_every_event() {
                    assert_eq!(handle(&Event::Start), "start");
                    assert_eq!(handle(&Event::Stop), "stop");
                    assert_eq!(handle(&Event::Pause), "pause");
                }

                #[test]
                fn describes_every_event() {
                    assert_eq!(describe(&Event::Start), "S");
                    assert_eq!(describe(&Event::Stop), "T");
                    assert_eq!(describe(&Event::Pause), "P");
                }

                #[test]
                fn descriptions_are_one_character() {
                    for event in [Event::Start, Event::Stop, Event::Pause] {
                        assert_eq!(describe(&event).chars().count(), 1);
                    }
                }
            """),
        },
        "reference": {
            "src/handler.rs": code("""
                //! Event handling.

                use crate::event::Event;

                pub fn handle(event: &Event) -> &'static str {
                    match event {
                        Event::Start => "start",
                        Event::Stop => "stop",
                        Event::Pause => "pause",
                    }
                }
            """),
            "src/audit.rs": code("""
                //! Audit descriptions.

                use crate::event::Event;

                pub fn describe(event: &Event) -> String {
                    match event {
                        Event::Start => "S".to_string(),
                        Event::Stop => "T".to_string(),
                        Event::Pause => "P".to_string(),
                    }
                }
            """),
        },
        "mutation": {"file": "src/handler.rs", "find": 'Event::Pause => "pause"',
                     "replace": 'Event::Pause => "hold"'},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0004"]},
    },
    {
        "id": "repo-06-error-conversion",
        "category": "repo-scale",
        "difficulty": "hard",
        "capability": "Add the From conversion that lets ? cross a module error boundary.",
        "prompt_intro": (
            "This crate does not compile. `service::load` uses `?` on a "
            "`StoreError`, but its return type is `AppError`; no conversion exists. "
            "Implement `From<StoreError> for AppError`: `Corrupt(key)` becomes "
            "`AppError::Corrupt(key)` and `NotFound(key)` becomes "
            "`AppError::Missing(key)`. Keep every signature."),
        "observed_failure": (
            "error[E0277]: the `?` operator can only be used in a function that returns `Result` or `Option`\n"
            " --> src/service.rs:5:35\n"
            "  |\n"
            "5 |     let value = store::read(key, data)?;\n"
            "  |                                   ^ cannot use the `?` operator in a function that returns `Result<String, AppError>`\n"
            "  |\n"
            "  = help: the trait `From<StoreError>` is not implemented for `AppError`"),
        "files": {
            "src/lib.rs": code("""
                //! Loading values through a store.

                pub mod error;
                pub mod service;
                pub mod store;

                pub use service::load;
            """),
            "src/error.rs": code("""
                //! Application errors.

                #[derive(Debug, PartialEq)]
                pub enum AppError {
                    Parse(String),
                    Missing(String),
                    Corrupt(String),
                }
            """),
            "src/store.rs": code("""
                //! Store reads.

                #[derive(Debug, PartialEq)]
                pub enum StoreError {
                    Corrupt(String),
                    NotFound(String),
                }

                pub fn read(key: &str, data: &[(&str, &str)]) -> Result<String, StoreError> {
                    if key.starts_with("bad") {
                        return Err(StoreError::Corrupt(key.to_string()));
                    }
                    data.iter()
                        .find(|(name, _value)| *name == key)
                        .map(|(_name, value)| value.to_string())
                        .ok_or_else(|| StoreError::NotFound(key.to_string()))
                }
            """),
            "src/service.rs": code("""
                //! Service loading.

                use crate::error::AppError;
                use crate::store;

                pub fn load(key: &str, data: &[(&str, &str)]) -> Result<String, AppError> {
                    let value = store::read(key, data)?;
                    if value.is_empty() {
                        return Err(AppError::Parse("empty".to_string()));
                    }
                    Ok(value)
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use repo_06_error_conversion::{error::AppError, load};

                fn data() -> Vec<(&'static str, &'static str)> {
                    vec![("alpha", "1"), ("blank", "")]
                }

                #[test]
                fn returns_a_present_value() {
                    assert_eq!(load("alpha", &data()), Ok("1".to_string()));
                }

                #[test]
                fn maps_a_missing_key_to_missing() {
                    assert_eq!(
                        load("gamma", &data()),
                        Err(AppError::Missing("gamma".to_string()))
                    );
                }

                #[test]
                fn maps_a_corrupt_key_to_corrupt() {
                    assert_eq!(
                        load("bad-key", &data()),
                        Err(AppError::Corrupt("bad-key".to_string()))
                    );
                }

                #[test]
                fn reports_an_empty_value_as_a_parse_error() {
                    assert_eq!(
                        load("blank", &data()),
                        Err(AppError::Parse("empty".to_string()))
                    );
                }
            """),
        },
        "reference": {
            "src/error.rs": code("""
                //! Application errors.

                use crate::store::StoreError;

                #[derive(Debug, PartialEq)]
                pub enum AppError {
                    Parse(String),
                    Missing(String),
                    Corrupt(String),
                }

                impl From<StoreError> for AppError {
                    fn from(error: StoreError) -> Self {
                        match error {
                            StoreError::Corrupt(key) => AppError::Corrupt(key),
                            StoreError::NotFound(key) => AppError::Missing(key),
                        }
                    }
                }
            """),
        },
        "mutation": {"file": "src/error.rs", "find": "AppError::Missing(key)",
                     "replace": "AppError::Parse(key)"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0277"]},
    },
    {
        "id": "repo-07-field-rename",
        "category": "repo-scale",
        "difficulty": "easy",
        "capability": "Rename a shared struct field without changing its callers' contract.",
        "prompt_intro": (
            "This crate does not compile. The shared `Config` field is still called "
            "`source`, while `reader` and `writer` already use the new public name "
            "`path`. Rename the field to `path`; keep `retries`, the constructor and "
            "all module behaviour unchanged."),
        "observed_failure": (
            "error[E0609]: no field `path` on type `&Config`\n"
            " --> src/reader.rs:4:24\n"
            "  |\n"
            "4 |     format!(\"{}:{}\", config.path, config.retries)\n"
            "  |                        ^^^^ unknown field"),
        "files": {
            "src/lib.rs": code("""
                //! Configuration shared by a reader and a writer.

                pub mod config;
                pub mod reader;
                pub mod writer;

                pub use config::Config;
                pub use reader::read;
                pub use writer::write;
            """),
            "src/config.rs": code("""
                //! Shared configuration.

                pub struct Config {
                    pub source: String,
                    pub retries: u32,
                }

                impl Config {
                    pub fn new(source: &str, retries: u32) -> Self {
                        Config { source: source.to_string(), retries }
                    }
                }
            """),
            "src/reader.rs": code("""
                //! Configuration reader.

                use crate::config::Config;

                pub fn read(config: &Config) -> String {
                    format!("{}:{}", config.path, config.retries)
                }
            """),
            "src/writer.rs": code("""
                //! Configuration writer.

                use crate::config::Config;

                pub fn write(config: &Config) -> String {
                    format!("write {}", config.path)
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use repo_07_field_rename::{read, write, Config};

                #[test]
                fn exposes_the_new_field() {
                    assert_eq!(Config::new("alpha", 2).path, "alpha");
                }

                #[test]
                fn keeps_retries() {
                    assert_eq!(Config::new("alpha", 2).retries, 2);
                }

                #[test]
                fn reader_uses_the_field() {
                    assert_eq!(read(&Config::new("alpha", 2)), "alpha:2");
                }

                #[test]
                fn writer_uses_the_field() {
                    assert_eq!(write(&Config::new("beta", 0)), "write beta");
                }
            """),
        },
        "reference": {
            "src/config.rs": code("""
                //! Shared configuration.

                pub struct Config {
                    pub path: String,
                    pub retries: u32,
                }

                impl Config {
                    pub fn new(path: &str, retries: u32) -> Self {
                        Config { path: path.to_string(), retries }
                    }
                }
            """),
        },
        "mutation": {"file": "src/config.rs", "find": "path: path.to_string()",
                     "replace": "path: path.to_uppercase()"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0609"]},
    },
    {
        "id": "repo-08-mutability-propagation",
        "category": "repo-scale",
        "difficulty": "hard",
        "capability": "Propagate a mutable receiver through a three-module pipeline.",
        "prompt_intro": (
            "This crate does not compile. `update::bump` assigns through a shared "
            "reference, and `run` binds the state immutably. Change the update "
            "signature to `&mut State` and make the binding in `run` mutable. Keep "
            "`run` returning the count after two increments of one."),
        "observed_failure": (
            "error[E0594]: cannot assign to `state.count`, as `state` is not declared as mutable\n"
            " --> src/update.rs:3:5\n"
            "  |\n"
            "3 |     state.count += by;\n"
            "  |     ^^^^^^^^^^^^^^^^^ cannot assign"),
        "files": {
            "src/lib.rs": code("""
                //! A state pipeline.

                pub mod report;
                pub mod state;
                pub mod update;

                pub fn run() -> i32 {
                    let state = state::initial();
                    update::bump(&state, 1);
                    update::bump(&state, 1);
                    report::count(&state)
                }
            """),
            "src/state.rs": code("""
                //! State construction.

                pub struct State {
                    pub count: i32,
                }

                pub fn initial() -> State {
                    State { count: 0 }
                }
            """),
            "src/update.rs": code("""
                //! State updates.

                use crate::state::State;

                pub fn bump(state: &State, by: i32) {
                    state.count += by;
                }
            """),
            "src/report.rs": code("""
                //! State reporting.

                use crate::state::State;

                pub fn count(state: &State) -> i32 {
                    state.count
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use repo_08_mutability_propagation::{report::count, state, update::bump, run};

                #[test]
                fn run_bumps_twice() {
                    assert_eq!(run(), 2);
                }

                #[test]
                fn initial_state_is_zero() {
                    assert_eq!(count(&state::initial()), 0);
                }

                #[test]
                fn bump_accumulates() {
                    let mut state = state::initial();
                    bump(&mut state, 5);
                    bump(&mut state, -2);
                    assert_eq!(count(&state), 3);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! A state pipeline.

                pub mod report;
                pub mod state;
                pub mod update;

                pub fn run() -> i32 {
                    let mut state = state::initial();
                    update::bump(&mut state, 1);
                    update::bump(&mut state, 1);
                    report::count(&state)
                }
            """),
            "src/update.rs": code("""
                //! State updates.

                use crate::state::State;

                pub fn bump(state: &mut State, by: i32) {
                    state.count += by;
                }
            """),
        },
        "mutation": {"file": "src/update.rs", "find": "state.count += by",
                     "replace": "state.count -= by"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0594"]},
    },
    {
        "id": "repo-09-trait-method-propagation",
        "category": "repo-scale",
        "difficulty": "hard",
        "capability": "Add a required trait method and implement it in every module.",
        "prompt_intro": (
            "This crate does not compile. The `Sink` trait gained a required `flush` "
            "method; the JSON and CSV implementations do not provide it. Implement "
            "`flush` returning `\"ok\"` in both, and keep the pipeline's output "
            "`{json};{csv};{flush}`. Keep every signature."),
        "observed_failure": (
            "error[E0046]: not all trait items implemented, missing: `flush`\n"
            " --> src/json.rs:4:1\n"
            "  |\n"
            "4 | impl Sink for Json {\n"
            "  | ^^^^^^^^^^^^^^^^^^ missing `flush` in implementation"),
        "files": {
            "src/lib.rs": code("""
                //! A two-sink pipeline.

                pub mod csv;
                pub mod json;
                pub mod pipeline;

                pub use pipeline::run;
            """),
            "src/pipeline.rs": code("""
                //! Sink pipeline.

                pub trait Sink {
                    fn write(&self, value: i32) -> String;
                    fn flush(&self) -> String;
                }

                pub fn run(value: i32) -> String {
                    let json = crate::json::Json;
                    let csv = crate::csv::Csv;
                    format!(
                        "{};{};{}",
                        Sink::write(&json, value),
                        Sink::write(&csv, value),
                        Sink::flush(&json)
                    )
                }
            """),
            "src/json.rs": code("""
                //! JSON sink.

                use crate::pipeline::Sink;

                pub struct Json;

                impl Sink for Json {
                    fn write(&self, value: i32) -> String {
                        format!("{{\\\"v\\\":{value}}}")
                    }
                }
            """),
            "src/csv.rs": code("""
                //! CSV sink.

                use crate::pipeline::Sink;

                pub struct Csv;

                impl Sink for Csv {
                    fn write(&self, value: i32) -> String {
                        format!("{value}")
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use repo_09_trait_method_propagation::{csv::Csv, json::Json, pipeline::Sink, run};

                #[test]
                fn pipeline_joins_every_sink() {
                    assert_eq!(run(3), "{\\"v\\":3};3;ok");
                }

                #[test]
                fn pipeline_handles_zero() {
                    assert_eq!(run(0), "{\\"v\\":0};0;ok");
                }

                #[test]
                fn pipeline_handles_negatives() {
                    assert_eq!(run(-1), "{\\"v\\":-1};-1;ok");
                }

                #[test]
                fn every_sink_flushes() {
                    assert_eq!(Json.flush(), "ok");
                    assert_eq!(Csv.flush(), "ok");
                }
            """),
        },
        "reference": {
            "src/json.rs": code("""
                //! JSON sink.

                use crate::pipeline::Sink;

                pub struct Json;

                impl Sink for Json {
                    fn write(&self, value: i32) -> String {
                        format!("{{\\\"v\\\":{value}}}")
                    }

                    fn flush(&self) -> String {
                        "ok".to_string()
                    }
                }
            """),
            "src/csv.rs": code("""
                //! CSV sink.

                use crate::pipeline::Sink;

                pub struct Csv;

                impl Sink for Csv {
                    fn write(&self, value: i32) -> String {
                        format!("{value}")
                    }

                    fn flush(&self) -> String {
                        "ok".to_string()
                    }
                }
            """),
        },
        "mutation": {"file": "src/json.rs", "find": '"ok"', "replace": '"no"'},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0046"]},
    },
]
