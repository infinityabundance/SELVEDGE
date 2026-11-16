"""Repo-scale repairs across several files (three tasks).

Each crate has a module tree of four or five files. The failure cannot be fixed
in one place: a signature, a bound or an error conversion must be propagated
through the modules that call it. Hidden tests exercise the public entry points
of every module.
"""

TASKS = [
    {
        "id": "repo-01-lifetime-propagation",
        "category": "repo-scale",
        "capability": "Propagate one lifetime annotation through a three-module call chain.",
        "prompt_intro": (
            "This crate does not build. `query::cheapest` elides a lifetime across two\n"
            "input references, and `lib::cheapest_name` does the same. Annotate the\n"
            "lifetime so the returned reference is tied to the item slice, and thread\n"
            "the same lifetime through `report::describe_cheapest`. Semantics: return\n"
            "the cheapest item, or the fallback when the slice is empty."),
        "observed_failure": (
            "error[E0106]: missing lifetime specifier\n"
            " --> src/query.rs:3:54\n"
            "  |\n"
            "3 | pub fn cheapest(items: &[Item], fallback: &Item) -> Option<&Item> {\n"
            "  |                           ----          ----            ^ expected named lifetime parameter\n"
            "  |\n"
            "  = help: this function's return type contains a borrowed value, but the\n"
            "          signature does not say whether it is borrowed from `items` or `fallback`"),
        "files": {
            "src/lib.rs": (
                "//! Prices, queries and reports.\n"
                "\n"
                "pub mod model;\n"
                "pub mod query;\n"
                "pub mod report;\n"
                "pub mod store;\n"
                "\n"
                "pub use model::Item;\n"
                "\n"
                "pub fn cheapest_name(items: &[Item], fallback: &Item) -> &str {\n"
                "    query::cheapest(items, fallback).name.as_str()\n"
                "}\n"),
            "src/model.rs": (
                "//! The item type.\n"
                "\n"
                "#[derive(Debug, Clone, PartialEq)]\n"
                "pub struct Item {\n"
                "    pub name: String,\n"
                "    pub price: u32,\n"
                "}\n"),
            "src/store.rs": (
                "//! Lookup by name.\n"
                "\n"
                "use crate::model::Item;\n"
                "\n"
                "pub fn find<'a>(items: &'a [Item], name: &str) -> Option<&'a Item> {\n"
                "    items.iter().find(|item| item.name == name)\n"
                "}\n"),
            "src/query.rs": (
                "//! The cheapest item.\n"
                "\n"
                "use crate::model::Item;\n"
                "\n"
                "pub fn cheapest(items: &[Item], fallback: &Item) -> Option<&Item> {\n"
                "    items.iter().min_by_key(|item| item.price).or(Some(fallback))\n"
                "}\n"),
            "src/report.rs": (
                "//! Render a one-line report.\n"
                "\n"
                "use crate::model::Item;\n"
                "use crate::query;\n"
                "\n"
                "pub fn describe_cheapest(items: &[Item], fallback: &Item) -> String {\n"
                "    let item = query::cheapest(items, fallback).unwrap();\n"
                "    format!(\"{} costs {}\", item.name, item.price)\n"
                "}\n"),
        },
        "hidden": {
            "tests/check.rs": (
                "use repo_01_lifetime_propagation::{cheapest_name, report, store, Item};\n"
                "\n"
                "fn item(name: &str, price: u32) -> Item {\n"
                "    Item { name: name.to_string(), price }\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn names_the_cheapest() {\n"
                "    let items = vec![item(\"b\", 5), item(\"a\", 2)];\n"
                "    let fallback = item(\"z\", 99);\n"
                "    assert_eq!(cheapest_name(&items, &fallback), \"a\");\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn uses_the_fallback_when_empty() {\n"
                "    let items: Vec<Item> = vec![];\n"
                "    let fallback = item(\"z\", 99);\n"
                "    assert_eq!(cheapest_name(&items, &fallback), \"z\");\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn describes_the_cheapest() {\n"
                "    let items = vec![item(\"b\", 5)];\n"
                "    let fallback = item(\"z\", 99);\n"
                "    assert_eq!(report::describe_cheapest(&items, &fallback), \"b costs 5\");\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn store_finds_by_name() {\n"
                "    let items = vec![item(\"b\", 5)];\n"
                "    assert_eq!(store::find(&items, \"b\").map(|item| item.price), Some(5));\n"
                "}\n"),
        },
        "reference": {
            "src/lib.rs": (
                "//! Prices, queries and reports.\n"
                "\n"
                "pub mod model;\n"
                "pub mod query;\n"
                "pub mod report;\n"
                "pub mod store;\n"
                "\n"
                "pub use model::Item;\n"
                "\n"
                "pub fn cheapest_name<'a>(items: &'a [Item], fallback: &'a Item) -> &'a str {\n"
                "    query::cheapest(items, fallback).name.as_str()\n"
                "}\n"),
            "src/query.rs": (
                "//! The cheapest item.\n"
                "\n"
                "use crate::model::Item;\n"
                "\n"
                "pub fn cheapest<'a>(items: &'a [Item], fallback: &'a Item) -> &'a Item {\n"
                "    items.iter().min_by_key(|item| item.price).unwrap_or(fallback)\n"
                "}\n"),
            "src/report.rs": (
                "//! Render a one-line report.\n"
                "\n"
                "use crate::model::Item;\n"
                "use crate::query;\n"
                "\n"
                "pub fn describe_cheapest<'a>(items: &'a [Item], fallback: &'a Item) -> String {\n"
                "    let item = query::cheapest(items, fallback);\n"
                "    format!(\"{} costs {}\", item.name, item.price)\n"
                "}\n"),
        },
        "mutation": {"file": "src/query.rs", "find": "min_by_key(|item| item.price)",
                     "replace": "max_by_key(|item| item.price)"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0106"]},
    },
    {
        "id": "repo-02-error-propagation",
        "category": "repo-scale",
        "capability": "Extend a shared error type with conversions so `?` works in a service module.",
        "prompt_intro": (
            "This crate does not build. `service::import` uses `?` on a parse error and\n"
            "on an I/O error, but `AppError` has neither the variants nor the `From`\n"
            "conversions. Add `AppError::Parse(usize)` and `AppError::Io` and the two\n"
            "conversions in `src/error.rs`, without changing `parse.rs`, `store.rs` or\n"
            "`service.rs`."),
        "observed_failure": (
            "error[E0277]: `?` couldn't convert the error to `AppError`\n"
            " --> src/service.rs:8:36\n"
            "  |\n"
            "8 |     for line in lines { rows.push(parse_row(line)?); }\n"
            "  |                                    ^^^^^^^^^^^^^^^ the trait\n"
            "  |                                    `From<ParseError>` is not implemented for `AppError`"),
        "files": {
            "src/lib.rs": (
                "//! Import rows into a file.\n"
                "\n"
                "pub mod error;\n"
                "pub mod parse;\n"
                "pub mod service;\n"
                "pub mod store;\n"),
            "src/error.rs": (
                "//! The application error.\n"
                "\n"
                "#[derive(Debug, PartialEq)]\n"
                "pub enum AppError {\n"
                "    Missing,\n"
                "}\n"
                "\n"
                "impl core::fmt::Display for AppError {\n"
                "    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {\n"
                "        match self {\n"
                "            AppError::Missing => write!(formatter, \"missing\"),\n"
                "        }\n"
                "    }\n"
                "}\n"
                "\n"
                "impl std::error::Error for AppError {}\n"),
            "src/parse.rs": (
                "//! Parse one CSV row.\n"
                "\n"
                "#[derive(Debug, PartialEq)]\n"
                "pub struct ParseError {\n"
                "    pub field: usize,\n"
                "}\n"
                "\n"
                "impl core::fmt::Display for ParseError {\n"
                "    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {\n"
                "        write!(formatter, \"malformed field {}\", self.field)\n"
                "    }\n"
                "}\n"
                "\n"
                "impl std::error::Error for ParseError {}\n"
                "\n"
                "pub fn parse_row(text: &str) -> Result<(String, i64), ParseError> {\n"
                "    let mut parts = text.split(',');\n"
                "    let name = parts.next().ok_or(ParseError { field: 0 })?;\n"
                "    let value = parts\n"
                "        .next()\n"
                "        .ok_or(ParseError { field: 1 })?\n"
                "        .trim()\n"
                "        .parse::<i64>()\n"
                "        .map_err(|_| ParseError { field: 1 })?;\n"
                "    Ok((name.to_string(), value))\n"
                "}\n"),
            "src/store.rs": (
                "//! Write rows to a file.\n"
                "\n"
                "use std::io;\n"
                "\n"
                "pub fn write_all(path: &str, rows: &[(String, i64)]) -> io::Result<()> {\n"
                "    let mut text = String::new();\n"
                "    for (name, value) in rows {\n"
                "        text.push_str(&format!(\"{name},{value}\\n\"));\n"
                "    }\n"
                "    std::fs::write(path, text)\n"
                "}\n"),
            "src/service.rs": (
                "//! Import a batch of rows.\n"
                "\n"
                "use crate::error::AppError;\n"
                "use crate::parse::parse_row;\n"
                "use crate::store::write_all;\n"
                "\n"
                "pub fn import(path: &str, lines: &[String]) -> Result<usize, AppError> {\n"
                "    let mut rows = Vec::new();\n"
                "    for line in lines {\n"
                "        rows.push(parse_row(line)?);\n"
                "    }\n"
                "    write_all(path, &rows)?;\n"
                "    Ok(rows.len())\n"
                "}\n"),
        },
        "hidden": {
            "tests/check.rs": (
                "use repo_02_error_propagation::{error::AppError, service::import};\n"
                "\n"
                "#[test]\n"
                "fn imports_rows() {\n"
                "    let path = std::env::temp_dir().join(\"repo-02-rows.txt\");\n"
                "    let lines = vec![\"a,1\".to_string(), \"b,2\".to_string()];\n"
                "    assert_eq!(import(path.to_str().unwrap(), &lines), Ok(2));\n"
                "    assert_eq!(std::fs::read_to_string(&path).unwrap(), \"a,1\\nb,2\\n\");\n"
                "    let _ = std::fs::remove_file(&path);\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn reports_the_parse_field() {\n"
                "    let path = std::env::temp_dir().join(\"repo-02-unused.txt\");\n"
                "    let lines = vec![\"a,1\".to_string(), \"b,x\".to_string()];\n"
                "    assert_eq!(import(path.to_str().unwrap(), &lines), Err(AppError::Parse(1)));\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn reports_io_errors() {\n"
                "    let lines = vec![\"a,1\".to_string()];\n"
                "    assert!(matches!(import(\"/nonexistent-dir/rows.txt\", &lines), Err(AppError::Io)));\n"
                "}\n"),
        },
        "reference": {
            "src/error.rs": (
                "//! The application error.\n"
                "\n"
                "use crate::parse::ParseError;\n"
                "\n"
                "#[derive(Debug, PartialEq)]\n"
                "pub enum AppError {\n"
                "    Missing,\n"
                "    Parse(usize),\n"
                "    Io,\n"
                "}\n"
                "\n"
                "impl core::fmt::Display for AppError {\n"
                "    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {\n"
                "        match self {\n"
                "            AppError::Missing => write!(formatter, \"missing\"),\n"
                "            AppError::Parse(field) => write!(formatter, \"malformed field {field}\"),\n"
                "            AppError::Io => write!(formatter, \"io error\"),\n"
                "        }\n"
                "    }\n"
                "}\n"
                "\n"
                "impl std::error::Error for AppError {}\n"
                "\n"
                "impl From<ParseError> for AppError {\n"
                "    fn from(error: ParseError) -> Self {\n"
                "        AppError::Parse(error.field)\n"
                "    }\n"
                "}\n"
                "\n"
                "impl From<std::io::Error> for AppError {\n"
                "    fn from(_error: std::io::Error) -> Self {\n"
                "        AppError::Io\n"
                "    }\n"
                "}\n"),
        },
        "mutation": {"file": "src/error.rs", "find": "AppError::Parse(error.field)",
                     "replace": "AppError::Parse(0)"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0277"]},
    },
    {
        "id": "repo-03-generic-bounds",
        "category": "repo-scale",
        "capability": "Add a missing trait impl and bound shared by three modules.",
        "prompt_intro": (
            "This crate does not build. `lib::total_u32` passes `&[u32]` to a function\n"
            "bounded by `AsI64`, but `u32` has no impl, and `stats::mean_floor` forgets\n"
            "the bound entirely. Implement `AsI64` for `u32`, add the missing bound, and\n"
            "make `mean_floor` floor towards negative infinity. Do not change the trait\n"
            "definition or `math::sum`."),
        "observed_failure": (
            "error[E0277]: the trait bound `u32: AsI64` is not satisfied\n"
            " --> src/lib.rs:7:21\n"
            "  |\n"
            "7 |     math::sum(items)\n"
            "  |     ---------- ^^^^^ the trait `AsI64` is not implemented for `u32`"),
        "files": {
            "src/lib.rs": (
                "//! Sums and means over integer types.\n"
                "\n"
                "pub mod math;\n"
                "pub mod stats;\n"
                "pub mod traits;\n"
                "\n"
                "pub fn total_u32(items: &[u32]) -> i64 {\n"
                "    math::sum(items)\n"
                "}\n"),
            "src/traits.rs": (
                "//! The conversion trait shared by the modules.\n"
                "\n"
                "pub trait AsI64 {\n"
                "    fn as_i64(&self) -> i64;\n"
                "}\n"
                "\n"
                "impl AsI64 for i32 {\n"
                "    fn as_i64(&self) -> i64 {\n"
                "        i64::from(*self)\n"
                "    }\n"
                "}\n"
                "\n"
                "impl AsI64 for i64 {\n"
                "    fn as_i64(&self) -> i64 {\n"
                "        *self\n"
                "    }\n"
                "}\n"),
            "src/math.rs": (
                "//! Sum a slice of anything convertible to i64.\n"
                "\n"
                "use crate::traits::AsI64;\n"
                "\n"
                "pub fn sum<T: AsI64>(items: &[T]) -> i64 {\n"
                "    items.iter().map(|value| value.as_i64()).sum()\n"
                "}\n"),
            "src/stats.rs": (
                "//! Flooring mean.\n"
                "\n"
                "pub fn mean_floor<T>(items: &[T]) -> i64 {\n"
                "    if items.is_empty() {\n"
                "        return 0;\n"
                "    }\n"
                "    let total: i64 = items.iter().map(|value| value.as_i64()).sum();\n"
                "    total / items.len() as i64\n"
                "}\n"),
        },
        "hidden": {
            "tests/check.rs": (
                "use repo_03_generic_bounds::{math, stats, total_u32, traits::AsI64};\n"
                "\n"
                "#[test]\n"
                "fn sums_u32() {\n"
                "    assert_eq!(math::sum(&[1u32, 2, 3]), 6);\n"
                "    assert_eq!(total_u32(&[4, 5]), 9);\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn sums_i64() {\n"
                "    assert_eq!(math::sum(&[1i64, -2, 3]), 2);\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn mean_floor_floors() {\n"
                "    assert_eq!(stats::mean_floor(&[1i32, 2, 2]), 1);\n"
                "    assert_eq!(stats::mean_floor(&[-3i64, -2]), -3);\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn empty_mean_is_zero() {\n"
                "    assert_eq!(stats::mean_floor(&[] as &[i64]), 0);\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn the_trait_is_usable() {\n"
                "    assert_eq!(7u32.as_i64(), 7);\n"
                "}\n"),
        },
        "reference": {
            "src/traits.rs": (
                "//! The conversion trait shared by the modules.\n"
                "\n"
                "pub trait AsI64 {\n"
                "    fn as_i64(&self) -> i64;\n"
                "}\n"
                "\n"
                "impl AsI64 for i32 {\n"
                "    fn as_i64(&self) -> i64 {\n"
                "        i64::from(*self)\n"
                "    }\n"
                "}\n"
                "\n"
                "impl AsI64 for i64 {\n"
                "    fn as_i64(&self) -> i64 {\n"
                "        *self\n"
                "    }\n"
                "}\n"
                "\n"
                "impl AsI64 for u32 {\n"
                "    fn as_i64(&self) -> i64 {\n"
                "        i64::from(*self)\n"
                "    }\n"
                "}\n"),
            "src/stats.rs": (
                "//! Flooring mean.\n"
                "\n"
                "use crate::traits::AsI64;\n"
                "\n"
                "pub fn mean_floor<T: AsI64>(items: &[T]) -> i64 {\n"
                "    if items.is_empty() {\n"
                "        return 0;\n"
                "    }\n"
                "    let total: i64 = items.iter().map(|value| value.as_i64()).sum();\n"
                "    total.div_euclid(items.len() as i64)\n"
                "}\n"),
        },
        "mutation": {"file": "src/stats.rs", "find": ".div_euclid(", "replace": " / ("},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0277"]},
    },
]
