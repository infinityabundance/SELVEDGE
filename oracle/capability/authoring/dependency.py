"""Dependency-version compatibility reasoning (three tasks).

All three are offline: the dependency is vendored inside the task, and the
failure is a real Cargo or compiler diagnostic (version resolution, edition
rules, an optional feature). No network, no registry, no lockfile guessing.
"""

_SCALING_MANIFEST = (
    "[package]\n"
    'name = "scaling"\n'
    'version = "2.0.0"\n'
    'edition = "2021"\n'
    "\n"
    "[lib]\n"
    'path = "src/lib.rs"\n'
)

_TEXTKIT_MANIFEST = (
    "[package]\n"
    'name = "textkit"\n'
    'version = "2.0.0"\n'
    'edition = "2021"\n'
    "\n"
    "[features]\n"
    "default = []\n"
    'fast = []\n'
    "\n"
    "[lib]\n"
    'path = "src/lib.rs"\n'
)

TASKS = [
    {
        "id": "deps-01-version-requirement",
        "category": "dependency-compat",
        "capability": "Diagnose a semver requirement that excludes the vendored version, then adapt the API.",
        "prompt_intro": (
            "This crate does not build. Its manifest requires `scaling` version 1.2,\n"
            "but the vendored dependency is 2.0.0, and version 2 changed `scale` from\n"
            "an integer factor to an `f64` factor. Update the dependency requirement to\n"
            "the vendored major version and change `scale_percent` so that\n"
            "`scale_percent(value, percent)` returns `value * percent / 100.0`. You may\n"
            "edit `Cargo.toml`; do not modify `dep/`."),
        "observed_failure": (
            "error: failed to select a version for the requirement `scaling = \"^1.2\"`\n"
            "candidate versions found which didn't match: 2.0.0\n"
            "location searched: `/work/.../dep`\n"
            "required by package `deps_01_version_requirement v0.1.0`"),
        "manifest_policy": "arm",
        "manifest": (
            "[package]\n"
            'name = "deps_01_version_requirement"\n'
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
            'scaling = { path = "dep", version = "1.2" }\n'),
        "files": {
            "dep/Cargo.toml": _SCALING_MANIFEST,
            "dep/src/lib.rs": (
                "//! Scaling, version 2. `scale` takes an `f64` factor; version 1 took an\n"
                "//! `i64` percentage.\n"
                "\n"
                "pub fn scale(value: f64, factor: f64) -> f64 {\n"
                "    value * factor\n"
                "}\n"
                "\n"
                "pub fn label() -> &'static str {\n"
                "    \"scaling 2.0.0\"\n"
                "}\n"),
            "src/lib.rs": (
                "//! Percentage scaling through the vendored crate.\n"
                "\n"
                "pub fn scale_percent(value: f64, percent: i64) -> f64 {\n"
                "    scaling::scale(value, percent)\n"
                "}\n"
                "\n"
                "pub fn dependency_label() -> &'static str {\n"
                "    scaling::label()\n"
                "}\n"),
        },
        "hidden": {
            "tests/check.rs": (
                "use deps_01_version_requirement::{dependency_label, scale_percent};\n"
                "\n"
                "#[test]\n"
                "fn scales_by_percentage() {\n"
                "    assert_eq!(scale_percent(200.0, 50), 100.0);\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn zero_percent_is_zero() {\n"
                "    assert_eq!(scale_percent(10.0, 0), 0.0);\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn fractional_percent() {\n"
                "    assert!((scale_percent(-5.0, 10) - (-0.5)).abs() < 1e-12);\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn the_vendored_version_is_the_one_in_use() {\n"
                "    assert_eq!(dependency_label(), \"scaling 2.0.0\");\n"
                "}\n"),
        },
        "readonly": ["dep/"],
        "reference": {
            "Cargo.toml": (
                "[package]\n"
                'name = "deps_01_version_requirement"\n'
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
                'scaling = { path = "dep", version = "2" }\n'),
            "src/lib.rs": (
                "//! Percentage scaling through the vendored crate.\n"
                "\n"
                "pub fn scale_percent(value: f64, percent: i64) -> f64 {\n"
                "    scaling::scale(value, percent as f64 / 100.0)\n"
                "}\n"
                "\n"
                "pub fn dependency_label() -> &'static str {\n"
                "    scaling::label()\n"
                "}\n"),
        },
        "mutation": {"file": "src/lib.rs", "find": "percent as f64 / 100.0", "replace": "percent as f64"},
        "initial_failure": {"kind": "compile_error",
                            "must_contain": ["failed to select a version"]},
    },
    {
        "id": "deps-02-edition-2024",
        "category": "dependency-compat",
        "capability": "Repair code that violates edition-2024 rules (reserved identifier, binding modes).",
        "prompt_intro": (
            "The crate declares edition 2024 but was written for edition 2021 and does\n"
            "not build. `gen` is a reserved keyword in edition 2024, and a `ref` binding\n"
            "modifier is not allowed when the default binding mode is not `move`.\n"
            "Rename the binding and remove the modifier; do not change the edition.\n"
            "Keep both function signatures and their behaviour."),
        "observed_failure": (
            "error: expected identifier, found keyword `gen`\n"
            " --> src/lib.rs:8:9\n"
            "  |\n"
            "8 |     let gen = seed.wrapping_mul(6364136223846793005).wrapping_add(1);\n"
            "  |         ^^^ expected identifier, found keyword\n"
            "\n"
            "error: binding modifiers may only be written when the default binding mode\n"
            "is `move`\n"
            " --> src/lib.rs:4:14\n"
            "  |\n"
            "4 |         Some(ref inner) => format!(\"some {inner}\"),\n"
            "  |              ^^^ binding modifier not allowed under `ref` default binding mode"),
        "manifest": (
            "[package]\n"
            'name = "deps_02_edition_2024"\n'
            'version = "0.1.0"\n'
            'edition = "2024"\n'
            "\n"
            "[workspace]\n"
            "\n"
            "[lib]\n"
            'path = "src/lib.rs"\n'),
        "files": {
            "src/lib.rs": (
                "//! Written for Rust 2021; this crate declares edition 2024 and no\n"
                "//! longer compiles.\n"
                "\n"
                "pub fn describe(value: &Option<u32>) -> String {\n"
                "    match value {\n"
                "        Some(ref inner) => format!(\"some {inner}\"),\n"
                "        None => \"none\".to_string(),\n"
                "    }\n"
                "}\n"
                "\n"
                "pub fn next_value(seed: u64) -> u64 {\n"
                "    let gen = seed.wrapping_mul(6364136223846793005).wrapping_add(1);\n"
                "    gen % 100\n"
                "}\n"),
        },
        "hidden": {
            "tests/check.rs": (
                "use deps_02_edition_2024::{describe, next_value};\n"
                "\n"
                "#[test]\n"
                "fn describes_some() {\n"
                "    assert_eq!(describe(&Some(3)), \"some 3\");\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn describes_none() {\n"
                "    assert_eq!(describe(&None), \"none\");\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn values_are_bounded_and_distinct() {\n"
                "    assert_eq!(next_value(0), 1);\n"
                "    assert!(next_value(0) < 100 && next_value(1) < 100);\n"
                "    assert_ne!(next_value(0), next_value(1));\n"
                "}\n"),
        },
        "reference": {
            "src/lib.rs": (
                "//! Edition 2024.\n"
                "\n"
                "pub fn describe(value: &Option<u32>) -> String {\n"
                "    match value {\n"
                "        Some(inner) => format!(\"some {inner}\"),\n"
                "        None => \"none\".to_string(),\n"
                "    }\n"
                "}\n"
                "\n"
                "pub fn next_value(seed: u64) -> u64 {\n"
                "    let state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);\n"
                "    state % 100\n"
                "}\n"),
        },
        "mutation": {"file": "src/lib.rs", "find": "some {inner}",
                     "replace": "some{}"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["gen"]},
    },
    {
        "id": "deps-03-optional-feature",
        "category": "dependency-compat",
        "capability": "Read a dependency's feature declaration and enable the optional API at the caller.",
        "prompt_intro": (
            "This crate does not build. The vendored `textkit` 2.0.0 exposes\n"
            "`normalise_fast` and `backend` only behind its non-default `fast` feature.\n"
            "Enable the feature from the application's manifest; do not modify `dep/`\n"
            "and do not reimplement the functions. Keep the signatures."),
        "observed_failure": (
            "error[E0425]: cannot find function `normalise_fast` in crate `textkit`\n"
            " --> src/lib.rs:2:40\n"
            "  |\n"
            "2 |     textkit::normalise_fast(input)\n"
            "  |              ^^^^^^^^^^^^^^ not found in `textkit`"),
        "manifest_policy": "arm",
        "manifest": (
            "[package]\n"
            'name = "deps_03_optional_feature"\n'
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
            'textkit = { path = "dep" }\n'),
        "files": {
            "dep/Cargo.toml": _TEXTKIT_MANIFEST,
            "dep/src/lib.rs": (
                "//! Text handling, version 2: the fast path is an optional feature.\n"
                "\n"
                "#[cfg(feature = \"fast\")]\n"
                "pub fn normalise_fast(input: &str) -> String {\n"
                "    input\n"
                "        .split_whitespace()\n"
                "        .collect::<Vec<_>>()\n"
                "        .join(\" \")\n"
                "        .to_lowercase()\n"
                "}\n"
                "\n"
                "#[cfg(feature = \"fast\")]\n"
                "pub fn backend() -> &'static str {\n"
                "    \"fast\"\n"
                "}\n"),
            "src/lib.rs": (
                "//! Text folding through the vendored crate.\n"
                "\n"
                "pub fn fold(input: &str) -> String {\n"
                "    textkit::normalise_fast(input)\n"
                "}\n"
                "\n"
                "pub fn backend() -> &'static str {\n"
                "    textkit::backend()\n"
                "}\n"),
        },
        "hidden": {
            "tests/check.rs": (
                "use deps_03_optional_feature::{backend, fold};\n"
                "\n"
                "#[test]\n"
                "fn folds_whitespace_and_case() {\n"
                "    assert_eq!(fold(\"  A\\tB  \"), \"a b\");\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn the_fast_backend_is_enabled() {\n"
                "    assert_eq!(backend(), \"fast\");\n"
                "}\n"),
        },
        "readonly": ["dep/"],
        "reference": {
            "Cargo.toml": (
                "[package]\n"
                'name = "deps_03_optional_feature"\n'
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
                'textkit = { path = "dep", features = ["fast"] }\n'),
        },
        "mutation": {"file": "Cargo.toml", "find": 'features = ["fast"]',
                     "replace": 'features = []'},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0425"]},
    },
]
