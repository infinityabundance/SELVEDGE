"""Dependency-version compatibility reasoning, batch two (six tasks).

Every task is offline: the dependency is vendored inside the task and the
failure is a real Cargo or compiler diagnostic. The arm may edit the manifest
where the task says so, subject to the harness's manifest policy.
"""

from .common import code


def _dep_manifest(name: str, features: str = "") -> str:
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


TASKS = [
    {
        "id": "deps-04-path-version-requirement",
        "category": "dependency-compat",
        "difficulty": "easy",
        "capability": "Diagnose a version requirement that excludes the only vendored version.",
        "prompt_intro": (
            "This crate does not build. Its manifest requires `mathx` version 3, "
            "but the only vendored copy is 2.0.0 and its API is unchanged. Correct "
            "the version requirement. You may edit `Cargo.toml`; do not modify "
            "`dep/`. Keep `pub fn twice(value: i64) -> i64`."),
        "observed_failure": (
            "error: failed to select a version for the requirement `mathx = \"^3\"`\n"
            "candidate versions found which didn't match: 2.0.0\n"
            "location searched: `/work/.../dep`\n"
            "required by package `deps_04_path_version_requirement v0.1.0`"),
        "manifest_policy": "arm",
        "manifest": _app("deps_04_path_version_requirement",
                         'mathx = { path = "dep", version = "3" }'),
        "files": {
            "dep/Cargo.toml": _dep_manifest("mathx"),
            "dep/src/lib.rs": code("""
                //! Integer arithmetic, version 2.

                pub fn twice(value: i64) -> i64 {
                    value * 2
                }

                pub fn label() -> &'static str {
                    "mathx 2.0.0"
                }
            """),
            "src/lib.rs": code("""
                //! Double through the vendored crate.

                pub fn twice(value: i64) -> i64 {
                    mathx::twice(value)
                }

                pub fn dependency_label() -> &'static str {
                    mathx::label()
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use deps_04_path_version_requirement::{dependency_label, twice};

                #[test]
                fn doubles_values() {
                    assert_eq!(twice(21), 42);
                }

                #[test]
                fn doubles_negatives() {
                    assert_eq!(twice(-4), -8);
                }

                #[test]
                fn doubles_zero() {
                    assert_eq!(twice(0), 0);
                }

                #[test]
                fn uses_the_vendored_version() {
                    assert_eq!(dependency_label(), "mathx 2.0.0");
                }
            """),
        },
        "readonly": ["dep/"],
        "reference": {
            "Cargo.toml": _app("deps_04_path_version_requirement",
                               'mathx = { path = "dep", version = "2" }'),
            "src/lib.rs": code("""
                //! Double through the vendored crate.

                pub fn twice(value: i64) -> i64 {
                    mathx::twice(value)
                }

                pub fn dependency_label() -> &'static str {
                    mathx::label()
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "mathx::twice(value)",
                     "replace": "mathx::twice(value) + 1"},
        "initial_failure": {"kind": "compile_error",
                            "must_contain": ["failed to select a version"]},
    },
    {
        "id": "deps-05-enable-feature",
        "category": "dependency-compat",
        "difficulty": "easy",
        "capability": "Read a dependency's feature declaration and enable the optional API.",
        "prompt_intro": (
            "This crate does not build. The vendored `codec` 2.0.0 exposes `json` "
            "only when its `json` feature is enabled. Enable that feature in "
            "`Cargo.toml`; do not modify `dep/`. Keep "
            "`pub fn encode(value: &str) -> String`."),
        "observed_failure": (
            "error[E0433]: failed to resolve: could not find `json` in `codec`\n"
            " --> src/lib.rs:3:12\n"
            "  |\n"
            "3 |     codec::json::to_json(value)\n"
            "  |            ^^^^ could not find `json` in `codec`"),
        "manifest_policy": "arm",
        "manifest": _app("deps_05_enable_feature",
                         'codec = { path = "dep" }'),
        "files": {
            "dep/Cargo.toml": _dep_manifest("codec", """
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
                //! Encode through the vendored crate. This does not build: the
                //! `json` feature is not enabled.

                pub fn encode(value: &str) -> String {
                    codec::json::to_json(value)
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use deps_05_enable_feature::encode;

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
            "Cargo.toml": _app("deps_05_enable_feature",
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
        "initial_failure": {"kind": "compile_error", "must_contain": ["could not find"]},
    },
    {
        "id": "deps-06-edition-reserved-keyword",
        "category": "dependency-compat",
        "difficulty": "easy",
        "capability": "Repair an identifier that edition 2024 reserves.",
        "prompt_intro": (
            "This crate does not build under its declared edition 2024. `gen` became "
            "a reserved keyword. Rename the local binding and keep the behaviour: "
            "`squares(count)` returns the squares of `0..count`. Keep the signature "
            "`pub fn squares(count: usize) -> Vec<usize>` and the edition."),
        "observed_failure": (
            "error: expected identifier, found reserved keyword `gen`\n"
            " --> src/lib.rs:3:9\n"
            "  |\n"
            "3 |     let gen = (0..count).collect::<Vec<_>>();\n"
            "  |         ^^^ expected identifier, found reserved keyword"),
        "manifest": code("""
            [package]
            name = "deps_06_edition_reserved_keyword"
            version = "0.1.0"
            edition = "2024"

            [workspace]

            [lib]
            path = "src/lib.rs"
        """),
        "files": {
            "src/lib.rs": code("""
                //! Squares of a prefix. This does not build under edition 2024:
                //! `gen` is a reserved keyword.

                pub fn squares(count: usize) -> Vec<usize> {
                    let gen = (0..count).collect::<Vec<_>>();
                    gen.iter().map(|n| n * n).collect()
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use deps_06_edition_reserved_keyword::squares;

                #[test]
                fn squares_the_prefix() {
                    assert_eq!(squares(4), vec![0, 1, 4, 9]);
                }

                #[test]
                fn empty_prefix_is_empty() {
                    assert!(squares(0).is_empty());
                }

                #[test]
                fn one_element_is_zero() {
                    assert_eq!(squares(1), vec![0]);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Squares of a prefix.

                pub fn squares(count: usize) -> Vec<usize> {
                    let generation = (0..count).collect::<Vec<_>>();
                    generation.iter().map(|n| n * n).collect()
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "n * n", "replace": "n + n"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["reserved keyword"]},
    },
    {
        "id": "deps-07-feature-conflict",
        "category": "dependency-compat",
        "difficulty": "medium",
        "capability": "Resolve two mutually exclusive features by selecting the registered one.",
        "prompt_intro": (
            "This crate does not build because the vendored `codec` 2.0.0 declares "
            "`fast` and `safe` mutually exclusive, and the manifest enables both. "
            "Select `safe` only. Do not modify `dep/`. Keep "
            "`pub fn selected() -> &'static str`, which must return the "
            "dependency's mode."),
        "observed_failure": (
            "error: features `fast` and `safe` are mutually exclusive\n"
            " --> dep/src/lib.rs:1:1\n"
            "  |\n"
            "1 | compile_error!(\"features `fast` and `safe` are mutually exclusive\");\n"
            "  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"),
        "manifest_policy": "arm",
        "manifest": _app("deps_07_feature_conflict",
                         'codec = { path = "dep", features = ["fast", "safe"] }'),
        "files": {
            "dep/Cargo.toml": _dep_manifest("codec", """
                [features]
                default = []
                fast = []
                safe = []
            """),
            "dep/src/lib.rs": code("""
                //! Mode selection, version 2. `fast` and `safe` are exclusive.

                #[cfg(all(feature = "fast", feature = "safe"))]
                compile_error!("features `fast` and `safe` are mutually exclusive");

                pub fn mode() -> &'static str {
                    #[cfg(feature = "fast")]
                    {
                        return "fast";
                    }
                    #[cfg(feature = "safe")]
                    {
                        return "safe";
                    }
                    "default"
                }
            """),
            "src/lib.rs": code("""
                //! Select the vendored mode.

                pub fn selected() -> &'static str {
                    codec::mode()
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use deps_07_feature_conflict::selected;

                #[test]
                fn selects_the_safe_mode() {
                    assert_eq!(selected(), "safe");
                }

                #[test]
                fn the_selection_is_stable() {
                    assert_eq!(selected(), selected());
                }

                #[test]
                fn the_fast_mode_is_not_selected() {
                    assert_ne!(selected(), "fast");
                }
            """),
        },
        "readonly": ["dep/"],
        "reference": {
            "Cargo.toml": _app("deps_07_feature_conflict",
                               'codec = { path = "dep", features = ["safe"] }'),
            "src/lib.rs": code("""
                //! Select the vendored mode.

                pub fn selected() -> &'static str {
                    codec::mode()
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "codec::mode()", "replace": '"fast"'},
        "initial_failure": {"kind": "compile_error",
                            "must_contain": ["mutually exclusive"]},
    },
    {
        "id": "deps-08-renamed-package",
        "category": "dependency-compat",
        "difficulty": "hard",
        "capability": "Alias a vendored package whose package name differs from the import name.",
        "prompt_intro": (
            "This crate does not build. The vendored dependency's package is named "
            "`textkit-impl`, but the code imports it as `textkit`. Add the manifest "
            "field that maps the import name to the package name. Do not modify "
            "`dep/`. Keep `pub fn normalise(value: &str) -> String`, which trims and "
            "upper-cases its input."),
        "observed_failure": (
            "error: no matching package named `textkit` found\n"
            "location searched: `/work/.../dep`\n"
            "required by package `deps_08_renamed_package v0.1.0`"),
        "manifest_policy": "arm",
        "manifest": _app("deps_08_renamed_package",
                         'textkit = { path = "dep" }'),
        "files": {
            "dep/Cargo.toml": _dep_manifest("textkit-impl"),
            "dep/src/lib.rs": code("""
                //! Text normalisation, version 2. The package is `textkit-impl`.

                pub fn normalise(value: &str) -> String {
                    value.trim().to_uppercase()
                }
            """),
            "src/lib.rs": code("""
                //! Normalise through the vendored package.

                pub fn normalise(value: &str) -> String {
                    textkit::normalise(value)
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use deps_08_renamed_package::normalise;

                #[test]
                fn trims_and_upper_cases() {
                    assert_eq!(normalise("  alpha  "), "ALPHA");
                }

                #[test]
                fn empty_stays_empty() {
                    assert_eq!(normalise("   "), "");
                }

                #[test]
                fn mixed_case_is_upper_cased() {
                    assert_eq!(normalise("Mixed"), "MIXED");
                }
            """),
        },
        "readonly": ["dep/"],
        "reference": {
            "Cargo.toml": _app("deps_08_renamed_package",
                               'textkit = { path = "dep", package = "textkit-impl" }'),
            "src/lib.rs": code("""
                //! Normalise through the vendored package.

                pub fn normalise(value: &str) -> String {
                    textkit::normalise(value)
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "textkit::normalise(value)",
                     "replace": "textkit::normalise(value).to_lowercase()"},
        "initial_failure": {"kind": "compile_error",
                            "must_contain": ["no matching package named"]},
    },
    {
        "id": "deps-09-disable-default-features",
        "category": "dependency-compat",
        "difficulty": "hard",
        "capability": "Disable a default feature and enable the API-bearing alternative.",
        "prompt_intro": (
            "This crate does not build. The vendored `codec` 2.0.0 enables its "
            "`legacy` feature by default, which hides `render`; `render` exists only "
            "with the `modern` feature. Disable the defaults and enable `modern`. Do "
            "not modify `dep/`. Keep `pub fn render(value: &str) -> String`, which "
            "trims and upper-cases."),
        "observed_failure": (
            "error[E0425]: cannot find function `render` in crate `codec`\n"
            " --> src/lib.rs:3:12\n"
            "  |\n"
            "3 |     codec::render(value)\n"
            "  |            ^^^^^^ not found in `codec`"),
        "manifest_policy": "arm",
        "manifest": _app("deps_09_disable_default_features",
                         'codec = { path = "dep" }'),
        "files": {
            "dep/Cargo.toml": _dep_manifest("codec", """
                [features]
                default = ["legacy"]
                legacy = []
                modern = []
            """),
            "dep/src/lib.rs": code("""
                //! Rendering, version 2. `render` requires `modern`; the default is
                //! the legacy API.

                #[cfg(feature = "modern")]
                pub fn render(value: &str) -> String {
                    value.trim().to_uppercase()
                }

                #[cfg(feature = "legacy")]
                pub fn render_legacy(value: &str) -> String {
                    value.to_string()
                }
            """),
            "src/lib.rs": code("""
                //! Render through the vendored crate. This does not build: the
                //! default feature enables the legacy API only.

                pub fn render(value: &str) -> String {
                    codec::render(value)
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use deps_09_disable_default_features::render;

                #[test]
                fn trims_and_upper_cases() {
                    assert_eq!(render("  alpha  "), "ALPHA");
                }

                #[test]
                fn empty_stays_empty() {
                    assert_eq!(render("   "), "");
                }

                #[test]
                fn mixed_case_is_upper_cased() {
                    assert_eq!(render("Mixed"), "MIXED");
                }
            """),
        },
        "readonly": ["dep/"],
        "reference": {
            "Cargo.toml": _app(
                "deps_09_disable_default_features",
                'codec = { path = "dep", default-features = false, features = ["modern"] }'),
            "src/lib.rs": code("""
                //! Render through the vendored crate.

                pub fn render(value: &str) -> String {
                    codec::render(value)
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "codec::render(value)",
                     "replace": "codec::render(value).to_lowercase()"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0425"]},
    },
]
