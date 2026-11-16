"""API migrations against a vendored dependency (four tasks).

The dependency directory is copied into the workdir by the harness and restored
after the arm writes its files, so "migrate the call sites" cannot be answered by
reverting the library. Each initial state fails to build against version 2 of the
vendored crate; the reference changes only the application code.
"""

_APP_MANIFEST = (
    "[package]\n"
    'name = "{name}"\n'
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
    '{dependency} = {{ path = "dep" }}\n'
)

_UNITS_MANIFEST = (
    "[package]\n"
    'name = "units"\n'
    'version = "2.0.0"\n'
    'edition = "2021"\n'
    "\n"
    "[lib]\n"
    'path = "src/lib.rs"\n'
)

_CLIENT_MANIFEST = (
    "[package]\n"
    'name = "client"\n'
    'version = "2.0.0"\n'
    'edition = "2021"\n'
    "\n"
    "[lib]\n"
    'path = "src/lib.rs"\n'
)

_TOKENS_MANIFEST = (
    "[package]\n"
    'name = "tokens"\n'
    'version = "2.0.0"\n'
    'edition = "2021"\n'
    "\n"
    "[lib]\n"
    'path = "src/lib.rs"\n'
)

_DOCUMENTS_MANIFEST = (
    "[package]\n"
    'name = "documents"\n'
    'version = "2.0.0"\n'
    'edition = "2021"\n'
    "\n"
    "[lib]\n"
    'path = "src/lib.rs"\n'
)

TASKS = [
    {
        "id": "migrate-01-signature",
        "category": "api-migration",
        "capability": "Migrate to an Option-returning parse API and preserve the caller's error text.",
        "prompt_intro": (
            "The crate does not build against the vendored `units` 2.0.0. In version "
            "1, `units::parse` returned `Result<Measurement, ParseError>`; in version "
            "2 it returns `Option<Measurement>` and `parse_strict` returns the "
            "`Result`. Do not modify `dep/`. Keep "
            "`pub fn load(input: &str) -> Result<f64, String>` and return the error "
            "string `invalid measurement` for any unparsable input."),
        "observed_failure": (
            "error[E0599]: no method named `map_err` found for enum `Option` in the\n"
            "current scope\n"
            " --> src/lib.rs:4:48\n"
            "  |\n"
            "4 |     let measurement = units::parse(input).map_err(|error| error.to_string())?;\n"
            "  |                                              ^^^^^^^ method not found in `Option<Measurement>`"),
        "manifest": _APP_MANIFEST.format(name="migrate_01_signature", dependency="units"),
        "files": {
            "dep/Cargo.toml": _UNITS_MANIFEST,
            "dep/src/lib.rs": (
                "//! Unit parsing, version 2. `parse` no longer reports why it failed;\n"
                "//! `parse_strict` does.\n"
                "\n"
                "#[derive(Debug, Clone, PartialEq)]\n"
                "pub struct Measurement {\n"
                "    pub value: f64,\n"
                "}\n"
                "\n"
                "#[derive(Debug, Clone, PartialEq)]\n"
                "pub enum ParseError {\n"
                "    Empty,\n"
                "    Malformed,\n"
                "}\n"
                "\n"
                "impl core::fmt::Display for ParseError {\n"
                "    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {\n"
                "        match self {\n"
                "            ParseError::Empty => write!(formatter, \"empty input\"),\n"
                "            ParseError::Malformed => write!(formatter, \"malformed measurement\"),\n"
                "        }\n"
                "    }\n"
                "}\n"
                "\n"
                "impl std::error::Error for ParseError {}\n"
                "\n"
                "fn number(input: &str) -> Option<f64> {\n"
                "    let digits: String = input\n"
                "        .chars()\n"
                "        .take_while(|character| character.is_ascii_digit() || *character == '.')\n"
                "        .collect();\n"
                "    digits.parse().ok()\n"
                "}\n"
                "\n"
                "pub fn parse(input: &str) -> Option<Measurement> {\n"
                "    if input.is_empty() {\n"
                "        return None;\n"
                "    }\n"
                "    number(input).map(|value| Measurement { value })\n"
                "}\n"
                "\n"
                "pub fn parse_strict(input: &str) -> Result<Measurement, ParseError> {\n"
                "    if input.is_empty() {\n"
                "        return Err(ParseError::Empty);\n"
                "    }\n"
                "    match number(input) {\n"
                "        Some(value) => Ok(Measurement { value }),\n"
                "        None => Err(ParseError::Malformed),\n"
                "    }\n"
                "}\n"),
            "src/lib.rs": (
                "//! Load a measurement through the unit crate. This does not build: the\n"
                "//! crate's `parse` returned a `Result` in version 1 and returns an\n"
                "//! `Option` in version 2.\n"
                "\n"
                "pub fn load(input: &str) -> Result<f64, String> {\n"
                "    let measurement = units::parse(input).map_err(|error| error.to_string())?;\n"
                "    Ok(measurement.value)\n"
                "}\n"),
        },
        "hidden": {
            "tests/check.rs": (
                "use migrate_01_signature::load;\n"
                "\n"
                "#[test]\n"
                "fn parses_a_value() {\n"
                "    assert_eq!(load(\"12.5kg\"), Ok(12.5));\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn rejects_garbage_with_the_registered_text() {\n"
                "    assert_eq!(load(\"nope\"), Err(\"invalid measurement\".to_string()));\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn rejects_empty_input() {\n"
                "    assert_eq!(load(\"\"), Err(\"invalid measurement\".to_string()));\n"
                "}\n"),
        },
        "readonly": ["dep/"],
        "reference": {
            "src/lib.rs": (
                "//! Load a measurement through the unit crate.\n"
                "\n"
                "pub fn load(input: &str) -> Result<f64, String> {\n"
                "    units::parse(input)\n"
                "        .map(|measurement| measurement.value)\n"
                "        .ok_or_else(|| \"invalid measurement\".to_string())\n"
                "}\n"),
        },
        "mutation": {"file": "src/lib.rs", "find": "invalid measurement", "replace": "invalid"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0599"]},
    },
    {
        "id": "migrate-02-builder",
        "category": "api-migration",
        "capability": "Replace a removed constructor with the builder API, including its error path.",
        "prompt_intro": (
            "The crate does not build against the vendored `client` 2.0.0. The\n"
            "constructor `Client::new(host, port)` was removed; version 2 provides\n"
            "`client::builder().host(..).port(..).build()`, which fails with\n"
            "`BuildError::EmptyHost` for an empty host. Do not modify `dep/`. Keep\n"
            "`pub fn connect(host: &str, port: u16, path: &str) -> Result<String, String>`\n"
            "and return the error string `empty host` when the host is empty."),
        "observed_failure": (
            "error[E0599]: no function or associated item named `new` found for struct\n"
            "`client::Client` in the current scope\n"
            " --> src/lib.rs:2:30\n"
            "  |\n"
            "2 |     let client = client::Client::new(host, port);\n"
            "  |                              ^^^ function or associated item not found"),
        "manifest": _APP_MANIFEST.format(name="migrate_02_builder", dependency="client"),
        "files": {
            "dep/Cargo.toml": _CLIENT_MANIFEST,
            "dep/src/lib.rs": (
                "//! Client construction, version 2: a builder replaces the constructor.\n"
                "\n"
                "#[derive(Debug, Clone, PartialEq)]\n"
                "pub struct Client {\n"
                "    host: String,\n"
                "    port: u16,\n"
                "}\n"
                "\n"
                "#[derive(Debug, Clone, PartialEq)]\n"
                "pub enum BuildError {\n"
                "    EmptyHost,\n"
                "}\n"
                "\n"
                "pub struct ClientBuilder {\n"
                "    host: Option<String>,\n"
                "    port: u16,\n"
                "}\n"
                "\n"
                "impl ClientBuilder {\n"
                "    pub fn host(mut self, host: &str) -> Self {\n"
                "        self.host = Some(host.to_string());\n"
                "        self\n"
                "    }\n"
                "\n"
                "    pub fn port(mut self, port: u16) -> Self {\n"
                "        self.port = port;\n"
                "        self\n"
                "    }\n"
                "\n"
                "    pub fn build(self) -> Result<Client, BuildError> {\n"
                "        match self.host {\n"
                "            Some(host) if !host.is_empty() => Ok(Client { host, port: self.port }),\n"
                "            _ => Err(BuildError::EmptyHost),\n"
                "        }\n"
                "    }\n"
                "}\n"
                "\n"
                "pub fn builder() -> ClientBuilder {\n"
                "    ClientBuilder { host: None, port: 80 }\n"
                "}\n"
                "\n"
                "impl Client {\n"
                "    pub fn endpoint(&self, path: &str) -> String {\n"
                "        format!(\"{}:{}{}\", self.host, self.port, path)\n"
                "    }\n"
                "}\n"),
            "src/lib.rs": (
                "//! Connect and build an endpoint. This does not build: the client\n"
                "//! constructor was removed in version 2.\n"
                "\n"
                "pub fn connect(host: &str, port: u16, path: &str) -> Result<String, String> {\n"
                "    let client = client::Client::new(host, port);\n"
                "    Ok(client.endpoint(path))\n"
                "}\n"),
        },
        "hidden": {
            "tests/check.rs": (
                "use migrate_02_builder::connect;\n"
                "\n"
                "#[test]\n"
                "fn builds_an_endpoint() {\n"
                "    assert_eq!(connect(\"example.org\", 8080, \"/x\"), Ok(\"example.org:8080/x\".to_string()));\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn rejects_an_empty_host() {\n"
                "    assert_eq!(connect(\"\", 80, \"/\"), Err(\"empty host\".to_string()));\n"
                "}\n"),
        },
        "readonly": ["dep/"],
        "reference": {
            "src/lib.rs": (
                "//! Connect and build an endpoint.\n"
                "\n"
                "pub fn connect(host: &str, port: u16, path: &str) -> Result<String, String> {\n"
                "    client::builder()\n"
                "        .host(host)\n"
                "        .port(port)\n"
                "        .build()\n"
                "        .map(|client| client.endpoint(path))\n"
                "        .map_err(|_| \"empty host\".to_string())\n"
                "}\n"),
        },
        "mutation": {"file": "src/lib.rs", "find": "empty host", "replace": "bad host"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0599"]},
    },
    {
        "id": "migrate-03-error-conversion",
        "category": "api-migration",
        "capability": "Add the conversion a new dependency error type requires at every `?` site.",
        "prompt_intro": (
            "The crate does not build against the vendored `tokens` 2.0.0. Its error\n"
            "type is now `tokens::Error` with variants `Empty` and `Malformed(usize)`,\n"
            "and the application's `AppError` has no conversion from it. Do not modify\n"
            "`dep/`. Add the conversion so `?` works, and map the variants exactly:\n"
            "`tokens::Error::Empty` becomes `AppError::Empty`, and\n"
            "`tokens::Error::Malformed(field)` becomes `AppError::Invalid { field }`."),
        "observed_failure": (
            "error[E0277]: `?` couldn't convert the error to `AppError`\n"
            " --> src/lib.rs:9:48\n"
            "  |\n"
            "9 |     let tokens = tokens::tokenize(text)?;\n"
            "  |                                            ^ the trait `From<tokens::Error>` is not implemented for `AppError`"),
        "manifest": _APP_MANIFEST.format(name="migrate_03_error_conversion", dependency="tokens"),
        "files": {
            "dep/Cargo.toml": _TOKENS_MANIFEST,
            "dep/src/lib.rs": (
                "//! Tokenisation, version 2: errors carry the failing field index.\n"
                "\n"
                "#[derive(Debug, Clone, PartialEq)]\n"
                "pub enum Error {\n"
                "    Empty,\n"
                "    Malformed(usize),\n"
                "}\n"
                "\n"
                "impl core::fmt::Display for Error {\n"
                "    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {\n"
                "        match self {\n"
                "            Error::Empty => write!(formatter, \"empty input\"),\n"
                "            Error::Malformed(field) => write!(formatter, \"malformed field {field}\"),\n"
                "        }\n"
                "    }\n"
                "}\n"
                "\n"
                "impl std::error::Error for Error {}\n"
                "\n"
                "pub fn tokenize(text: &str) -> Result<Vec<u32>, Error> {\n"
                "    if text.is_empty() {\n"
                "        return Err(Error::Empty);\n"
                "    }\n"
                "    let mut output = Vec::new();\n"
                "    for (index, field) in text.split_whitespace().enumerate() {\n"
                "        match field.parse::<u32>() {\n"
                "            Ok(value) => output.push(value),\n"
                "            Err(_) => return Err(Error::Malformed(index)),\n"
                "        }\n"
                "    }\n"
                "    Ok(output)\n"
                "}\n"),
            "src/lib.rs": (
                "//! Count fields. This does not build: `AppError` has no conversion from\n"
                "//! the version-2 dependency error, so `?` is rejected.\n"
                "\n"
                "#[derive(Debug, PartialEq)]\n"
                "pub enum AppError {\n"
                "    Empty,\n"
                "    Invalid { field: usize },\n"
                "}\n"
                "\n"
                "pub fn count_fields(text: &str) -> Result<usize, AppError> {\n"
                "    let tokens = tokens::tokenize(text)?;\n"
                "    Ok(tokens.len())\n"
                "}\n"),
        },
        "hidden": {
            "tests/check.rs": (
                "use migrate_03_error_conversion::{count_fields, AppError};\n"
                "\n"
                "#[test]\n"
                "fn counts_fields() {\n"
                "    assert_eq!(count_fields(\"1 2 3\"), Ok(3));\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn maps_the_empty_error() {\n"
                "    assert_eq!(count_fields(\"\"), Err(AppError::Empty));\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn maps_the_malformed_field() {\n"
                "    assert_eq!(count_fields(\"1 x\"), Err(AppError::Invalid { field: 1 }));\n"
                "}\n"),
        },
        "readonly": ["dep/"],
        "reference": {
            "src/lib.rs": (
                "//! Count fields, converting the dependency error into the application\n"
                "//! error.\n"
                "\n"
                "#[derive(Debug, PartialEq)]\n"
                "pub enum AppError {\n"
                "    Empty,\n"
                "    Invalid { field: usize },\n"
                "}\n"
                "\n"
                "impl From<tokens::Error> for AppError {\n"
                "    fn from(error: tokens::Error) -> Self {\n"
                "        match error {\n"
                "            tokens::Error::Empty => AppError::Empty,\n"
                "            tokens::Error::Malformed(field) => AppError::Invalid { field },\n"
                "        }\n"
                "    }\n"
                "}\n"
                "\n"
                "pub fn count_fields(text: &str) -> Result<usize, AppError> {\n"
                "    let tokens = tokens::tokenize(text)?;\n"
                "    Ok(tokens.len())\n"
                "}\n"),
        },
        "mutation": {"file": "src/lib.rs", "find": "AppError::Invalid { field }",
                     "replace": "AppError::Invalid { field: 0 }"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0277"]},
    },
    {
        "id": "migrate-04-trait-import",
        "category": "api-migration",
        "capability": "Bring an extension trait into scope after the method moved into it.",
        "prompt_intro": (
            "The crate does not build against the vendored `documents` 2.0.0. The\n"
            "`render` method moved from an inherent method into the `Render` trait.\n"
            "Do not modify `dep/`. Make `render_doc` compile by changing the\n"
            "application's imports; the output must remain `<h1>{title}</h1>`."),
        "observed_failure": (
            "error[E0599]: no method named `render` found for struct `documents::Doc`\n"
            "in the current scope\n"
            " --> src/lib.rs:2:31\n"
            "  |\n"
            "2 |     documents::doc(title).render()\n"
            "  |                           ^^^^^^ help: trait `Render` which provides `render`\n"
            "  |                                   is implemented, but is not in scope"),
        "manifest": _APP_MANIFEST.format(name="migrate_04_trait_import", dependency="documents"),
        "files": {
            "dep/Cargo.toml": _DOCUMENTS_MANIFEST,
            "dep/src/lib.rs": (
                "//! Documents, version 2: rendering is an extension trait.\n"
                "\n"
                "pub struct Doc {\n"
                "    pub title: String,\n"
                "}\n"
                "\n"
                "pub trait Render {\n"
                "    fn render(&self) -> String;\n"
                "}\n"
                "\n"
                "impl Render for Doc {\n"
                "    fn render(&self) -> String {\n"
                "        format!(\"<h1>{}</h1>\", self.title)\n"
                "    }\n"
                "}\n"
                "\n"
                "pub fn doc(title: &str) -> Doc {\n"
                "    Doc { title: title.to_string() }\n"
                "}\n"),
            "src/lib.rs": (
                "//! Render a document. This does not build: `render` moved into a trait\n"
                "//! that this module does not import.\n"
                "\n"
                "pub fn render_doc(title: &str) -> String {\n"
                "    documents::doc(title).render()\n"
                "}\n"),
        },
        "hidden": {
            "tests/check.rs": (
                "use migrate_04_trait_import::render_doc;\n"
                "\n"
                "#[test]\n"
                "fn renders_a_heading() {\n"
                "    assert_eq!(render_doc(\"t\"), \"<h1>t</h1>\");\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn renders_an_empty_title() {\n"
                "    assert_eq!(render_doc(\"\"), \"<h1></h1>\");\n"
                "}\n"),
        },
        "readonly": ["dep/"],
        "reference": {
            "src/lib.rs": (
                "//! Render a document via the extension trait.\n"
                "\n"
                "use documents::Render;\n"
                "\n"
                "pub fn render_doc(title: &str) -> String {\n"
                "    documents::doc(title).render()\n"
                "}\n"),
        },
        "mutation": {"file": "src/lib.rs", "find": ".render()",
                     "replace": ".render().replace('h', \"H\")"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0599"]},
    },
]
