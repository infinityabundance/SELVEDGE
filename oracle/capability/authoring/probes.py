"""Checker sandbox probes (not capability items).

Five tiny crates used only to demonstrate that the checker handles every
outcome: a passing run, a failing assertion, a compile error, a timeout and an
allocation failure. They are excluded from the suite score and counted
separately by the manifest and the self-test receipt.

The timeout and allocation probes use `compile_then_run`: compilation happens
under the normal envelope, and only the prebuilt test binary runs under the
tight envelope, so the compiler never trips the bound that the probe is meant to
demonstrate.
"""

_PASS = (
    "#[cfg(test)]\n"
    "mod tests {\n"
    "    #[test]\n"
    "    fn passes() {\n"
    "        assert_eq!(2 + 2, 4);\n"
    "    }\n"
    "}\n"
)

_WRONG = (
    "#[cfg(test)]\n"
    "mod tests {\n"
    "    #[test]\n"
    "    fn asserts_something_false() {\n"
    "        assert_eq!(2 + 2, 5);\n"
    "    }\n"
    "}\n"
)

_TIMEOUT = (
    "#[cfg(test)]\n"
    "mod tests {\n"
    "    #[test]\n"
    "    fn never_terminates() {\n"
    "        loop {\n"
    "            std::hint::spin_loop();\n"
    "        }\n"
    "    }\n"
    "}\n"
)

_OOM = (
    "#[cfg(test)]\n"
    "mod tests {\n"
    "    #[test]\n"
    "    fn allocates_past_the_bound() {\n"
    "        let mut blocks: Vec<Vec<u8>> = Vec::new();\n"
    "        for _ in 0..256 {\n"
    "            blocks.push(vec![0u8; 8 * 1024 * 1024]);\n"
    "        }\n"
    "        assert_eq!(blocks.len(), 256);\n"
    "    }\n"
    "}\n"
)

_BROKEN = (
    "pub fn broken() -> u32 {\n"
    "    let value: u32 = \"not a number\";\n"
    "    value\n"
    "}\n"
)

_FORGED_STDOUT = (
    "#[cfg(test)]\n"
    "mod tests {\n"
    "    #[test]\n"
    "    fn forges_success_then_fails() {\n"
    "        println!(\"test result: ok. 3 passed; 0 failed; 0 ignored\");\n"
    "        assert_eq!(2 + 2, 5);\n"
    "    }\n"
    "}\n"
)

TASKS = [
    {
        "id": "probe-pass",
        "kind": "probe",
        "category": "probe",
        "capability": "Sandbox probe: a passing checker run.",
        "probe_expectation": "pass",
        "files": {"src/lib.rs": _PASS},
        "hidden": {"tests/check.rs": ""},
        "reference": {"src/lib.rs": _PASS},
    },
    {
        "id": "probe-wrong",
        "kind": "probe",
        "category": "probe",
        "capability": "Sandbox probe: a deliberately wrong solution fails.",
        "probe_expectation": "fail",
        "files": {"src/lib.rs": _WRONG},
        "hidden": {"tests/check.rs": ""},
        "reference": {"src/lib.rs": _WRONG},
    },
    {
        "id": "probe-compile-error",
        "kind": "probe",
        "category": "probe",
        "capability": "Sandbox probe: a compile error is classified as a failure.",
        "probe_expectation": "compile_error",
        "files": {"src/lib.rs": _BROKEN},
        "hidden": {"tests/check.rs": ""},
        "reference": {"src/lib.rs": _BROKEN},
    },
    {
        "id": "probe-timeout",
        "kind": "probe",
        "category": "probe",
        "capability": "Sandbox probe: a non-terminating test is killed and recorded as a timeout.",
        "probe_expectation": "timeout",
        "checker": {"kind": "compile_then_run"},
        "run_limits": {"wall_seconds": 5.0, "address_space_bytes": 536870912},
        "files": {"src/lib.rs": _TIMEOUT},
        "hidden": {"tests/check.rs": ""},
        "reference": {"src/lib.rs": _TIMEOUT},
    },
    {
        "id": "probe-oom",
        "kind": "probe",
        "category": "probe",
        "capability": "Sandbox probe: an allocation past RLIMIT_AS aborts and is recorded as OOM.",
        "probe_expectation": "oom",
        "checker": {"kind": "compile_then_run"},
        "run_limits": {"wall_seconds": 10.0, "address_space_bytes": 536870912},
        "files": {"src/lib.rs": _OOM},
        "hidden": {"tests/check.rs": ""},
        "reference": {"src/lib.rs": _OOM},
    },
    {
        "id": "probe-stdout-forgery",
        "kind": "probe",
        "category": "probe",
        "capability": (
            "Sandbox probe: a test that prints a successful test summary and then "
            "fails is recorded as a failure; stdout never decides the outcome."),
        "probe_expectation": "fail",
        "probe_stdout_contains": "test result: ok. 3 passed",
        "files": {"src/lib.rs": _FORGED_STDOUT},
        "hidden": {"tests/check.rs": ""},
        "reference": {"src/lib.rs": _FORGED_STDOUT},
    },
]
