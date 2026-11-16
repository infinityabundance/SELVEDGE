#!/usr/bin/env python3
"""The fresh held-out task set for P8.1 — disjoint from every T2/T1 item.

The set is authored here, not fetched, so it can be committed with its checks:
no item text, answer or test comes from HumanEval, GSM8K, TruthfulQA or the
tokenised held-out windows. Disjointness is not asserted in prose: 
`oracle/eval/fresh_contamination.py` builds a receipt that hashes every item and
compares 8-word text shingles and 13-token shingles against every evaluation
source, and `--require-disjoint` fails closed on any hit.

Four domains, three checkable answer kinds:

* **rust** — compiler diagnostics, lifetime/borrow errors, thread-safety and
  failing tests. Answers are checked for the expected `rustc` error code(s) and
  required phrases. Compiling the snippets with `rustc` (available in the rust
  court) is a later strengthening; the receipt records that the current check is
  textual.
* **code** — a function body that must pass embedded unit tests, executed in the
  same sandbox the registered HumanEval harness uses.
* **math** — a numeric answer, compared after normalisation.
* **factual** — either a multiple-choice letter or a numeric answer, with the
  gold value fixed here.

Every item carries `id`, `domain`, `kind`, `prompt`, a checker payload and a
one-line `skill` label used only for reporting, never for selection.
"""

from __future__ import annotations

import hashlib
import re
import resource
import subprocess
import sys
import tempfile

#: The sandbox limits are the registered HumanEval harness's, so an item's
#: executable check cannot outgrow the harness that will score the arms.
ADDRESS_SPACE = 1_500_000_000
TIMEOUT = 3.0

TASKS: tuple[dict, ...] = (
    # ------------------------------------------------------------------ rust
    {
        "id": "rust-return-reference-to-local",
        "domain": "rust", "kind": "diagnostic", "skill": "lifetimes",
        "prompt": (
            "This function does not compile:\n\n"
            "    fn greeting() -> &str {\n"
            "        let text = String::from(\"hello\");\n"
            "        &text\n"
            "    }\n\n"
            "Name the rustc error code and say in one sentence what the compiler "
            "is preventing."),
        "expect": {"error_codes": ["E0515"],
                   "phrases": ["reference", "local"]},
    },
    {
        "id": "rust-immutable-borrow-then-push",
        "domain": "rust", "kind": "diagnostic", "skill": "borrowck",
        "prompt": (
            "This fails to compile:\n\n"
            "    let mut v = vec![1, 2, 3];\n"
            "    let first = &v[0];\n"
            "    v.push(4);\n"
            "    println!(\"{first}\");\n\n"
            "Name the rustc error code and state which borrow conflicts with "
            "which."),
        "expect": {"error_codes": ["E0502"],
                   "phrases": ["borrow", "mutable"]},
    },
    {
        "id": "rust-two-mutable-borrows-same-field",
        "domain": "rust", "kind": "diagnostic", "skill": "borrowck",
        "prompt": (
            "Given `let mut point = Point { x: 1, y: 2 };`, this fails:\n\n"
            "    let a = &mut point.x;\n"
            "    let b = &mut point.x;\n"
            "    *a += *b;\n\n"
            "Name the rustc error code. Then explain why taking `&mut point.y` "
            "instead of the second `&mut point.x` compiles."),
        "expect": {"error_codes": ["E0499"],
                   "phrases": ["field", "borrow"]},
    },
    {
        "id": "rust-borrowed-value-does-not-live-long-enough",
        "domain": "rust", "kind": "diagnostic", "skill": "lifetimes",
        "prompt": (
            "This fails to compile:\n\n"
            "    let r;\n"
            "    {\n"
            "        let x = 5;\n"
            "        r = &x;\n"
            "    }\n"
            "    println!(\"{r}\");\n\n"
            "Name the rustc error code and say what the borrow checker is "
            "rejecting."),
        "expect": {"error_codes": ["E0597"],
                   "phrases": ["borrow", "lifetime"]},
    },
    {
        "id": "rust-move-out-of-index",
        "domain": "rust", "kind": "diagnostic", "skill": "ownership",
        "prompt": (
            "This fails to compile:\n\n"
            "    let v = vec![String::from(\"a\")];\n"
            "    let s = v[0];\n\n"
            "Name the rustc error code and give the two-word reason indexing "
            "cannot move a value out."),
        "expect": {"error_codes": ["E0507"],
                   "phrases": ["move", "index"]},
    },
    {
        "id": "rust-non-exhaustive-match",
        "domain": "rust", "kind": "diagnostic", "skill": "patterns",
        "prompt": (
            "This fails to compile for `x: Option<i32>`:\n\n"
            "    match x {\n"
            "        Some(n) => println!(\"{n}\"),\n"
            "    }\n\n"
            "Name the rustc error code and state the smallest change that fixes "
            "it."),
        "expect": {"error_codes": ["E0004"],
                   "phrases": ["pattern", "none"]},
    },
    {
        "id": "rust-rc-across-threads",
        "domain": "rust", "kind": "diagnostic", "skill": "threads",
        "prompt": (
            "This fails to compile:\n\n"
            "    let shared = std::rc::Rc::new(7);\n"
            "    std::thread::spawn(move || println!(\"{shared}\"));\n\n"
            "Name the rustc error code and name the trait that `Rc` does not "
            "implement but the closure requires."),
        "expect": {"error_codes": ["E0277"],
                   "phrases": ["send", "rc"]},
    },
    {
        "id": "rust-mutex-guard-across-await",
        "domain": "rust", "kind": "diagnostic", "skill": "async",
        "prompt": (
            "This fails when the future is spawned on a multi-threaded executor:\n\n"
            "    async fn bump(m: &std::sync::Mutex<i32>) {\n"
            "        let mut guard = m.lock().unwrap();\n"
            "        other_async().await;\n"
            "        *guard += 1;\n"
            "    }\n\n"
            "Name the rustc error code and give the fix that keeps the mutex "
            "(two or three words)."),
        "expect": {"error_codes": ["E0277"],
                   "phrases": ["send", "guard"]},
    },
    {
        "id": "rust-trait-object-without-dyn",
        "domain": "rust", "kind": "diagnostic", "skill": "traits",
        "prompt": (
            "Rust 2021 edition. This fails:\n\n"
            "    trait Shape { fn area(&self) -> f64; }\n"
            "    fn total(items: Vec<Box<Shape>>) -> f64 { 0.0 }\n\n"
            "Name the rustc error code and the keyword that fixes the parameter "
            "type."),
        "expect": {"error_codes": ["E0782"],
                   "phrases": ["dyn"]},
    },
    {
        "id": "rust-debug-build-overflow",
        "domain": "rust", "kind": "diagnostic", "skill": "arithmetic",
        "prompt": (
            "In a debug build, this runs:\n\n"
            "    let x: u8 = 255;\n"
            "    let y = x + 1;\n\n"
            "State exactly what happens at runtime, and what happens instead in "
            "a release build."),
        "expect": {"error_codes": [],
                   "phrases": ["panic", "wrap"]},
    },
    # ------------------------------------------------------------------ code
    {
        "id": "code-dedupe-stable",
        "domain": "code", "kind": "code", "skill": "sequences",
        "prompt": (
            "def dedupe(values):\n"
            "    \"\"\"Return the values with duplicates removed, keeping the order\n"
            "    of first occurrence.\"\"\"\n"),
        "tests": (
            "assert dedupe([3, 1, 3, 2, 1]) == [3, 1, 2]\n"
            "assert dedupe([]) == []\n"
            "assert dedupe(['a', 'a']) == ['a']\n"),
        "entry": "dedupe",
    },
    {
        "id": "code-chunks",
        "domain": "code", "kind": "code", "skill": "sequences",
        "prompt": (
            "def chunks(values, size):\n"
            "    \"\"\"Split into consecutive lists of at most `size` items; the\n"
            "    last one may be shorter. Raise ValueError if size < 1.\"\"\"\n"),
        "tests": (
            "assert chunks([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]\n"
            "assert chunks([], 3) == []\n"
            "try:\n"
            "    chunks([1], 0)\n"
            "except ValueError:\n"
            "    pass\n"
            "else:\n"
            "    raise AssertionError('expected ValueError')\n"),
        "entry": "chunks",
    },
    {
        "id": "code-word-count",
        "domain": "code", "kind": "code", "skill": "text",
        "prompt": (
            "def word_count(text):\n"
            "    \"\"\"Count lower-cased whitespace-separated words, ignoring\n"
            "    punctuation attached to them.\"\"\"\n"),
        "tests": (
            "assert word_count('The cat, the hat.') == {'the': 2, 'cat': 1, 'hat': 1}\n"
            "assert word_count('') == {}\n"),
        "entry": "word_count",
    },
    {
        "id": "code-parse-ints",
        "domain": "code", "kind": "code", "skill": "parsing",
        "prompt": (
            "def parse_ints(text):\n"
            "    \"\"\"Return the integers in `text`, split on commas and spaces,\n"
            "    skipping any field that is not an integer.\"\"\"\n"),
        "tests": (
            "assert parse_ints('1, x, 3 -4') == [1, 3, -4]\n"
            "assert parse_ints('') == []\n"
            "assert parse_ints('7') == [7]\n"),
        "entry": "parse_ints",
    },
    {
        "id": "code-ring-buffer",
        "domain": "code", "kind": "code", "skill": "state",
        "prompt": (
            "class Ring:\n"
            "    \"\"\"A fixed-capacity integer buffer that overwrites the oldest\n"
            "    value. Ring(2) then push(1), push(2), push(3) holds [2, 3] in\n"
            "    insertion order, oldest first.\"\"\"\n"
            "    def __init__(self, capacity):\n"
            "        pass\n"
            "    def push(self, value):\n"
            "        pass\n"
            "    def values(self):\n"
            "        pass\n"),
        "tests": (
            "r = Ring(2)\n"
            "r.push(1); r.push(2); r.push(3)\n"
            "assert r.values() == [2, 3]\n"
            "r = Ring(1)\n"
            "r.push(9)\n"
            "assert r.values() == [9]\n"),
        "entry": "Ring",
    },
    {
        "id": "code-balanced-brackets",
        "domain": "code", "kind": "code", "skill": "parsing",
        "prompt": (
            "def balanced(text):\n"
            "    \"\"\"True iff (), [] and {} are correctly nested and closed.\n"
            "    Ignore every other character.\"\"\"\n"),
        "tests": (
            "assert balanced('([]{})') is True\n"
            "assert balanced('([)]') is False\n"
            "assert balanced('abc') is True\n"
            "assert balanced('(') is False\n"),
        "entry": "balanced",
    },
    # ------------------------------------------------------------------ math
    {
        "id": "math-product-plus",
        "domain": "math", "kind": "numeric", "skill": "arithmetic",
        "prompt": "Compute 17 * 23 + 11. Answer with the number only.",
        "gold": "402",
    },
    {
        "id": "math-gcd",
        "domain": "math", "kind": "numeric", "skill": "number theory",
        "prompt": "What is the greatest common divisor of 84 and 132? Answer with the number only.",
        "gold": "12",
    },
    {
        "id": "math-fraction-of",
        "domain": "math", "kind": "numeric", "skill": "arithmetic",
        "prompt": "What is three quarters of 96? Answer with the number only.",
        "gold": "72",
    },
    {
        "id": "math-sum-1-to-100",
        "domain": "math", "kind": "numeric", "skill": "series",
        "prompt": "What is the sum of the integers from 1 to 100 inclusive? Answer with the number only.",
        "gold": "5050",
    },
    {
        "id": "math-power-two",
        "domain": "math", "kind": "numeric", "skill": "arithmetic",
        "prompt": "What is 2 raised to the power 10? Answer with the number only.",
        "gold": "1024",
    },
    # --------------------------------------------------------------- factual
    {
        "id": "fact-anchor-once",
        "domain": "factual", "kind": "choice", "skill": "rust ownership",
        "prompt": (
            "Which statement about Rust ownership is correct?\n"
            "A) A value may have several owning bindings at once.\n"
            "B) A value has exactly one owner at a time, and the value is dropped "
            "when that owner goes out of scope.\n"
            "C) Ownership is decided by the garbage collector at run time.\n"
            "D) Only heap values have owners.\n"
            "Answer with the letter."),
        "gold": "B",
    },
    {
        "id": "fact-cargo-check",
        "domain": "factual", "kind": "choice", "skill": "tooling",
        "prompt": (
            "What does `cargo check` do that `cargo build` does not have to do?\n"
            "A) It compiles and links a binary.\n"
            "B) It type-checks and borrow-checks the crate without producing "
            "final codegen artefacts.\n"
            "C) It runs the test suite.\n"
            "D) It publishes the crate.\n"
            "Answer with the letter."),
        "gold": "B",
    },
    {
        "id": "fact-async-runtime",
        "domain": "factual", "kind": "choice", "skill": "async",
        "prompt": (
            "Which of these is an asynchronous runtime crate for Rust?\n"
            "A) serde\n"
            "B) tokio\n"
            "C) clap\n"
            "D) rand\n"
            "Answer with the letter."),
        "gold": "B",
    },
    {
        "id": "fact-option-reference-size",
        "domain": "factual", "kind": "numeric", "skill": "representation",
        "prompt": (
            "On a 64-bit target, how many bytes does `Option<&u8>` occupy? "
            "Answer with the number only."),
        "gold": "8",
    },
    {
        "id": "fact-f64-bits",
        "domain": "factual", "kind": "numeric", "skill": "representation",
        "prompt": (
            "How many explicit fraction (mantissa) bits does the IEEE-754 binary64 "
            "format have? Answer with the number only."),
        "gold": "52",
    },
)

DOMAINS = ("rust", "code", "math", "factual")
KINDS = ("diagnostic", "code", "numeric", "choice")

NUMBER = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
ERROR_CODE = re.compile(r"\bE\d{4}\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# text normalisation, hashing and shingles (shared with the contamination check)


def normalise(text: str) -> str:
    """Collapse whitespace and lower-case: the form both sides are compared in."""
    return re.sub(r"\s+", " ", text).strip().lower()


def sha256_text(text: str) -> str:
    return hashlib.sha256(normalise(text).encode("utf-8")).hexdigest()


def item_text(item: dict) -> str:
    """The item's model-independent text: prompt plus any embedded tests."""
    return item["prompt"] + "\n" + item.get("tests", "")


def word_shingles(text: str, width: int = 8) -> set[str]:
    words = re.findall(r"[a-z0-9_]+", normalise(text))
    if len(words) < width:
        return {hashlib.sha256(" ".join(words).encode()).hexdigest()} if words else set()
    return {hashlib.sha256(" ".join(words[start:start + width]).encode()).hexdigest()
            for start in range(len(words) - width + 1)}


def token_shingles(tokens: list[int], width: int = 13) -> set[str]:
    if len(tokens) < width:
        return {str(tokens)} if tokens else set()
    return {",".join(str(token) for token in tokens[start:start + width])
            for start in range(len(tokens) - width + 1)}


# ---------------------------------------------------------------------------
# checkers


def _run_python(program: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="selvedge-p8-") as directory:
        path = f"{directory}/candidate.py"
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(program)
        environment = {"PATH": "/usr/local/bin:/usr/bin:/bin", "PYTHONHASHSEED": "0",
                       "HOME": directory, "TMPDIR": directory}

        def limit():
            resource.setrlimit(resource.RLIMIT_AS, (ADDRESS_SPACE, ADDRESS_SPACE))

        try:
            completed = subprocess.run([sys.executable, "-I", path], cwd=directory,
                                       env=environment, capture_output=True, text=True,
                                       timeout=TIMEOUT, preexec_fn=limit)
        except subprocess.TimeoutExpired:
            return {"ok": False, "detail": "timeout"}
    return {"ok": completed.returncode == 0,
            "detail": (completed.stderr or "")[-600:]}


def _normalise_number(text: str) -> str | None:
    match = NUMBER.findall(text.replace(",", ""))
    if not match:
        return None
    value = match[-1]
    try:
        number = float(value)
    except ValueError:
        return value
    return str(int(number)) if number.is_integer() else f"{number:.6f}".rstrip("0").rstrip(".")


def check(item: dict, answer: str) -> dict:
    """Score one answer. `correct` is the only field the harness aggregates."""
    kind = item["kind"]
    if kind == "diagnostic":
        expect = item["expect"]
        codes = {code.upper() for code in ERROR_CODE.findall(answer)}
        wanted = {code.upper() for code in expect["error_codes"]}
        lowered = normalise(answer)
        phrases_ok = all(phrase in lowered for phrase in expect["phrases"])
        codes_ok = (wanted <= codes) if wanted else not codes
        return {"correct": bool(codes_ok and phrases_ok),
                "detail": {"codes": sorted(codes), "wanted": sorted(wanted),
                           "phrases_ok": phrases_ok}}
    if kind == "code":
        body = answer.strip("\n")
        program = item["prompt"] + body + "\n" + item["tests"] + \
            f"\nprint('p8-ok')\n"
        outcome = _run_python(program)
        return {"correct": bool(outcome["ok"]), "detail": outcome["detail"]}
    if kind == "numeric":
        got = _normalise_number(answer)
        gold = _normalise_number(item["gold"])
        return {"correct": got is not None and got == gold,
                "detail": {"got": got, "gold": gold}}
    if kind == "choice":
        letters = re.findall(r"\b([A-D])\b", answer.upper())
        got = letters[-1] if letters else None
        return {"correct": got == item["gold"], "detail": {"got": got,
                                                           "gold": item["gold"]}}
    raise ValueError(f"unknown item kind {kind!r}")


def by_id(item_id: str) -> dict:
    for item in TASKS:
        if item["id"] == item_id:
            return item
    raise KeyError(item_id)


def manifest() -> dict:
    """A stable description of the set, for the receipt."""
    counts = {domain: sum(1 for item in TASKS if item["domain"] == domain)
              for domain in DOMAINS}
    return {
        "schema": "selvedge/p8-fresh-tasks/v1",
        "items": len(TASKS),
        "by_domain": counts,
        "by_kind": {kind: sum(1 for item in TASKS if item["kind"] == kind)
                    for kind in KINDS},
        "sha256": hashlib.sha256("".join(
            sorted(sha256_text(item_text(item)) for item in TASKS)).encode()).hexdigest(),
        "ids": [item["id"] for item in TASKS],
    }
