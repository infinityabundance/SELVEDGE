"""The capability court (P12.8-P12.11, P11.21, P12.9).

One instrument, three levels, and a fresh executable Rust suite:

* `sandbox` — bounded subprocess execution (wall clock, address space, CPU,
  file size, open files, no inherited fds, deterministic environment, a
  verified no-network check);
* `suite` — task loading, prompt rendering and content hashing;
* `extract` — code extraction from an adapter's answer;
* `checkers` — workdir construction and the executable checker;
* `stats` — pure-standard-library paired and stratified bootstrap;
* `adapters` — the uniform arm interface (HF checkpoint, Rust runtime, fake);
* `court` — the levels, the control matrix, the pre-registration and the
  anti-gaming checks.

Nothing here imports a third-party package, because the check stage runs in the
rust court, which carries `cargo` and a standard-library Python but no torch.
"""
