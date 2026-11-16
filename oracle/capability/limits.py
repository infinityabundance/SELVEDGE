#!/usr/bin/env python3
"""Hard limits for every capability-court subprocess.

AGENTS.md directive 2: no command runs without an explicit, workload-sized
memory limit, and a script must fail closed on a missing or malformed one. This
module is the single place a capability-court limit is declared, validated and
recorded.

Why the address-space default is 3 GiB while the measured peak RSS of a tiny
crate build is about 114 MiB:

* `RLIMIT_AS` bounds *virtual* address space. The pinned toolchain's linker
  (`rust-lld` via clang) reserves thread stacks and an arena; at 2 GiB it fails
  with `terminate called after throwing an instance of 'std::system_error'`
  (Resource temporarily unavailable) before it links a one-file crate, and at
  3 GiB it succeeds. That was measured in the rust court on 2026-11-16.
* RSS is what the container's `SELVEDGE_MEM` bound enforces. The per-checker
  limit is the smallest address space the pinned toolchain actually completes
  under, so a repaired crate cannot use the sandbox as an allocation amplifier
  without hitting a recorded failure.

Both numbers are recorded in every receipt, so a reader can see which bound
stopped a run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

#: `rustc` cannot link a one-file crate below this virtual size with the pinned
#: toolchain (measured: 2 GiB fails, 3 GiB passes).
TOOLCHAIN_MIN_ADDRESS_SPACE = 3 * 1024 * 1024 * 1024

#: Default per-checker limits. `wall_seconds` is 3x the largest reference
#: compile+test time measured during authoring; `cpu_seconds` is slightly below
#: it so a busy loop is killed by the CPU limit even if wall time is generous.
DEFAULT_TIMEOUT_SECONDS = 45.0
DEFAULT_CPU_SECONDS = 60
DEFAULT_ADDRESS_SPACE_BYTES = TOOLCHAIN_MIN_ADDRESS_SPACE
DEFAULT_FILE_BYTES = 64 * 1024 * 1024
DEFAULT_OPEN_FILES = 256
DEFAULT_OUTPUT_BYTES = 256 * 1024
DEFAULT_WORKDIR_BUDGET_BYTES = 2 * 1024 * 1024 * 1024


class LimitError(ValueError):
    """A limit is missing, malformed or unsafe."""


@dataclass(frozen=True)
class Limits:
    """One bounded-execution envelope."""

    wall_seconds: float
    cpu_seconds: int
    address_space_bytes: int
    file_bytes: int
    open_files: int
    output_bytes: int = DEFAULT_OUTPUT_BYTES

    def validate(self, where: str, toolchain_floor: bool = True) -> None:
        if not isinstance(self.wall_seconds, (int, float)) or self.wall_seconds <= 0:
            raise LimitError(f"{where}: wall_seconds must be a positive number")
        if not isinstance(self.cpu_seconds, int) or self.cpu_seconds <= 0:
            raise LimitError(f"{where}: cpu_seconds must be a positive integer")
        if self.cpu_seconds <= self.wall_seconds:
            # The wall clock is the primary bound. If the CPU limit fired first, a
            # busy loop would be recorded as a signal exit rather than a timeout,
            # and a deadlocked process (which burns no CPU) would never be stopped
            # by it either. CPU time is a backstop above the wall bound.
            raise LimitError(
                f"{where}: cpu_seconds {self.cpu_seconds} must exceed wall_seconds "
                f"{self.wall_seconds} so the wall clock is the primary bound")
        if not isinstance(self.address_space_bytes, int) or self.address_space_bytes <= 0:
            raise LimitError(f"{where}: address_space_bytes must be a positive integer")
        if toolchain_floor and self.address_space_bytes < TOOLCHAIN_MIN_ADDRESS_SPACE:
            # A checker that cannot link is not a measurement. The sandbox probes
            # that must trigger OOM run a *prebuilt* binary, so they do not need
            # to compile under the tighter limit.
            raise LimitError(
                f"{where}: address_space_bytes {self.address_space_bytes} is below the "
                f"measured toolchain floor {TOOLCHAIN_MIN_ADDRESS_SPACE}")
        if not isinstance(self.file_bytes, int) or self.file_bytes <= 0:
            raise LimitError(f"{where}: file_bytes must be a positive integer")
        if not isinstance(self.open_files, int) or self.open_files <= 0:
            raise LimitError(f"{where}: open_files must be a positive integer")
        if not isinstance(self.output_bytes, int) or self.output_bytes <= 0:
            raise LimitError(f"{where}: output_bytes must be a positive integer")

    def as_dict(self) -> dict:
        return asdict(self)


#: The envelope a *prebuilt* run-only probe uses. Address space is deliberately
#: below the toolchain floor because no compiler runs in it; it is the bound the
#: OOM probe must trip.
RUN_ONLY_LIMITS = Limits(
    wall_seconds=10.0,
    cpu_seconds=20,
    address_space_bytes=512 * 1024 * 1024,
    file_bytes=16 * 1024 * 1024,
    open_files=64,
)


def from_mapping(mapping: dict | None, where: str) -> Limits:
    """Build limits from a task's `limits` object, failing closed.

    A task may override a subset; anything it omits comes from the default
    envelope. An unknown key is an error, because a typo would silently leave a
    bound at its default.
    """
    values = {
        "wall_seconds": DEFAULT_TIMEOUT_SECONDS,
        "cpu_seconds": DEFAULT_CPU_SECONDS,
        "address_space_bytes": DEFAULT_ADDRESS_SPACE_BYTES,
        "file_bytes": DEFAULT_FILE_BYTES,
        "open_files": DEFAULT_OPEN_FILES,
        "output_bytes": DEFAULT_OUTPUT_BYTES,
    }
    if mapping is not None:
        if not isinstance(mapping, dict):
            raise LimitError(f"{where}: limits must be an object")
        unknown = set(mapping) - set(values) - {"kind"}
        if unknown:
            raise LimitError(f"{where}: unknown limit keys {sorted(unknown)}")
        for key in values:
            if key in mapping:
                values[key] = mapping[key]
    limits = Limits(**values)
    limits.validate(where)
    return limits


def run_only_from_mapping(mapping: dict | None, where: str) -> Limits:
    """Run-only limits, allowed below the toolchain floor."""
    base = asdict(RUN_ONLY_LIMITS)
    if mapping is not None:
        if not isinstance(mapping, dict):
            raise LimitError(f"{where}: limits must be an object")
        unknown = set(mapping) - set(base)
        if unknown:
            raise LimitError(f"{where}: unknown run-only limit keys {sorted(unknown)}")
        base.update(mapping)
    limits = Limits(**base)
    if not isinstance(limits.wall_seconds, (int, float)) or limits.wall_seconds <= 0:
        raise LimitError(f"{where}: wall_seconds must be positive")
    if not isinstance(limits.address_space_bytes, int) or limits.address_space_bytes <= 0:
        raise LimitError(f"{where}: address_space_bytes must be positive")
    if not isinstance(limits.cpu_seconds, int) or limits.cpu_seconds <= 0:
        raise LimitError(f"{where}: cpu_seconds must be positive")
    if limits.cpu_seconds <= limits.wall_seconds:
        raise LimitError(
            f"{where}: cpu_seconds must exceed wall_seconds so the wall clock is the "
            f"primary bound")
    return limits
