#!/bin/sh
# Emits tab-separated toolchain facts from inside the court container.
# Consumed by scripts/courts/write-receipt.py. Never fails the court run; a
# missing tool is reported as "absent" rather than omitted silently.
set -u

emit() {
  printf '%s\t%s\n' "$1" "$2"
}

first_line() {
  "$@" 2>&1 | head -1
}

if command -v rustc >/dev/null 2>&1; then
  emit rustc "$(first_line rustc --version)"
else
  emit rustc absent
fi

if command -v cargo >/dev/null 2>&1; then
  emit cargo "$(first_line cargo --version)"
else
  emit cargo absent
fi

if command -v python3 >/dev/null 2>&1; then
  emit python3 "$(first_line python3 --version)"
else
  emit python3 absent
fi

if command -v pip3 >/dev/null 2>&1; then
  emit pip3 "$(first_line pip3 --version)"
elif command -v pip >/dev/null 2>&1; then
  emit pip3 "$(first_line pip --version)"
else
  emit pip3 absent
fi

if command -v cc >/dev/null 2>&1; then
  emit cc "$(first_line cc --version)"
else
  emit cc absent
fi

if command -v git >/dev/null 2>&1; then
  emit git "$(first_line git --version)"
else
  emit git absent
fi

emit nproc "$(nproc 2>/dev/null || echo unknown)"
emit kernel_release "$(uname -r 2>/dev/null || echo unknown)"
emit machine "$(uname -m 2>/dev/null || echo unknown)"
