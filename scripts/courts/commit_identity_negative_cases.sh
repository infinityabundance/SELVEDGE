#!/usr/bin/env bash
# Negative-case self-test for scripts/check-commit-identity.sh.
#
# Builds throwaway repositories under /tmp inside the court and asserts the
# identity court fails closed on: a wrong repository-local identity, a commit
# authored by an agent identity, and an AI co-author attribution line. The real
# repository is only read, never mutated.
#
# Run through the Python court:
#   scripts/docker-court.sh python -- bash scripts/courts/commit_identity_negative_cases.sh
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
checker="$script_dir/../check-commit-identity.sh"

operator_name="infinityabundance"
operator_email="255699974+infinityabundance@users.noreply.github.com"

work=$(mktemp -d /tmp/selvedge-id-negative.XXXXXX)
trap 'rm -rf "$work"' EXIT

failures=0
case_count=0

# The court container has no global git config and a private HOME, so every
# throwaway repository must pin its own identity.
new_repo() {
  local path="$1" name="$2" email="$3"
  mkdir -p "$path"
  git -C "$path" init -q
  git -C "$path" config user.name "$name"
  git -C "$path" config user.email "$email"
  printf 'x\n' >"$path/file.txt"
  git -C "$path" add file.txt
}

expect_failure() {
  local label="$1" repo="$2" needle="$3"
  local out rc
  case_count=$((case_count + 1))
  set +e
  out=$("$checker" --repo "$repo" 2>&1)
  rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then
    printf 'FAIL %-28s expected a non-zero exit, got 0\n' "$label"
    failures=$((failures + 1))
    return
  fi
  if [[ "$out" != *"$needle"* ]]; then
    printf 'FAIL %-28s missing diagnostic %q\n     output: %s\n' "$label" "$needle" "$out"
    failures=$((failures + 1))
    return
  fi
  printf 'PASS %-28s %s\n' "$label" "$needle"
}

# 1. correct identity is accepted (baseline)
baseline="$work/baseline"
new_repo "$baseline" "$operator_name" "$operator_email"
git -C "$baseline" commit -q -m "baseline commit"
case_count=$((case_count + 1))
if out=$("$checker" --repo "$baseline" 2>&1); then
  printf 'PASS %-28s correct operator identity accepted\n' "baseline"
else
  printf 'FAIL %-28s correct operator identity rejected: %s\n' "baseline" "$out"
  failures=$((failures + 1))
fi

# 2. repository-local identity is an agent identity
wrong="$work/wrong-local"
new_repo "$wrong" "SELVEDGE Agent" "agent@selvedge.local"
git -C "$wrong" commit -q -m "agent identity"
expect_failure "wrong-local-identity" "$wrong" "user.name='SELVEDGE Agent'"

# 3. local identity correct, but the commit was authored by an agent identity
authored="$work/agent-authored"
new_repo "$authored" "$operator_name" "$operator_email"
git -C "$authored" -c user.name="Claude" -c user.email="noreply@anthropic.com" commit -q -m "authored by an agent"
expect_failure "agent-authored-commit" "$authored" "is not the operator identity"

# 4. operator identity but an AI co-author attribution line
attributed="$work/ai-attribution"
new_repo "$attributed" "$operator_name" "$operator_email"
git -C "$attributed" commit -q -m "Add feature" -m "Co-authored-by: Claude <noreply@anthropic.com>"
expect_failure "ai-attribution" "$attributed" "AI co-author"

# 5. operator identity but a generated-with attribution phrase
generated="$work/ai-generated"
new_repo "$generated" "$operator_name" "$operator_email"
git -C "$generated" commit -q -m "Generated with ChatGPT"
expect_failure "ai-generated-phrase" "$generated" "AI attribution phrase"

if ((failures > 0)); then
  printf 'commit-identity negative cases: %d of %d failed\n' "$failures" "$case_count" >&2
  exit 1
fi
printf 'commit-identity negative cases: %d cases behaved as required\n' "$case_count"
