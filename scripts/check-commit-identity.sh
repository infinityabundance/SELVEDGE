#!/usr/bin/env bash
# SELVEDGE commit-identity court.
#
# Fails closed unless:
#   1. the repository-local git user.name/user.email are the operator identity;
#   2. every scanned commit's author and committer are that identity;
#   3. no scanned commit carries a forbidden agent identity or AI attribution.
#
# Usage:
#   scripts/check-commit-identity.sh              # scan HEAD
#   scripts/check-commit-identity.sh <rev>        # scan one commit
#   scripts/check-commit-identity.sh --all        # scan every commit reachable from HEAD
#   scripts/check-commit-identity.sh --repo PATH  # explicit repository root
#
# Runs as the `commit_identity` court. It is self-contained: it does not depend
# on ~/.githooks or any global configuration, so it works identically inside a
# container where only the repository is mounted.
set -Eeuo pipefail
IFS=$'\n\t'

readonly EXPECTED_NAME="infinityabundance"
readonly EXPECTED_EMAIL="255699974+infinityabundance@users.noreply.github.com"

fail() {
  printf 'commit-identity: FAIL — %s\n' "$1" >&2
  exit 1
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo="${script_dir}/.."
mode="head"
rev=""

while (($#)); do
  case "$1" in
    --all)
      [[ "$mode" == "head" ]] || fail "--all cannot be combined with an explicit revision"
      mode="all"
      shift
      ;;
    --repo)
      (($# >= 2)) || fail "--repo needs a path"
      repo="$2"
      shift 2
      ;;
    -h | --help)
      sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    -*)
      fail "unknown option '$1'"
      ;;
    *)
      [[ "$mode" == "head" ]] || fail "only one revision may be supplied"
      mode="rev"
      rev="$1"
      shift
      ;;
  esac
done

repo=$(cd -- "$repo" && pwd -P) || fail "repository path does not exist"
git -C "$repo" rev-parse --git-dir >/dev/null 2>&1 || fail "$repo is not a git repository"

# ---------------------------------------------------------------------------
# 1. repository-local identity must be pinned to the operator
# ---------------------------------------------------------------------------
local_name=$(git -C "$repo" config --local --get user.name 2>/dev/null || true)
local_email=$(git -C "$repo" config --local --get user.email 2>/dev/null || true)

if [[ -z "$local_name" || -z "$local_email" ]]; then
  fail "repository-local user.name/user.email are not set (got '$local_name <$local_email>'); expected '$EXPECTED_NAME <$EXPECTED_EMAIL>'"
fi
if [[ "$local_name" != "$EXPECTED_NAME" ]]; then
  fail "repository-local user.name='$local_name'; expected '$EXPECTED_NAME'"
fi
if [[ "$local_email" != "$EXPECTED_EMAIL" ]]; then
  fail "repository-local user.email='$local_email'; expected '$EXPECTED_EMAIL'"
fi

# ---------------------------------------------------------------------------
# 2. collect the commits to scan
# ---------------------------------------------------------------------------
declare -a commits=()
case "$mode" in
  head)
    sha=$(git -C "$repo" rev-parse --verify --quiet HEAD) || fail "repository has no commits yet"
    commits=("$sha")
    ;;
  rev)
    sha=$(git -C "$repo" rev-parse --verify --quiet "${rev}^{commit}") || fail "'$rev' is not a commit"
    commits=("$sha")
    ;;
  all)
    mapfile -t commits < <(git -C "$repo" rev-list HEAD)
    ((${#commits[@]} > 0)) || fail "repository has no commits yet"
    ;;
esac

# ---------------------------------------------------------------------------
# 3. forbidden identities and AI attribution
# ---------------------------------------------------------------------------
forbidden_identities=(
  'agent@selvedge\.local'
  '@selvedge\.local'
  'anthropic\.com'
  'openai\.com'
  'noreply@zed\.dev'
  'noreply@github\.com'
  'cursor\.(sh|com)'
  'selvedge[ _-]?agent'
  '(^|[/< ])(claude|gpt|gemini|copilot|codex)([/> .]|$)'
)
forbidden_name_words=(
  'claude' 'anthropic' 'openai' 'chatgpt' 'copilot' 'gemini' 'codex' 'cursor'
  'selvedge agent' 'selvedge-agent'
  '^agent$' '^selvedge$' '^ai$'
)
attribution_re='co-authored-by:.*(claude|anthropic|openai|gpt|gemini|copilot|codex|zed)'
generated_re='(generated|written|powered|assisted) (with|by)[: ]+.*(claude|anthropic|chatgpt|openai|copilot|codex|gemini|zed|gpt)'

violations=0
report() {
  printf 'commit-identity: FAIL — commit %s: %s\n' "$1" "$2" >&2
  violations=$((violations + 1))
}

for commit in "${commits[@]}"; do
  short=${commit:0:12}
  mapfile -t fields < <(git -C "$repo" show -s --format='%an%n%ae%n%cn%n%ce%n%B' "$commit")
  author_name="${fields[0]:-}"
  author_email="${fields[1]:-}"
  committer_name="${fields[2]:-}"
  committer_email="${fields[3]:-}"
  message=$(printf '%s\n' "${fields[@]:4}")

  if [[ "$author_name <$author_email>" != "$EXPECTED_NAME <$EXPECTED_EMAIL>" ]]; then
    report "$short" "author '$author_name <$author_email>' is not the operator identity"
  fi
  if [[ "$committer_name <$committer_email>" != "$EXPECTED_NAME <$EXPECTED_EMAIL>" ]]; then
    report "$short" "committer '$committer_name <$committer_email>' is not the operator identity"
  fi

  ident_lower=$(printf '%s <%s>\n%s <%s>' "$author_name" "$author_email" "$committer_name" "$committer_email" | tr '[:upper:]' '[:lower:]')
  for pattern in "${forbidden_identities[@]}"; do
    if printf '%s' "$ident_lower" | grep -Eq "$pattern"; then
      report "$short" "identity matches forbidden pattern '$pattern'"
    fi
  done
  for word in "${forbidden_name_words[@]}"; do
    if printf '%s' "$ident_lower" | grep -Eq "$word"; then
      report "$short" "identity matches forbidden agent token '$word'"
    fi
  done

  message_lower=$(printf '%s' "$message" | tr '[:upper:]' '[:lower:]')
  if printf '%s' "$message_lower" | grep -Eq "$attribution_re"; then
    report "$short" "message carries an AI co-author line"
  fi
  if printf '%s' "$message_lower" | grep -Eq "$generated_re"; then
    report "$short" "message carries an AI attribution phrase"
  fi
  if printf '%s' "$message" | grep -q '🤖'; then
    report "$short" "message carries a robot emoji attribution marker"
  fi
done

if ((violations > 0)); then
  printf 'commit-identity: %d violation(s); refusing\n' "$violations" >&2
  exit 1
fi

printf 'commit-identity: OK — %d commit(s) scanned, identity %s <%s>\n' \
  "${#commits[@]}" "$EXPECTED_NAME" "$EXPECTED_EMAIL"
