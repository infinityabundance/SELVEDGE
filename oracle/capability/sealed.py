#!/usr/bin/env python3
"""Custody and scoring gate for the sealed holdback (P14.7).

The sealed set is authored by the same process as the scored suite and frozen
when its registration is written: the registration names every item hash, the
sealed aggregate and the freeze date. The scoring path is deliberately awkward:

* the harness refuses to score the sealed set unless `--unlock-sealed` is
  supplied **and** the supplied SHA-256 equals the registered sealed aggregate;
* an accepted unlock is written to an unlock record naming the flag, the hash
  and the UTC time, before any item is checked;
* the registration is append-only: re-freezing requires a new file and a stated
  reason, exactly like the pre-registration.

`authorize` is pure and therefore testable without cargo; the scoring script is
`scripts/capability_sealed.py`.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os

from . import suite as suite_module

SCHEMA_REGISTRATION = "selvedge/capability-sealed-registration/v1"
SCHEMA_UNLOCK = "selvedge/capability-sealed-unlock/v1"
REGISTRATION_NAME = "SEALED.json"
UNLOCK_FLAG = "--unlock-sealed"

#: The rule the registration records and the harness enforces.
RULE = (
    "The sealed set is frozen at the registered UTC time and sealed hash. It may "
    "not be used for tuning, prompt selection, few-shot examples, checkpoint "
    "selection, threshold selection, or any iterative development. It is opened "
    "once, for a declared external comparison, with the unlock flag and the "
    "registered hash, and its result is published once per revision. Re-running "
    "it after seeing a result is a new revision and must be reported as such."
)

#: The exact procedure. Recorded in the registration so it cannot drift from the
#: implementation without the hash changing.
PROCEDURE = (
    "1. copy the sealed tree and verify every item hash against this registration "
    "(`scripts/capability_sealed.py --verify`); "
    "2. run the scoring command exactly once with `--score --unlock-sealed "
    "--sealed-sha256 {sealed_sha256}`; "
    "3. the harness writes an unlock record with the UTC time before checking any "
    "item, then refuses to run again for the same revision; "
    "4. report the sealed result once, with the unlock record and the revision."
)


class SealedError(RuntimeError):
    """The sealed set may not be scored or its custody is broken."""


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def registration_path(suite_root: str) -> str:
    return os.path.join(suite_root, REGISTRATION_NAME)


def sealed_root(suite_root: str) -> str:
    return os.path.join(suite_root, "sealed")


def load_items(suite_root: str) -> list:
    root = sealed_root(suite_root)
    if not os.path.isdir(root):
        raise SealedError(f"{root}: the sealed collection is missing")
    return suite_module.load_collection(root)


def item_hashes(items: list) -> dict:
    return {task.id: task.tree_sha256 for task in items}


def aggregate(hashes: dict) -> str:
    digest = hashlib.sha256()
    for task_id in sorted(hashes):
        digest.update(task_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashes[task_id].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def summarise(items: list) -> dict:
    by_category: dict[str, int] = {}
    for task in items:
        by_category[task.category] = by_category.get(task.category, 0) + 1
    return {
        "total": len(items),
        "by_category": dict(sorted(by_category.items())),
        "by_difficulty": _count([task.metadata.get("difficulty", "medium")
                                 for task in items]),
    }


def _count(values: list) -> dict:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def build_registration(suite_root: str, freeze_utc: str | None = None,
                       suite_sha256: str | None = None) -> dict:
    """Build the registration payload for the sealed tree as it stands."""
    items = load_items(suite_root)
    hashes = item_hashes(items)
    sealed_sha256 = aggregate(hashes)
    frozen = freeze_utc or utc_now()
    return {
        "schema": SCHEMA_REGISTRATION,
        "frozen_utc": frozen,
        "freeze_date": frozen[:10],
        "rule": RULE,
        "unlock_procedure": PROCEDURE.format(sealed_sha256=sealed_sha256),
        "unlock": {
            "flag": UNLOCK_FLAG,
            "sealed_sha256": sealed_sha256,
            "one_shot": True,
            "may_not_be_used_for_tuning": True,
        },
        "counts": summarise(items),
        "suite_sha256": suite_sha256,
        "sealed_sha256": sealed_sha256,
        "items": [
            {"id": task.id, "category": task.category,
             "tree_sha256": hashes[task.id]}
            for task in sorted(items, key=lambda task: task.id)
        ],
    }


def write_registration(suite_root: str, payload: dict, refuse_overwrite: bool = True) -> str:
    path = registration_path(suite_root)
    if os.path.exists(path) and refuse_overwrite:
        raise SealedError(
            f"{path} already exists; the sealed registration is append-only. A new "
            f"sealed set requires a new file and a recorded reason")
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)
    return path


def load_registration(suite_root: str) -> dict:
    path = registration_path(suite_root)
    if not os.path.isfile(path):
        raise SealedError(f"{path}: the sealed registration is missing")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") != SCHEMA_REGISTRATION:
        raise SealedError(f"{path}: schema must be {SCHEMA_REGISTRATION!r}")
    for key in ("frozen_utc", "rule", "unlock", "sealed_sha256", "items"):
        if key not in payload:
            raise SealedError(f"{path}: missing required key {key!r}")
    return payload


def check_registration(suite_root: str) -> tuple[bool, str]:
    """Recompute every item hash and the aggregate; fail closed on drift."""
    try:
        registration = load_registration(suite_root)
        items = load_items(suite_root)
    except (SealedError, suite_module.SuiteError) as exc:
        return False, str(exc)
    recorded = {entry["id"]: entry["tree_sha256"] for entry in registration["items"]}
    current = item_hashes(items)
    missing = sorted(set(recorded) - set(current))
    extra = sorted(set(current) - set(recorded))
    if missing:
        return False, f"sealed items recorded but absent: {missing}"
    if extra:
        return False, f"sealed items present but not recorded: {extra}"
    changed = sorted(task_id for task_id in current if current[task_id] != recorded[task_id])
    if changed:
        return False, f"sealed items differ from the freeze: {changed}"
    if aggregate(current) != registration["sealed_sha256"]:
        return False, "the sealed aggregate differs from the registered hash"
    if registration.get("counts") != summarise(items):
        return False, "the sealed counts differ from the registered counts"
    return True, "the sealed tree matches the registration"


def authorize(registration: dict, unlock_flag: bool, supplied_sha256: str | None,
              now: str | None = None) -> dict:
    """The gate. Refuses unless the flag and the registered hash are both given.

    Returns the unlock record that must be written *before* any item is checked.
    """
    if not unlock_flag:
        raise SealedError(
            f"the sealed set is frozen; scoring requires {UNLOCK_FLAG}. It may not "
            f"be used for tuning, prompt selection or iterative development")
    if not supplied_sha256:
        raise SealedError("the sealed set requires the registered --sealed-sha256")
    if supplied_sha256.strip().lower() != registration["sealed_sha256"]:
        raise SealedError(
            "the supplied sealed hash does not match the registration; refusing to "
            "score the sealed set")
    unlocked = now or utc_now()
    return {
        "schema": SCHEMA_UNLOCK,
        "unlocked": True,
        "unlocked_utc": unlocked,
        "flag": UNLOCK_FLAG,
        "sealed_sha256": registration["sealed_sha256"],
        "frozen_utc": registration["frozen_utc"],
        "one_shot": True,
        "policy": RULE,
    }


def write_unlock_record(path: str, record: dict) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)
    return path
