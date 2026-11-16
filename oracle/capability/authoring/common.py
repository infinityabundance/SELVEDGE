"""Shared helpers for the expanded authoring batches (P14.4).

The first suite used explicit line concatenation so the materialised text could
be read beside the metadata. The expansion keeps the same rule -- the authoring
source is the origin -- but uses one de-indenting helper so a task stays one
readable block. `code()` is intentionally the only transformer: no text is
rewritten at materialisation time.
"""

from __future__ import annotations

import textwrap


def code(source: str) -> str:
    """Return file content with the declaration's own indentation removed."""
    return textwrap.dedent(source).lstrip("\n").rstrip() + "\n"
