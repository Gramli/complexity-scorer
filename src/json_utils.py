"""Shared JSON parsing. Dataset-specific validation stays in each caller."""

import json
from collections.abc import Generator
from pathlib import Path
from typing import Any


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject repeated property names instead of silently keeping the last value."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON property: {key}")
        result[key] = value
    return result


def iter_jsonl(path: Path) -> Generator[tuple[int, Any], None, None]:
    """Yield (line number, parsed value) for each nonblank line in a JSONL file.

    JSONL stores one JSON value per line. This reader handles the file format;
    callers decide which properties and values are required.
    """
    # Like C# using, 'with' closes the file when this block exits.
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue

            try:
                # The JSON parser calls this hook for every object, including
                # nested objects, before duplicate property names can be lost.
                record = json.loads(line, object_pairs_hook=reject_duplicate_keys)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error

            # Like C# yield return, this pauses until the next record is requested.
            # A caller using a limit can stop without parsing the remaining lines.
            yield line_number, record
