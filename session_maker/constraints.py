import csv
from itertools import combinations
from pathlib import Path

_PARALLEL_SUFFIX = " cannot run in parallel sessions"
_SLOTS_MARKER = " cannot run in slots "

def _parse_int_csv(raw: str, label: str) -> list[int]:
    tokens = [token.strip() for token in raw.split(",")]
    if not tokens or any(not token for token in tokens):
        raise ValueError(f"Invalid {label} list: '{raw}'")
    if any(not token.isdigit() for token in tokens):
        raise ValueError(f"Invalid {label} list: '{raw}'")
    return [int(token) for token in tokens]

def _parse_bracketed_ids(raw: str) -> list[int]:
    text = raw.strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError(
            "Paper IDs must be bracketed, e.g. '[1,2,3]'"
        )
    inner = text[1:-1].strip()
    if not inner:
        raise ValueError("Paper IDs list cannot be empty")
    return _parse_int_csv(inner, "paper id")

def parse_constraints_csv(path: str | Path) -> dict[str, object]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise ValueError(f"Constraints file not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as file_handle:
        rows = list(csv.reader(file_handle))

    if not rows:
        return {
            "paper_not_parallel": [],
            "paper_forbidden_slots": {},
            "raw_constraints": [],
            "ignored_comment_lines": 0,
            "ignored_blank_lines": 0,
        }

    start_idx = 1 if len(rows[0]) == 1 and rows[0][0].strip().lower() == "constraint" else 0
    paper_not_parallel: set[tuple[int, int]] = set()
    paper_forbidden_slots: dict[int, set[int]] = {}
    raw_constraints: list[str] = []
    ignored_comment_lines = 0
    ignored_blank_lines = 0

    for line_idx, row in enumerate(rows[start_idx:], start=start_idx + 1):
        text = ",".join(part.strip() for part in row).strip()
        if not text or text.startswith("#"):
            if not text:
                ignored_blank_lines += 1
            else:
                ignored_comment_lines += 1
            continue

        compact = " ".join(text.split())
        lower_compact = compact.lower()
        raw_constraints.append(compact)

        if lower_compact.endswith(_PARALLEL_SUFFIX):
            left = compact[: -len(_PARALLEL_SUFFIX)].strip()
            paper_ids = sorted(set(_parse_bracketed_ids(left)))
            if len(paper_ids) < 2:
                raise ValueError(
                    f"Invalid parallel constraint on line {line_idx}: need at least 2 paper ids"
                )
            for paper_a, paper_b in combinations(paper_ids, 2):
                paper_not_parallel.add((paper_a, paper_b))
            continue

        marker_idx = lower_compact.find(_SLOTS_MARKER)
        if marker_idx != -1:
            left = compact[:marker_idx].strip()
            right = compact[marker_idx + len(_SLOTS_MARKER) :].strip()
            paper_ids = _parse_bracketed_ids(left)
            slots = set(_parse_int_csv(right, "slot"))
            for paper_id in paper_ids:
                paper_forbidden_slots.setdefault(paper_id, set()).update(slots)
            continue

        raise ValueError(
            f"Invalid constraint format on line {line_idx}: '{compact}'. "
            "Allowed forms: '[1,2,3] cannot run in parallel sessions' and "
            "'[190] cannot run in slots 1,2,3'"
        )

    return {
        "paper_not_parallel": sorted(paper_not_parallel),
        "paper_forbidden_slots": {
            paper_id: sorted(slot_ids)
            for paper_id, slot_ids in sorted(paper_forbidden_slots.items())
        },
        "raw_constraints": raw_constraints,
        "ignored_comment_lines": ignored_comment_lines,
        "ignored_blank_lines": ignored_blank_lines,
    }
