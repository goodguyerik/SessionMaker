import csv
import json
from pathlib import Path
from typing import Any

SLOT_HEADERS = ("slot", "session", "duration")
PAPER_HEADERS = ("paperid", "duration", "paper_title", "abstract", "author_emails")
ENRICHED_PAPER_HEADERS = (
    "paperid",
    "duration",
    "paper_title",
    "author_emails",
    "summary",
    "embedding",
)

def _validate_headers(csv_path: Path, headers: list[str], required: tuple[str, ...]) -> None:
    if any(header not in headers for header in required):
        raise ValueError(f"CSV '{csv_path}' must have headers: {', '.join(required)}")

def _read_rows(path: str | Path) -> tuple[Path, list[dict[str, str]], list[str]]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle, skipinitialspace=True)
        headers = [header.strip() for header in (reader.fieldnames or [])]
        rows = [dict(row) for row in reader]
    return csv_path, rows, headers

def _to_int(row: dict[str, Any], key: str) -> int:
    return int((row.get(key) or "").strip())

def parse_slots_csv(path: str | Path) -> list[dict[str, int]]:
    csv_path, raw_rows, headers = _read_rows(path)
    _validate_headers(csv_path, headers, SLOT_HEADERS)
    return [
        {
            "slot": _to_int(row, "slot"),
            "session": _to_int(row, "session"),
            "duration": _to_int(row, "duration"),
        }
        for row in raw_rows
    ]

def parse_papers_csv(path: str | Path) -> list[dict[str, int | str]]:
    csv_path, raw_rows, headers = _read_rows(path)
    _validate_headers(csv_path, headers, PAPER_HEADERS)
    rows: list[dict[str, int | str]] = []
    for row in raw_rows:
        rows.append(
            {
                "paperid": _to_int(row, "paperid"),
                "duration": _to_int(row, "duration"),
                "paper_title": (row.get("paper_title") or "").strip(),
                "abstract": (row.get("abstract") or "").strip(),
                "author_emails": (row.get("author_emails") or "").strip(),
            }
        )
    return rows

def parse_enriched_papers_csv(path: str | Path) -> list[dict[str, int | str | list[float]]]:
    csv_path, raw_rows, headers = _read_rows(path)
    _validate_headers(csv_path, headers, ENRICHED_PAPER_HEADERS)

    rows: list[dict[str, int | str | list[float]]] = []
    for idx, row in enumerate(raw_rows, start=2):
        embedding_raw = (row.get("embedding") or "").strip()
        try:
            embedding_data = json.loads(embedding_raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"CSV '{csv_path}' has invalid embedding JSON at line {idx}"
            ) from exc

        if not isinstance(embedding_data, list) or any(
            not isinstance(value, (int, float)) for value in embedding_data
        ):
            raise ValueError(
                f"CSV '{csv_path}' has non-numeric embedding at line {idx}"
            )

        rows.append(
            {
                "paperid": _to_int(row, "paperid"),
                "duration": _to_int(row, "duration"),
                "paper_title": (row.get("paper_title") or "").strip(),
                "author_emails": (row.get("author_emails") or "").strip(),
                "summary": (row.get("summary") or "").strip(),
                "embedding": [float(value) for value in embedding_data],
            }
        )
    return rows
