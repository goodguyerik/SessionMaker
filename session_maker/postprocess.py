import csv
from collections import defaultdict
from io import StringIO
from itertools import combinations
from typing import Any 
from openai import OpenAI
from session_maker.prompts import render_session_assignment_prompt


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _normalize_authors(author_emails: str) -> set[str]:
    return {
        token.strip().lower()
        for token in author_emails.split(";")
        if token.strip()
    }


def _author_parallel_pairs(
    papers: list[dict[str, Any]],
    paper_to_cluster: dict[int, str],
) -> set[tuple[str, str]]:
    author_clusters: dict[str, set[str]] = defaultdict(set)
    for paper in papers:
        cluster_id = paper_to_cluster.get(int(paper["paperid"]))
        if not cluster_id:
            continue
        for author in _normalize_authors(str(paper.get("author_emails", ""))):
            author_clusters[author].add(cluster_id)

    pairs: set[tuple[str, str]] = set()
    for clusters in author_clusters.values():
        for cluster_a, cluster_b in combinations(sorted(clusters), 2):
            pairs.add(_pair_key(cluster_a, cluster_b))
    return pairs


def _map_paper_constraints(
    constraints: dict[str, Any],
    paper_to_cluster: dict[int, str],
) -> tuple[set[tuple[str, str]], dict[str, set[int]]]:
    cluster_not_parallel: set[tuple[str, str]] = set()
    cluster_forbidden_slots: dict[str, set[int]] = defaultdict(set)

    for paper_a, paper_b in constraints.get("paper_not_parallel", []):
        cluster_a = paper_to_cluster.get(int(paper_a))
        cluster_b = paper_to_cluster.get(int(paper_b))
        if not cluster_a:
            raise ValueError(f"Constraint conflict: paper {paper_a} has no assigned cluster")
        if not cluster_b:
            raise ValueError(f"Constraint conflict: paper {paper_b} has no assigned cluster")
        if cluster_a != cluster_b:
            cluster_not_parallel.add(_pair_key(cluster_a, cluster_b))

    for paper_id, forbidden_slots in constraints.get("paper_forbidden_slots", {}).items():
        cluster_id = paper_to_cluster.get(int(paper_id))
        if not cluster_id:
            raise ValueError(f"Constraint conflict: paper {paper_id} has no assigned cluster")
        cluster_forbidden_slots[cluster_id].update(int(slot) for slot in forbidden_slots)

    return cluster_not_parallel, cluster_forbidden_slots


def _llm_text(client: OpenAI, model: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": prompt}],
        temperature=0,
    )
    content = response.choices[0].message.content
    return content.strip() if content else ""


def _build_sessions_csv(
    slot_rows: list[dict[str, Any]],
    paper_assignments: list[dict[str, Any]],
) -> str:
    paper_counts: dict[str, int] = defaultdict(int)
    for row in paper_assignments:
        paper_counts[str(row["assigned_cluster_id"])] += 1

    lines = ["cluster,session_name,paper_count,duration"]
    for row in sorted(slot_rows, key=lambda item: str(item["cluster_id"])):
        cluster_id = str(row["cluster_id"])
        session_name = str(row.get("topic", "")).strip() or cluster_id
        session_name = session_name.replace('"', "''")
        lines.append(
            f'{cluster_id},"{session_name}",{paper_counts.get(cluster_id, 0)},{int(row["slot_duration"])}'
        )
    return "\n".join(lines)


def _build_slots_csv(slots: list[dict[str, int]]) -> str:
    lines = ["slot,track,duration"]
    for row in sorted(slots, key=lambda item: (int(item["slot"]), int(item["session"]))):
        lines.append(f"{int(row['slot'])},{int(row['session'])},{int(row['duration'])}")
    return "\n".join(lines)


def _build_constraints_text(
    not_parallel: set[tuple[str, str]],
    forbidden_slots: dict[str, set[int]],
    retry_error: str,
) -> str:
    lines: list[str] = []
    if not not_parallel and not forbidden_slots:
        lines.append("- none")
    else:
        for cluster_a, cluster_b in sorted(not_parallel):
            lines.append(
                f"- {cluster_a}, {cluster_b} cannot run in parallel (same slot number)"
            )
        for cluster_id in sorted(forbidden_slots):
            slots_text = ",".join(str(slot_num) for slot_num in sorted(forbidden_slots[cluster_id]))
            lines.append(f"- {cluster_id} cannot run in slots {slots_text}")

    if retry_error:
        lines.append("")
        lines.append("Previous attempt violated hard constraints:")
        lines.append(f"- {retry_error}")

    return "\n".join(lines)


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def _parse_assignment_csv(text: str) -> list[dict[str, Any]]:
    cleaned = _strip_fences(text)
    lines = cleaned.splitlines()
    header_idx = -1
    for idx, raw_line in enumerate(lines):
        compact = "".join(raw_line.lower().split())
        if compact == "slot,track,cluster,session_name,duration":
            header_idx = idx
            break

    if header_idx == -1:
        raise ValueError("LLM output did not contain expected assignment CSV header")

    extracted_lines = [lines[header_idx]]
    started = False
    for raw_line in lines[header_idx + 1 :]:
        line = raw_line.strip()
        if not line:
            if started:
                break
            continue
        try:
            values = next(csv.reader([line], skipinitialspace=True))
        except Exception:
            if started:
                break
            continue
        if len(values) < 5:
            if started:
                break
            continue
        first = values[0].strip()
        if not first.isdigit():
            if started:
                break
            continue
        extracted_lines.append(raw_line)
        started = True

    reader = csv.DictReader(StringIO("\n".join(extracted_lines)), skipinitialspace=True)
    rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("LLM output did not contain a valid assignment CSV table")

    parsed: list[dict[str, Any]] = []
    for row in rows:
        normalized = {
            str(key).strip().lower(): ("" if value is None else str(value).strip())
            for key, value in row.items()
            if key is not None
        }
        slot_raw = normalized.get("slot", "")
        track_raw = normalized.get("track", normalized.get("session", ""))
        cluster_raw = normalized.get("cluster", normalized.get("cluster_id", ""))
        session_name = normalized.get("session_name", "")
        duration_raw = normalized.get("duration", "")

        if not slot_raw or not track_raw or not cluster_raw or not duration_raw:
            raise ValueError("LLM CSV must contain slot, track, cluster, session_name, duration")

        try:
            slot_num = int(slot_raw)
            track_num = int(track_raw)
            duration_num = int(duration_raw)
        except ValueError as exc:
            raise ValueError(
                f"LLM CSV contains non-integer slot/track/duration: {row}"
            ) from exc

        parsed.append(
            {
                "slot": slot_num,
                "session": track_num,
                "cluster_id": cluster_raw,
                "session_name": session_name,
                "duration": duration_num,
            }
        )
    return parsed


def _validate_assignments(
    assignments: list[dict[str, Any]],
    slot_rows: list[dict[str, Any]],
    slots: list[dict[str, int]],
    not_parallel: set[tuple[str, str]],
    forbidden_slots: dict[str, set[int]],
) -> dict[str, tuple[int, int]]:
    clusters = {str(row["cluster_id"]): int(row["slot_duration"]) for row in slot_rows}
    expected_clusters = set(clusters)
    slot_duration_map = {
        (int(row["slot"]), int(row["session"])): int(row["duration"])
        for row in slots
    }

    seen_clusters: set[str] = set()
    used_positions: set[tuple[int, int]] = set()
    cluster_positions: dict[str, tuple[int, int]] = {}

    if len(assignments) != len(expected_clusters):
        raise ValueError(
            f"Expected {len(expected_clusters)} assigned rows but got {len(assignments)}"
        )

    for assignment in assignments:
        cluster_id = str(assignment["cluster_id"])
        slot_num = int(assignment["slot"])
        session_num = int(assignment["session"])
        duration = int(assignment["duration"])

        if cluster_id not in expected_clusters:
            raise ValueError(f"Unknown cluster in LLM output: {cluster_id}")
        if cluster_id in seen_clusters:
            raise ValueError(f"Duplicate cluster assignment: {cluster_id}")

        position = (slot_num, session_num)
        if position not in slot_duration_map:
            raise ValueError(
                f"Assigned position slot={slot_num}, track={session_num} does not exist"
            )
        if position in used_positions:
            raise ValueError(
                f"Duplicate slot assignment: slot={slot_num}, track={session_num}"
            )

        slot_duration = slot_duration_map[position]
        cluster_duration = clusters[cluster_id]
        if duration != slot_duration:
            raise ValueError(
                f"Duration mismatch for slot={slot_num}, track={session_num}: "
                f"slot duration is {slot_duration}, assignment says {duration}"
            )
        if duration != cluster_duration:
            raise ValueError(
                f"Duration mismatch for {cluster_id}: cluster needs {cluster_duration}, got {duration}"
            )

        blocked_slots = forbidden_slots.get(cluster_id, set())
        if slot_num in blocked_slots:
            blocked = ",".join(str(item) for item in sorted(blocked_slots))
            raise ValueError(
                f"Forbidden slot violation for {cluster_id}: slot {slot_num} blocked by {blocked}"
            )

        seen_clusters.add(cluster_id)
        used_positions.add(position)
        cluster_positions[cluster_id] = position

    missing = sorted(expected_clusters - seen_clusters)
    if missing:
        raise ValueError(f"Missing assignments for clusters: {', '.join(missing)}")

    for cluster_a, cluster_b in sorted(not_parallel):
        slot_a = cluster_positions.get(cluster_a, (None, None))[0]
        slot_b = cluster_positions.get(cluster_b, (None, None))[0]
        if slot_a is not None and slot_a == slot_b:
            raise ValueError(
                f"Parallel conflict: {cluster_a} and {cluster_b} both assigned to slot {slot_a}"
            )

    return cluster_positions


def reassign_clusters(
    slot_rows: list[dict[str, Any]],
    paper_assignments: list[dict[str, Any]],
    papers: list[dict[str, Any]],
    slots: list[dict[str, int]],
    client: OpenAI,
    general_model: str,
    main_field: str,
    parsed_constraints: dict[str, Any] | None = None,
    max_assignment_attempts: int = 8,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    if not slot_rows:
        return slot_rows

    constraints = parsed_constraints or {
        "paper_not_parallel": [],
        "paper_forbidden_slots": {},
    }

    paper_to_cluster = {
        int(row["paperid"]): str(row["assigned_cluster_id"]) for row in paper_assignments
    }

    user_not_parallel, cluster_forbidden_slots = _map_paper_constraints(
        constraints=constraints,
        paper_to_cluster=paper_to_cluster,
    )
    author_not_parallel = _author_parallel_pairs(
        papers=papers,
        paper_to_cluster=paper_to_cluster,
    )
    all_not_parallel = user_not_parallel | author_not_parallel

    sessions_csv = _build_sessions_csv(slot_rows=slot_rows, paper_assignments=paper_assignments)
    slots_csv = _build_slots_csv(slots=slots)

    last_error = ""
    attempt_errors: list[str] = []
    for attempt_idx in range(1, max_assignment_attempts + 1):
        constraints_text = _build_constraints_text(
            not_parallel=all_not_parallel,
            forbidden_slots=cluster_forbidden_slots,
            retry_error=last_error,
        )
        if verbose:
            print(
                f"[postprocess] attempt {attempt_idx}/{max_assignment_attempts}: "
                f"clusters={len(slot_rows)} slots={len(slots)} "
                f"parallel_constraints={len(all_not_parallel)} "
                f"slot_constraints={len(cluster_forbidden_slots)}"
            )
        prompt = render_session_assignment_prompt(
            main_field=main_field,
            sessions_csv=sessions_csv,
            slots_csv=slots_csv,
            constraints_text=constraints_text,
        )
        response_text = _llm_text(client=client, model=general_model, prompt=prompt)

        try:
            assignments = _parse_assignment_csv(response_text)
            if verbose:
                print(f"[postprocess] parsed assignment rows={len(assignments)}")
            positions = _validate_assignments(
                assignments=assignments,
                slot_rows=slot_rows,
                slots=slots,
                not_parallel=all_not_parallel,
                forbidden_slots=cluster_forbidden_slots,
            )
        except ValueError as exc:
            last_error = str(exc)
            attempt_errors.append(f"attempt {attempt_idx}: {last_error}")
            if verbose:
                print(f"[postprocess] validation failed: {last_error}")
            continue

        if verbose:
            print("[postprocess] assignment accepted")

        updated_rows: list[dict[str, Any]] = []
        for row in slot_rows:
            cluster_id = str(row["cluster_id"])
            slot_num, session_num = positions[cluster_id]
            updated = dict(row)
            updated["slot"] = int(slot_num)
            updated["session"] = int(session_num)
            updated_rows.append(updated)

        updated_rows.sort(key=lambda item: (int(item["slot"]), int(item["session"])))
        return updated_rows

    details = "; ".join(attempt_errors) if attempt_errors else (last_error or "unknown error")
    raise ValueError(
        "Constraint conflict: session assignment prompt could not produce a valid schedule "
        f"after {max_assignment_attempts} attempts. Details: {details}"
    )
