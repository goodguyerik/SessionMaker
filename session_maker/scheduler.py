import json
import random
from typing import Any
from openai import OpenAI
from session_maker.prompts import render_coherence_prompt, render_session_selection_prompt

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def _pick_candidate_pool(
    remaining: list[dict[str, Any]],
    max_candidates: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if len(remaining) <= max_candidates:
        pool = list(remaining)
        rng.shuffle(pool)
        return pool
    seed = rng.choice(remaining)
    scored = [
        (paper, _cosine_similarity(seed["embedding"], paper["embedding"]))
        for paper in remaining
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return [paper for paper, _ in scored[:max_candidates]]

def _extract_json_fragment(text: str, start_token: str, end_token: str) -> Any:
    start = text.find(start_token)
    end = text.rfind(end_token)
    if start == -1 or end == -1 or end < start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data

def _extract_json_array(text: str) -> list[dict[str, Any]]:
    data = _extract_json_fragment(text, "[", "]")
    return data if isinstance(data, list) else []

def _extract_json_object(text: str) -> dict[str, Any]:
    data = _extract_json_fragment(text, "{", "}")
    return data if isinstance(data, dict) else {}

def _build_papers_block(papers: list[dict[str, Any]], include_length: bool) -> str:
    lines = []
    for paper in papers:
        block = f"- Title: {paper['paper_title']}\n"
        if include_length:
            block += f"  Length: {paper['duration']}\n"
        block += f"  Abstract: {paper['summary']}"
        lines.append(block)
    return "\n".join(lines)

def _llm_text(client: OpenAI, model: str, system_prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}],
        temperature=0,
    )
    content = response.choices[0].message.content
    return content.strip() if content else ""

def _attempt_slot(
    client: OpenAI,
    general_model: str,
    main_field: str,
    slot_duration: int,
    slot_label: str,
    candidates: list[dict[str, Any]],
    presentation_times: list[int],
    buffer_left: int,
    buffer_right: int,
) -> tuple[list[dict[str, Any]], int, str]:
    by_title = {paper["paper_title"]: paper for paper in candidates}

    selection_prompt = render_session_selection_prompt(
        session_length=slot_duration,
        papers_block=_build_papers_block(candidates, include_length=True),
        presentation_times=presentation_times,
        buffer_left=buffer_left,
        buffer_right=buffer_right,
    )
    selection_text = _llm_text(client, general_model, selection_prompt)
    selected_raw = _extract_json_array(selection_text)

    selected_titles: list[str] = []
    for item in selected_raw:
        if isinstance(item, dict) and isinstance(item.get("title"), str):
            selected_titles.append(item["title"])

    picked: list[dict[str, Any]] = []
    used = set()
    total_duration = 0
    for title in selected_titles:
        paper = by_title.get(title)
        if paper is None or paper["paperid"] in used:
            continue
        paper_duration = int(paper["duration"])
        if total_duration + paper_duration > slot_duration:
            continue
        picked.append(paper)
        used.add(paper["paperid"])
        total_duration += paper_duration

    if not picked:
        return [], 0, ""

    coherence_prompt = render_coherence_prompt(
        main_field=main_field,
        cluster_id=slot_label,
        papers_block=_build_papers_block(picked, include_length=False),
    )
    coherence_text = _llm_text(client, general_model, coherence_prompt)
    coherence_obj = _extract_json_object(coherence_text)
    score = coherence_obj.get("coherence_score", 0)
    try:
        coherence_score = int(score)
    except (TypeError, ValueError):
        coherence_score = 0
    theme = coherence_obj.get("theme", "")
    topic = theme.strip() if isinstance(theme, str) else ""
    return picked, coherence_score, topic

def schedule_papers(
    client: OpenAI,
    general_model: str,
    main_field: str,
    slots: list[dict[str, int]],
    enriched_papers: list[dict[str, Any]],
    max_candidates: int = 100,
    max_attempts_per_slot: int = 8,
    buffer_left: int = 2,
    buffer_right: int = 12,
    verbose: bool = False,
    seed: int = 7,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, int]]]:
    remaining = [dict(paper) for paper in enriched_papers]
    slots_ordered = sorted(slots, key=lambda s: (-s["duration"], s["slot"], s["session"]))
    presentation_times = sorted({int(p["duration"]) for p in remaining})
    slot_rows: list[dict[str, Any]] = []
    paper_rows: list[dict[str, Any]] = []
    rng = random.Random(seed)
    cluster_counter = 1

    for slot_index, slot in enumerate(slots_ordered, start=1):
        if not remaining:
            break

        slot_duration = int(slot["duration"])
        slot_label = f"slot-{slot['slot']}-session-{slot['session']}"

        if verbose:
            print(
                f"[slot {slot_index}/{len(slots_ordered)}] "
                f"slot={slot['slot']} session={slot['session']} duration={slot_duration} "
                f"remaining papers={len(remaining)}"
            )

        best_picked: list[dict[str, Any]] = []
        best_score = -1
        best_total = -1
        best_topic = ""
        accepted = False

        for attempt_idx in range(1, max_attempts_per_slot + 1):
            candidates = _pick_candidate_pool(remaining, max_candidates, rng)
            picked, score, topic = _attempt_slot(
                client=client,
                general_model=general_model,
                main_field=main_field,
                slot_duration=slot_duration,
                slot_label=slot_label,
                candidates=candidates,
                presentation_times=presentation_times,
                buffer_left=buffer_left,
                buffer_right=buffer_right,
            )
            total = sum(int(p["duration"]) for p in picked)
            if verbose:
                print(
                    f"  attempt {attempt_idx}/{max_attempts_per_slot}: "
                    f"candidates={len(candidates)} picked={len(picked)} "
                    f"total={total} score={score}"
                )
            if score > best_score or (score == best_score and total > best_total):
                best_picked = picked
                best_score = score
                best_total = total
                best_topic = topic
            if score >= 4 and picked:
                accepted = True
                break

        if not best_picked:
            fallback = sorted(remaining, key=lambda p: int(p["duration"]), reverse=True)
            total = 0
            for paper in fallback:
                d = int(paper["duration"])
                if total + d <= slot_duration:
                    best_picked.append(paper)
                    total += d
            best_total = total
            best_score = 0
            best_topic = "Fallback Cluster"

        selected_ids = {int(p["paperid"]) for p in best_picked}
        if best_picked:
            cluster_id = f"C{cluster_counter:03d}"
            cluster_counter += 1

            slot_rows.append(
                {
                    "cluster_id": cluster_id,
                    "slot": int(slot["slot"]),
                    "session": int(slot["session"]),
                    "slot_duration": slot_duration,
                    "used_duration": int(best_total),
                    "buffer_minutes": int(slot_duration - best_total),
                    "coherence_score": int(best_score),
                    "topic": best_topic,
                }
            )

            for paper in best_picked:
                paper_rows.append(
                    {
                        "paperid": int(paper["paperid"]),
                        "assigned_cluster_id": cluster_id,
                    }
                )

        if verbose:
            decision = "accepted (score>=4)" if accepted else "best attempt fallback"
            print(
                f"  -> {decision}: picked={len(best_picked)} total={best_total} "
                f"score={best_score} topic='{best_topic}'"
            )

        remaining = [p for p in remaining if int(p["paperid"]) not in selected_ids]
        if verbose:
            print(f"  remaining after slot: {len(remaining)}")

    used_slots = {(a["slot"], a["session"]) for a in slot_rows}
    unfilled_slots = [
        s for s in slots_ordered if (int(s["slot"]), int(s["session"])) not in used_slots
    ]
    return slot_rows, paper_rows, remaining, unfilled_slots