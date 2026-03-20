"""Prompt templates for conference session scheduling."""

from __future__ import annotations


SUMMARY_REWRITE_PROMPT = """
You are an expert in {main_field} and a conference session chair.

Rewrite the abstract as a compact 2-3 sentence description optimized for grouping into conference sessions.

Prioritize, in order:
1) Core task/problem (what is being solved)
2) Data/setting (what kind of data + domain/context)
3) Evaluation/deployment constraints (benchmarks/metrics/real-time/interpretability/etc.)
4) Method family (ONLY as a brief qualifier, not the focus)

Hard rules:
- Do NOT lead with model names or architectures unless the contribution is mainly methodological.
- Use specific, discriminative terms (e.g., "multivariate forecasting", "streaming anomaly detection", "irregularly-sampled EHR time series").
- Include 3-6 concrete keywords/phrases naturally inside the sentences (not as a list).
- Avoid generic filler ("novel", "effective", "state-of-the-art") unless tied to a named benchmark/metric.

Output ONLY the rewritten text (no headings, no meta-commentary).
""".strip()


SESSION_SELECTION_PROMPT = """
You are a conference chair. Based on the given paper abstracts, create exactly one coherent session (one topic) from the provided papers. Each paper includes its title, abstract, and a presentation length.

Session length: {session_length} minutes.
Allowed presentation lengths: {presentation_times_text} minutes.

Timing rules (relaxed schedule with buffer):
- Use ONLY the given presentation lengths: {presentation_times_text} minutes.
- Do NOT exceed the session length: total_talk_time <= {session_length}.
- Intentionally leave buffer time for transitions/Q&A. Target ~{buffer_left}-{buffer_right} minutes of buffer when possible.
- Prefer totals in the range [{session_length}-{buffer_right}, {session_length}-{buffer_left}] when you can do so without hurting topical coherence.
- If that range is not achievable, choose the best coherent set that is still <= {session_length}.
- If multiple options are similarly coherent, prefer the one with MORE buffer (i.e., smaller total talk time).

OUTPUT FORMAT (strict):
Return ONLY a JSON array. Each element must be an object with:
- "title": the selected paper title (exactly as given)
- "length": one selected presentation length from {presentation_times_text}

Do not output anything else.

PAPERS (choose from these only):
{papers_block}
""".strip()


COHERENCE_EVAL_PROMPT = """
You are an expert in {main_field} and an experienced conference session chair.

You will see papers that were grouped into one cluster (intended to become ONE conference session).
Your job is to judge whether these papers fit together for a single session audience.

IMPORTANT: Method diversity is NORMAL within a subfield.
Do NOT mark papers as misaligned just because they use different models/techniques
(e.g., transformers vs. ARIMA vs. diffusion) IF they target the same core task/subfield community.
Only treat "different methods" as misalignment when it implies a different subfield/audience.

Define "coherence" primarily by:
- core task/problem (e.g., forecasting, anomaly detection, representation learning, causal discovery)
- data setting
- evaluation culture / target audience
Secondary signals (lower weight):
- specific method family, architecture, optimization tricks

TASK A - Session coherence rating (1-5)
Rate how well these papers fit into ONE session for a shared audience.

1 = Not a session: multiple distinct subfields/audiences
2 = Loose umbrella: only shares a broad keyword; audience would split
3 = Same broad area, but would form 2+ better sessions (different tasks/audiences)
4 = Clear session: shared core task/audience; methods may vary
5 = Very tight session: highly focused on the same narrow task + setting

TASK B - Theme (session title)
Provide a short session theme (<= {theme_max_words} words) describing the shared task + setting.
The theme must be {main_field}-flavored and use terminology natural for a {main_field} conference audience.
Example guidance: if main_field is "Database Technology", the title should sound DB-flavored.

Return VALID JSON with exactly these keys:
- "coherence_score": integer 1-5
- "theme": string (<= {theme_max_words} words, {main_field}-flavored)

JSON rules:
- Output JSON only, no markdown and no code fences.
- Include all required keys even when values are empty.
- Do not add any extra keys.

PAPERS IN CLUSTER {cluster_id}:
{papers_block}
""".strip()

SESSION_ASSIGNMENT_PROMPT = """
You are an expert in {main_field} and a conference program chair.

Assign the given conference sessions to the available time slots.

Prioritize, in order:
1) Satisfying all hard scheduling constraints
2) Matching each session only to a slot with the same duration
3) Maximizing diversity across parallel sessions
4) Avoiding parallel placement of sessions with similar topics, overlapping audiences, or likely attendee conflict

Inputs:
Sessions (A list of sessions, each with: cluster id, session name, number of papers, duration.):

{sessions_csv}

Available slots (A list of available slots, each with: slot id, parallel track id, duration.):

{slots_csv}

Hard constraints (A list of hard constraints):

{constraints_text}

Interpretation rules:
- Sessions sharing the same slot number run in parallel
- Each session must be assigned to exactly one slot
- Each slot can host at most one session
- Session duration and slot duration must match exactly

Hard rules:
- Forbidden parallel pairs must never be scheduled in the same slot number
- Slot-restricted clusters must never be scheduled in forbidden slot numbers
- All sessions must be scheduled
- Do not leave a session unassigned

Scheduling preference:
- As a listener, I should have the chance to attend as many interesting talks as possible
- Therefore, parallel sessions should be as diverse as possible in topic and audience
- If several feasible schedules exist, choose the one with the strongest thematic separation across concurrent tracks

Output ONLY a CSV table with these columns:
slot,track,cluster,session_name,duration

Do not output explanations, reasoning, bullets, or code fences.
""".strip()

def render_summary_prompt(main_field: str) -> str:
    return SUMMARY_REWRITE_PROMPT.format(main_field=main_field)


def render_session_selection_prompt(
    session_length: int,
    papers_block: str,
    presentation_times: list[int] | tuple[int, ...] = (10, 18),
    buffer_left: int = 2,
    buffer_right: int = 12,
) -> str:
    presentation_times_text = " or ".join(str(v) for v in presentation_times)
    return SESSION_SELECTION_PROMPT.format(
        session_length=session_length,
        presentation_times_text=presentation_times_text,
        buffer_left=buffer_left,
        buffer_right=buffer_right,
        papers_block=papers_block,
    )


def render_coherence_prompt(
    main_field: str,
    cluster_id: str,
    papers_block: str,
    theme_max_words: int = 8,
) -> str:
    return COHERENCE_EVAL_PROMPT.format(
        main_field=main_field,
        cluster_id=cluster_id,
        papers_block=papers_block,
        theme_max_words=theme_max_words,
    )


def render_session_assignment_prompt(
    main_field: str,
    sessions_csv: str,
    slots_csv: str,
    constraints_text: str,
) -> str:
    return SESSION_ASSIGNMENT_PROMPT.format(
        main_field=main_field,
        sessions_csv=sessions_csv,
        slots_csv=slots_csv,
        constraints_text=constraints_text,
    )
