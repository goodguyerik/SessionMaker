# Session Maker

Session Maker builds a conference-style schedule from CSV input (slots + papers), using LLM enrichment and assignment.

## What it does

- Groups papers into session slots based on topic fit.
- Tries to keep parallel sessions diverse (not topically too similar).
- Applies default and optional constraints during scheduling/post-processing.
- Writes final and intermediate CSV outputs for review.

## Quick setup

From the repository root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Then edit `.env` and set your real values (especially `OPENAI_API_KEY`).

Load `.env` into your shell and run:

```bash
set -a
source .env
set +a
python main.py --verbose
```

For a first run, `python main.py --verbose` is the best check that everything is wired correctly.

## Files and examples

- `main.py`: main pipeline entry point.
- `resource/slots.csv`: example slot input.
- `resource/papers.csv`: example paper input.
- `resource/constraints.csv`: example user constraints (optional).
- `.env.example`: example environment config (safe to commit).
- `.env`: your local config/secrets (do not commit).

Expected input shape:

- `slots.csv` headers: `slot`, `session`, `duration`
- `papers.csv` headers: `paperid`, `duration`, `paper_title`, `abstract`, `author_emails`
- `constraints.csv`: one rule per line (with header `constraint` or without header)

`author_emails` is used as the person identifier for conflict checks because emails are unique (more reliable than author names).

`slots.csv` example meaning:

- A row `1,1,90` means: slot 1, session 1, duration 90 minutes.
- A row `1,2,90` means: slot 1, session 2, duration 90 minutes.
- Because both rows share `slot=1`, sessions 1 and 2 run in parallel.

Blank lines and lines starting with `#` in `constraints.csv` are ignored.

## Constraints

Default behavior:

- Author conflicts are treated as hard constraints: one author cannot present in parallel sessions.

Allowed user constraint formats:

- `[paper_ids] cannot run in parallel sessions`
- `[paper_ids] cannot run in slots 1,2,3`

Examples:

- `[6,8] cannot run in parallel sessions`
- `[190] cannot run in slots 2,3`

If constraints cannot be satisfied after retry attempts, the tool falls back to the base schedule and returns output without fully enforced user constraints.

## Run commands

Recommended:

```bash
python main.py --verbose
```

Basic:

```bash
python main.py
```

## CLI flags

- `--main-field`: set the conference/topic context for LLM prompts (for example `Database Technology`) to improve summary quality and scheduling relevance.
- `--buffer-left`: override left buffer.
- `--buffer-right`: override right buffer.
- `--max-candidates`: override candidate pool size.
- `--max-attempts`: override max attempts per slot.
- `--constraints-path`: use a custom constraints file.
- `--skip-enrichment`: skip summary/embedding generation and reuse enriched CSV.
- `--skip-postprocess`: skip post-processing reassignment.
- `--verbose`: print detailed logs.

Buffer quick explanation:

- `buffer-left` and `buffer-right` define a safety margin for filling each session.
- This gives the scheduler room to swap papers/presenters and keep constraints feasible.
- Result: a `90` minute slot does not always have to be filled to exactly `90` minutes (for example, `85` can be acceptable depending on buffer settings).

## Outputs

Running `main.py` writes:

- `resource/temp_papers_enriched.csv`
- `resource/temp_slots.csv`
- `resource/temp_papers.csv`
- `resource/temp_unassigned_papers.csv`
- `resource/temp_unfilled_slots.csv`
