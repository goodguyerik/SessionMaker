import argparse
import json
from pathlib import Path
import pandas as pd

from session_maker.csv_import import (
    parse_enriched_papers_csv,
    parse_papers_csv,
    parse_slots_csv,
)
from session_maker.config import load_settings
from session_maker.constraints import parse_constraints_csv
from session_maker.llm import build_client
from session_maker.pipeline import enrich_papers
from session_maker.postprocess import reassign_clusters
from session_maker.scheduler import schedule_papers

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run session maker pipeline")
    parser.add_argument("--main-field", type=str, default=None)
    parser.add_argument("--buffer-left", type=int, default=None)
    parser.add_argument("--buffer-right", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--constraints-path", type=str, default=None)
    parser.add_argument("--skip-enrichment", action="store_true")
    parser.add_argument("--skip-postprocess", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = _parse_args()
    settings = load_settings()

    token = str(settings["api_token"])
    if not token:
        raise ValueError("Missing OPENAI_API_KEY")

    base_url = str(settings["api_base_url"])
    general_model = str(settings["general_model"])
    embedding_model = str(settings["embedding_model"])
    main_field = args.main_field if args.main_field else str(settings["main_field"])
    slots_path = Path(settings["slots_path"])
    papers_path = Path(settings["papers_path"])
    constraints_path = (
        Path(args.constraints_path)
        if args.constraints_path
        else Path(settings["constraints_path"])
        if str(settings["constraints_path"])
        else None
    )
    output_path = Path(settings["output_path"])
    temperature = float(settings["temperature"])
    buffer_left = (
        args.buffer_left if args.buffer_left is not None else int(settings["buffer_left"])
    )
    buffer_right = (
        args.buffer_right if args.buffer_right is not None else int(settings["buffer_right"])
    )
    max_candidates = (
        args.max_candidates
        if args.max_candidates is not None
        else int(settings["max_candidates"])
    )
    max_attempts = (
        args.max_attempts
        if args.max_attempts is not None
        else int(settings["max_attempts_per_slot"])
    )
    verbose = bool(args.verbose)

    slots_output_path = output_path.parent / "temp_slots.csv"
    papers_output_path = output_path.parent / "temp_papers.csv"
    unassigned_path = output_path.parent / "temp_unassigned_papers.csv"
    unfilled_slots_path = output_path.parent / "temp_unfilled_slots.csv"

    slots = parse_slots_csv(slots_path)
    papers = parse_papers_csv(papers_path)
    presentation_times = sorted({int(paper["duration"]) for paper in papers})

    print(f"Loaded {len(slots)} slots and {len(papers)} papers")
    print(f"Presentation times from papers: {presentation_times}")

    client = build_client(api_token=token, api_base_url=base_url)

    if args.skip_enrichment:
        if not output_path.exists():
            raise ValueError(
                "--skip-enrichment requires an existing enriched CSV at "
                f"{output_path}"
            )
        enriched_rows = parse_enriched_papers_csv(output_path)
        print(
            "Skipping enrichment (--skip-enrichment); "
            f"loaded {len(enriched_rows)} rows from {output_path}"
        )
    else:
        enriched_rows = enrich_papers(
            papers=papers,
            client=client,
            general_model=general_model,
            embedding_model=embedding_model,
            main_field=main_field,
            temperature=temperature,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df = pd.DataFrame(enriched_rows)
        output_df["embedding"] = output_df["embedding"].apply(json.dumps)
        output_df = output_df[
            ["paperid", "duration", "paper_title", "author_emails", "summary", "embedding"]
        ]
        output_df.to_csv(output_path, index=False)

        print(f"Wrote {len(enriched_rows)} rows to {output_path}")

    slot_rows, paper_assignments, unassigned_papers, unfilled_slots = schedule_papers(
        client=client,
        general_model=general_model,
        main_field=main_field,
        slots=slots,
        enriched_papers=enriched_rows,
        max_candidates=max_candidates,
        max_attempts_per_slot=max_attempts,
        buffer_left=buffer_left,
        buffer_right=buffer_right,
        verbose=verbose,
    )

    parsed_constraints = {
        "paper_not_parallel": [],
        "paper_forbidden_slots": {},
        "raw_constraints": [],
    }
    if constraints_path:
        parsed_constraints = parse_constraints_csv(constraints_path)
        raw_constraints = parsed_constraints.get("raw_constraints", [])
        raw_count = len(raw_constraints) if isinstance(raw_constraints, list) else 0
        ignored_comment_lines_raw = parsed_constraints.get("ignored_comment_lines", 0)
        ignored_comment_lines = (
            ignored_comment_lines_raw
            if isinstance(ignored_comment_lines_raw, int)
            else 0
        )
        print(
            "Loaded constraints: "
            f"{raw_count} rules from {constraints_path}"
        )
        if verbose:
            if raw_count == 0:
                print("Processed constraints: none")
            else:
                print("Processed constraints:")
                for constraint in raw_constraints if isinstance(raw_constraints, list) else []:
                    print(f"- {constraint}")
        if raw_count == 0 and ignored_comment_lines > 0:
            print(
                "Warning: constraints file contains only commented rules; "
                "no constraints will be enforced"
            )

    base_slot_rows = [dict(row) for row in slot_rows]
    postprocess_applied = False

    if args.skip_postprocess:
        print("Skipping post-processing reassignment (--skip-postprocess)")
    else:
        try:
            slot_rows = reassign_clusters(
                slot_rows=slot_rows,
                paper_assignments=paper_assignments,
                papers=papers,
                slots=slots,
                client=client,
                general_model=general_model,
                main_field=main_field,
                parsed_constraints=parsed_constraints,
                verbose=verbose,
            )
            postprocess_applied = True
        except ValueError as exc:
            slot_rows = base_slot_rows
            print(
                "Post-processing failed; using base schedule without constraint enforcement"
            )
            if verbose:
                print(f"Post-processing error details: {exc}")

    assignment_map = {
        int(row["paperid"]): str(row["assigned_cluster_id"]) for row in paper_assignments
    }
    papers_with_clusters_df = pd.DataFrame(papers)
    papers_with_clusters_df["assigned_cluster_id"] = papers_with_clusters_df["paperid"].apply(
        lambda paper_id: assignment_map.get(int(paper_id), "")
    )

    pd.DataFrame(slot_rows).to_csv(slots_output_path, index=False)
    papers_with_clusters_df.to_csv(papers_output_path, index=False)
    pd.DataFrame(unassigned_papers).to_csv(unassigned_path, index=False)
    pd.DataFrame(unfilled_slots).to_csv(unfilled_slots_path, index=False)

    if args.skip_postprocess:
        print("Final slots source: base_schedule_skipped_postprocess")
    elif postprocess_applied:
        print("Final slots source: postprocessed")
    else:
        print("Final slots source: fallback_base_schedule")

    print(f"Wrote {len(slot_rows)} slot rows to {slots_output_path}")
    print(f"Wrote {len(papers_with_clusters_df)} paper rows to {papers_output_path}")
    print(f"Wrote {len(unassigned_papers)} unassigned papers to {unassigned_path}")
    print(f"Wrote {len(unfilled_slots)} unfilled slots to {unfilled_slots_path}")

if __name__ == "__main__":
    main()
