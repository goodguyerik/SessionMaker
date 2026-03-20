from openai import OpenAI
from session_maker.llm import embed_text, summarize_abstract

def enrich_papers(
    papers: list[dict[str, int | str]],
    client: OpenAI,
    general_model: str,
    embedding_model: str,
    main_field: str,
    temperature: float,
) -> list[dict[str, int | str | list[float]]]:
    enriched_rows: list[dict[str, int | str | list[float]]] = []
    total = len(papers)

    for idx, paper in enumerate(papers, start=1):
        print(f"Processing paper {idx}/{total}")
        summary = summarize_abstract(
            client=client,
            general_model=general_model,
            abstract=str(paper["abstract"]),
            main_field=main_field,
            temperature=temperature,
        )
        embedding = embed_text(client=client, embedding_model=embedding_model, text=summary)

        enriched_rows.append(
            {
                "paperid": int(paper["paperid"]),
                "duration": int(paper["duration"]),
                "paper_title": str(paper["paper_title"]),
                "author_emails": str(paper["author_emails"]),
                "summary": summary,
                "embedding": embedding,
            }
        )

    return enriched_rows