import os

def load_settings() -> dict[str, str]:
    return {
        "api_token": os.getenv("OPENAI_API_KEY", "").strip(),
        "api_base_url": os.getenv("API_BASE_URL", "https://api.openai.com/v1").strip(),
        "general_model": os.getenv("GENERAL_MODEL", "gpt-5").strip(),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-large").strip(),
        "main_field": os.getenv("MAIN_FIELD", "Database Technology").strip(),
        "slots_path": os.getenv("SLOTS_PATH", "resource/slots.csv").strip(),
        "papers_path": os.getenv("PAPERS_PATH", "resource/papers.csv").strip(),
        "constraints_path": os.getenv("CONSTRAINTS_PATH", "resource/constraints.csv").strip(),
        "output_path": os.getenv(
            "OUTPUT_PATH", "resource/temp_papers_enriched.csv"
        ).strip(),
        "temperature": os.getenv("TEMPERATURE", "0.2").strip(),
        "buffer_left": os.getenv("BUFFER_LEFT", "2").strip(),
        "buffer_right": os.getenv("BUFFER_RIGHT", "12").strip(),
        "max_candidates": os.getenv("MAX_CANDIDATES", "100").strip(),
        "max_attempts_per_slot": os.getenv("MAX_ATTEMPTS_PER_SLOT", "8").strip(),
    }
