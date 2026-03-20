from openai import OpenAI
from session_maker.prompts import render_summary_prompt

def build_client(api_token: str, api_base_url: str) -> OpenAI:
    return OpenAI(api_key=api_token, base_url=api_base_url)

def summarize_abstract(
    client: OpenAI,
    general_model: str,
    abstract: str,
    main_field: str,
    temperature: float,
) -> str:
    response = client.chat.completions.create(
        model=general_model,
        messages=[
            {"role": "system", "content": render_summary_prompt(main_field)},
            {"role": "user", "content": abstract},
        ],
        temperature=temperature,
    )
    content = response.choices[0].message.content
    return content.strip() if content else ""

def embed_text(client: OpenAI, embedding_model: str, text: str) -> list[float]:
    response = client.embeddings.create(model=embedding_model, input=text)
    return response.data[0].embedding