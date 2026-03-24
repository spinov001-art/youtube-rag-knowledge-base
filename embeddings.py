"""Generate embeddings using Sentence Transformers (free), OpenAI, or Ollama."""

import json
from pathlib import Path

import requests


def generate_embeddings(
    chunks: list[dict],
    model: str = "all-MiniLM-L6-v2",
    provider: str = "sentence-transformers",
    api_key: str = None,
    batch_size: int = 32,
) -> list[list[float]]:
    """Generate embeddings for text chunks.

    Args:
        chunks: List of dicts with 'text' key
        model: Model name
        provider: One of 'sentence-transformers', 'openai', 'ollama'
        api_key: API key (for OpenAI)
        batch_size: Batch size for processing

    Returns:
        List of embedding vectors
    """
    texts = [c["text"] for c in chunks]

    if provider == "sentence-transformers":
        return _embed_sentence_transformers(texts, model, batch_size)
    elif provider == "openai":
        return _embed_openai(texts, model, api_key, batch_size)
    elif provider == "ollama":
        return _embed_ollama(texts, model)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _embed_sentence_transformers(
    texts: list[str], model: str, batch_size: int
) -> list[list[float]]:
    """Free, local embeddings using Sentence Transformers."""
    from sentence_transformers import SentenceTransformer

    st_model = SentenceTransformer(model)
    embeddings = st_model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return embeddings.tolist()


def _embed_openai(
    texts: list[str], model: str, api_key: str, batch_size: int
) -> list[list[float]]:
    """OpenAI embeddings (paid)."""
    import os
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key=")

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"input": batch, "model": model},
        )
        resp.raise_for_status()
        data = resp.json()
        batch_embeddings = [item["embedding"] for item in data["data"]]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def _embed_ollama(texts: list[str], model: str) -> list[list[float]]:
    """Local embeddings using Ollama."""
    embeddings = []
    for text in texts:
        resp = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": model, "prompt": text},
        )
        resp.raise_for_status()
        embeddings.append(resp.json()["embedding"])
    return embeddings
