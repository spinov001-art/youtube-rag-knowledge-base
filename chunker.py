"""Intelligent text chunking for RAG — sentence-aware with overlap."""

import re


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    respect_sentences: bool = True,
) -> list[dict]:
    """Split text into chunks with overlap.

    Args:
        text: Input text to chunk
        chunk_size: Target chunk size in words
        overlap: Number of overlapping words between chunks
        respect_sentences: If True, break at sentence boundaries

    Returns:
        List of dicts with 'text', 'start_word', 'end_word' keys
    """
    if not text.strip():
        return []

    if respect_sentences:
        sentences = _split_sentences(text)
        return _chunk_sentences(sentences, chunk_size, overlap)

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append({
            "text": " ".join(chunk_words),
            "start_word": start,
            "end_word": end,
        })
        start += chunk_size - overlap

    return chunks


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    # Simple but effective sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def _chunk_sentences(
    sentences: list[str],
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    """Chunk sentences respecting boundaries."""
    chunks = []
    current_words = []
    current_sentences = []
    word_offset = 0

    for sentence in sentences:
        words = sentence.split()

        if len(current_words) + len(words) > chunk_size and current_words:
            # Save current chunk
            chunks.append({
                "text": " ".join(current_words),
                "start_word": word_offset,
                "end_word": word_offset + len(current_words),
            })

            # Calculate overlap — keep last N words worth of sentences
            overlap_words = 0
            overlap_start = len(current_sentences)
            for i in range(len(current_sentences) - 1, -1, -1):
                s_words = len(current_sentences[i].split())
                if overlap_words + s_words > overlap:
                    break
                overlap_words += s_words
                overlap_start = i

            word_offset += len(current_words) - overlap_words
            current_sentences = current_sentences[overlap_start:]
            current_words = []
            for s in current_sentences:
                current_words.extend(s.split())

        current_sentences.append(sentence)
        current_words.extend(words)

    if current_words:
        chunks.append({
            "text": " ".join(current_words),
            "start_word": word_offset,
            "end_word": word_offset + len(current_words),
        })

    return chunks


if __name__ == "__main__":
    # Demo
    sample = "This is a test. " * 100
    chunks = chunk_text(sample, chunk_size=50, overlap=10)
    print(f"Text: {len(sample.split())} words → {len(chunks)} chunks")
    for i, c in enumerate(chunks[:3]):
        print(f"  Chunk {i}: {len(c['text'].split())} words")
