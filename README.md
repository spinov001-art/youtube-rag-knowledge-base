# YouTube RAG Knowledge Base 🎥🧠

**Build an AI knowledge base from any YouTube channel — no API key required.**

Extract transcripts using YouTube's internal Innertube API, chunk text intelligently, generate embeddings, store in a vector database, and query with natural language using RAG (Retrieval-Augmented Generation).

## Why This Exists

A startup founder wanted to learn everything from a 500-video YouTube channel. Watching 500 videos = 250+ hours. With this tool: extract all transcripts in 10 minutes, build a searchable knowledge base, and ask questions in natural language.

**Real use cases:**
- 🎓 Students: Turn lecture playlists into searchable study notes
- 🔬 Researchers: Extract insights from conference talks
- 📈 Marketers: Analyze competitor YouTube content at scale
- 🤖 Developers: Build AI assistants trained on video content

## Quick Start

```bash
# Clone and install
git clone https://github.com/spinov001-art/youtube-rag-knowledge-base.git
cd youtube-rag-knowledge-base
pip install -r requirements.txt

# Extract transcripts from a channel
python extract.py --channel "https://www.youtube.com/@channel_name" --max-videos 50

# Build knowledge base
python build_kb.py --input transcripts/ --output knowledge_base/

# Query your knowledge base
python query.py "What are the best practices for web scraping?"
```

## How It Works

### 1. Transcript Extraction (No API Key!)

Uses YouTube's internal **Innertube API** — the same API the YouTube website uses. No API key, no OAuth, no quotas.

```python
from youtube_transcript import get_transcript

# Extract transcript from any video
transcript = get_transcript("VIDEO_ID")
print(transcript[:500])
```

### 2. Intelligent Chunking

Splits transcripts into semantically meaningful chunks using sentence boundaries and overlap:

```python
from chunker import chunk_text

chunks = chunk_text(transcript, chunk_size=500, overlap=50)
# Each chunk is ~500 tokens with 50-token overlap for context
```

### 3. Embedding Generation

Generate embeddings using OpenAI, Sentence Transformers (free), or Ollama (local):

```python
from embeddings import generate_embeddings

# Free option — Sentence Transformers (runs locally)
embeddings = generate_embeddings(chunks, model="all-MiniLM-L6-v2")

# OpenAI option
embeddings = generate_embeddings(chunks, model="text-embedding-3-small", provider="openai")

# Ollama option (fully local)
embeddings = generate_embeddings(chunks, model="nomic-embed-text", provider="ollama")
```

### 4. Vector Storage

Store in ChromaDB (local, free) or Pinecone (cloud):

```python
from vector_store import VectorStore

store = VectorStore(provider="chroma", path="./knowledge_base")
store.add(chunks, embeddings, metadata)
```

### 5. RAG Query

Ask questions and get answers grounded in actual video content:

```python
from rag import RAGPipeline

rag = RAGPipeline(store, llm="ollama/llama3")  # or "openai/gpt-4"
answer = rag.query("What did they say about machine learning trends?")
print(answer.text)
print(answer.sources)  # Links back to specific videos + timestamps
```

## Architecture

```
YouTube Channel
    ↓ Innertube API (no key needed)
Transcripts (raw text + timestamps)
    ↓ Sentence-aware chunking
Text Chunks (500 tokens, 50 overlap)
    ↓ Embedding model
Vector Embeddings
    ↓ ChromaDB / Pinecone
Vector Database
    ↓ Similarity search + LLM
RAG Answers with Source Citations
```

## Features

| Feature | Description |
|---------|-------------|
| **No API Key** | Uses Innertube API — no Google Cloud setup |
| **Free Tier** | Sentence Transformers + ChromaDB + Ollama = $0 |
| **Timestamps** | Answers link to exact video timestamps |
| **Multi-language** | Supports 100+ languages via YouTube's auto-captions |
| **Batch Processing** | Process entire channels/playlists efficiently |
| **Incremental** | Add new videos without rebuilding entire KB |

## Configuration

```yaml
# config.yaml
extraction:
  max_videos: 100
  languages: ["en", "ru"]  # Preferred transcript languages

chunking:
  chunk_size: 500
  overlap: 50

embeddings:
  provider: "sentence-transformers"  # free, local
  model: "all-MiniLM-L6-v2"

vector_store:
  provider: "chroma"
  path: "./knowledge_base"

llm:
  provider: "ollama"  # free, local
  model: "llama3"
```

## Requirements

- Python 3.9+
- No API keys required for basic usage
- ~500MB disk for Sentence Transformers model
- ~4GB RAM for Ollama (optional, for local LLM)

## Performance

| Channel Size | Extraction | Embedding | Total |
|-------------|-----------|-----------|-------|
| 50 videos | ~2 min | ~1 min | ~3 min |
| 500 videos | ~15 min | ~8 min | ~23 min |
| 5000 videos | ~2 hours | ~1 hour | ~3 hours |

## Related Tools

- [awesome-web-scraping-2026](https://github.com/spinov001-art/awesome-web-scraping-2026) — 77 free web scraping tools
- [youtube-data-tools](https://github.com/spinov001-art/youtube-data-tools) — YouTube scrapers on Apify
- [Innertube API Tutorial](https://dev.to/0012303) — How the Innertube API works

## Need Custom Data Extraction?

I build custom web scraping and data extraction solutions.

👉 [View all 77 free scrapers](https://github.com/spinov001-art/awesome-web-scraping-2026)
👉 [Hire me for custom work](https://github.com/spinov001-art)

## License

MIT — use freely for personal and commercial projects.

## Star History

If this helped you, please ⭐ the repo — it helps others find it!
