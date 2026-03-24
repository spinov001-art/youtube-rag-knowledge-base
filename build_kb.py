"""Build a vector knowledge base from extracted transcripts."""

import argparse
import json
from pathlib import Path

import chromadb
from tqdm import tqdm

from chunker import chunk_text
from embeddings import generate_embeddings


def build_knowledge_base(
    input_dir: str = "transcripts",
    output_dir: str = "knowledge_base",
    chunk_size: int = 500,
    overlap: int = 50,
    embedding_model: str = "all-MiniLM-L6-v2",
    embedding_provider: str = "sentence-transformers",
):
    """Build vector knowledge base from transcript JSON files."""
    input_path = Path(input_dir)
    transcript_files = list(input_path.glob("*.json"))

    if not transcript_files:
        print(f"No transcript files found in {input_dir}/")
        return

    print(f"Found {len(transcript_files)} transcript files")

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=output_dir)
    collection = client.get_or_create_collection(
        name="youtube_knowledge_base",
        metadata={"hnsw:space": "cosine"},
    )

    all_chunks = []
    all_ids = []
    all_metadata = []

    for tf in tqdm(transcript_files, desc="Processing transcripts"):
        data = json.loads(tf.read_text())
        video_id = data["video_id"]
        title = data.get("title", "")
        full_text = data.get("full_text", "")

        if not full_text:
            continue

        chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{video_id}_{i}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadata.append({
                "video_id": video_id,
                "title": title,
                "chunk_index": i,
                "youtube_url": f"https://www.youtube.com/watch?v={video_id}",
            })

    print(f"Total chunks: {len(all_chunks)}")

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = generate_embeddings(
        all_chunks, model=embedding_model, provider=embedding_provider
    )

    # Store in ChromaDB
    print("Storing in vector database...")
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        end = min(i + batch_size, len(all_chunks))
        collection.add(
            ids=all_ids[i:end],
            embeddings=embeddings[i:end],
            documents=[c["text"] for c in all_chunks[i:end]],
            metadatas=all_metadata[i:end],
        )

    print(f"\nKnowledge base built: {len(all_chunks)} chunks in {output_dir}/")
    print(f"Collection: {collection.count()} documents")


def main():
    parser = argparse.ArgumentParser(description="Build knowledge base from transcripts")
    parser.add_argument("--input", default="transcripts", help="Input transcripts directory")
    parser.add_argument("--output", default="knowledge_base", help="Output vector DB path")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--provider", default="sentence-transformers")
    args = parser.parse_args()

    build_knowledge_base(
        input_dir=args.input,
        output_dir=args.output,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        embedding_model=args.model,
        embedding_provider=args.provider,
    )


if __name__ == "__main__":
    main()
