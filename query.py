"""Query the YouTube knowledge base using RAG."""

import argparse
import json

import chromadb
import requests

from embeddings import generate_embeddings


class RAGPipeline:
    """RAG pipeline for querying YouTube knowledge base."""

    def __init__(
        self,
        kb_path: str = "knowledge_base",
        llm_provider: str = "ollama",
        llm_model: str = "llama3",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_provider: str = "sentence-transformers",
        top_k: int = 5,
    ):
        self.client = chromadb.PersistentClient(path=kb_path)
        self.collection = self.client.get_collection("youtube_knowledge_base")
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.top_k = top_k

    def query(self, question: str) -> dict:
        """Query the knowledge base and generate an answer."""
        # Generate embedding for the question
        q_embedding = generate_embeddings(
            [{"text": question}],
            model=self.embedding_model,
            provider=self.embedding_provider,
        )[0]

        # Search vector database
        results = self.collection.query(
            query_embeddings=[q_embedding],
            n_results=self.top_k,
        )

        # Build context from results
        contexts = []
        sources = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            contexts.append(f"[Source: {meta['title']}]\n{doc}")
            sources.append({
                "title": meta["title"],
                "url": meta["youtube_url"],
                "video_id": meta["video_id"],
            })

        context_text = "\n\n---\n\n".join(contexts)

        # Generate answer using LLM
        prompt = f"""Based on the following excerpts from YouTube videos, answer the question.
If the answer is not in the provided context, say so.
Always cite which video the information comes from.

Context:
{context_text}

Question: {question}

Answer:"""

        answer = self._call_llm(prompt)

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "num_sources": len(sources),
        }

    def _call_llm(self, prompt: str) -> str:
        """Call LLM for answer generation."""
        if self.llm_provider == "ollama":
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": self.llm_model, "prompt": prompt, "stream": False},
            )
            resp.raise_for_status()
            return resp.json()["response"]

        elif self.llm_provider == "openai":
            import os
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                json={
                    "model": self.llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

        else:
            # Fallback: return context without LLM
            return f"[No LLM configured. Raw context provided.]\n\n{prompt}"


def main():
    parser = argparse.ArgumentParser(description="Query YouTube knowledge base")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("--kb-path", default="knowledge_base")
    parser.add_argument("--llm", default="ollama/llama3", help="LLM in format provider/model")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    llm_parts = args.llm.split("/", 1)
    llm_provider = llm_parts[0]
    llm_model = llm_parts[1] if len(llm_parts) > 1 else "llama3"

    rag = RAGPipeline(
        kb_path=args.kb_path,
        llm_provider=llm_provider,
        llm_model=llm_model,
        top_k=args.top_k,
    )

    if args.interactive:
        print("YouTube Knowledge Base — Interactive Query Mode")
        print("Type 'quit' to exit\n")
        while True:
            question = input("Question: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                break
            result = rag.query(question)
            print(f"\nAnswer: {result['answer']}")
            print(f"\nSources ({result['num_sources']}):")
            for s in result["sources"]:
                print(f"  - {s['title']}: {s['url']}")
            print()
    elif args.question:
        result = rag.query(args.question)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
