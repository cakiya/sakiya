import glob
import os

import numpy as np
from sentence_transformers import SentenceTransformer

from .settings import settings


class PersonaIndex:
    def __init__(self, embedder, lines: list[str], sources: list[str], embeddings):
        self.embedder = embedder
        self.lines = lines
        self.sources = sources
        self.embeddings = embeddings

    def search(self, query: str, top_k: int) -> str:
        if not self.lines or top_k == 0:
            return ""
        query_vec = self.embedder.encode([query], convert_to_numpy=True)
        scores = np.dot(self.embeddings, query_vec.T).squeeze()

        # Handle single-entry array dimensions
        if scores.ndim == 0:
            scores = np.array([scores.item()])

        actual_k = min(top_k, len(scores))
        if actual_k == 0:
            return ""

        top_indices = np.argsort(scores)[::-1][:actual_k]

        retrieved_chunks = []
        for i in top_indices:
            retrieved_chunks.append(f"[Source: {self.sources[i]}] {self.lines[i]}")

        return "\n".join(retrieved_chunks)


_index: PersonaIndex | None = None


def build_index() -> PersonaIndex:
    global _index

    print("Loading embedding model and knowledge files...")
    embedder = SentenceTransformer(settings.embedding_model, local_files_only=settings.embedding_local_only)

    history_lines: list[str] = []
    history_sources: list[str] = []

    for filepath in glob.glob(os.path.join(settings.personas_dir, "*.txt")):
        filename = os.path.basename(filepath)
        try:
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    stripped_line = line.strip()
                    if stripped_line:
                        # Slice massive paragraphs into smaller, digestible pieces
                        for i in range(0, len(stripped_line), settings.rag_chunk_chars):
                            history_lines.append(stripped_line[i:i + settings.rag_chunk_chars])
                            history_sources.append(filename)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    if history_lines:
        history_embeddings = embedder.encode(history_lines, convert_to_numpy=True)
        print(f"Successfully embedded {len(history_lines)} lines across {len(set(history_sources))} file(s).")
    else:
        print("WARNING: No text files found in the personas folder. Running without RAG context.")
        history_embeddings = np.array([])

    _index = PersonaIndex(embedder, history_lines, history_sources, history_embeddings)
    return _index


def get_relevant_context(query: str, top_k: int | None = None) -> str:
    if _index is None:
        return ""
    return _index.search(query, settings.rag_top_k if top_k is None else top_k)
