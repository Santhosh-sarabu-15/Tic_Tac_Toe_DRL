import os
import re
from typing import List, Dict, Tuple
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    HAS_RAG_DEPS = True
except ImportError:
    HAS_RAG_DEPS = False


class StrategyRAG:
    """
    RAG (Retrieval-Augmented Generation) engine for Tic-Tac-Toe strategies.
    Uses Sentence Transformers for semantic embeddings and FAISS for vector search.
    """

    def __init__(self, kb_path: str = "knowledge_base/strategies.txt", model_name: str = "all-MiniLM-L6-v2"):
        self.kb_path = kb_path
        self.model_name = model_name
        self.passages: List[Dict[str, str]] = []
        self.index = None
        self.encoder = None

        if HAS_RAG_DEPS:
            try:
                self.encoder = SentenceTransformer(self.model_name)
            except Exception as e:
                print(f"[RAG WARNING] Failed to load SentenceTransformer: {e}")
                self.encoder = None

        self._load_and_index_kb()

    def _load_and_index_kb(self):
        """Parse knowledge base text file into passages and build FAISS index."""
        if not os.path.exists(self.kb_path):
            print(f"[RAG WARNING] Knowledge base file not found at '{self.kb_path}'")
            return

        with open(self.kb_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split by markdown headers
        sections = re.split(r'\n(?=## )', content)
        for sec in sections:
            sec = sec.strip()
            if not sec or sec.startswith("# Tic-Tac-Toe Strategy"):
                continue

            lines = sec.split("\n", 1)
            title = lines[0].replace("##", "").strip()
            text = lines[1].strip() if len(lines) > 1 else title
            
            self.passages.append({
                "title": title,
                "text": text,
                "full_content": f"{title}: {text}"
            })

        if self.encoder is not None and self.passages:
            corpus_texts = [p["full_content"] for p in self.passages]
            embeddings = self.encoder.encode(corpus_texts, convert_to_numpy=True)
            
            # L2 normalize embeddings for Cosine Similarity via Inner Product
            faiss.normalize_L2(embeddings)
            
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(embeddings)

    def query(self, query_text: str, k: int = 2) -> List[Dict]:
        """
        Perform similarity search for query_text.
        Returns list of dicts with passage details and similarity score.
        """
        if not self.passages:
            return []

        if self.encoder is not None and self.index is not None:
            q_emb = self.encoder.encode([query_text], convert_to_numpy=True)
            faiss.normalize_L2(q_emb)
            scores, indices = self.index.search(q_emb, min(k, len(self.passages)))

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.passages):
                    res = self.passages[idx].copy()
                    res["similarity_score"] = float(round(score, 4))
                    results.append(res)
            return results
        else:
            # Fallback simple keyword matching if FAISS/encoder unavailable
            results = []
            q_lower = query_text.lower()
            for p in self.passages:
                score = 0.5 if any(w in p["full_content"].lower() for w in q_lower.split()) else 0.1
                res = p.copy()
                res["similarity_score"] = score
                results.append(res)
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            return results[:k]


if __name__ == "__main__":
    rag = StrategyRAG()
    res = rag.query("blocking opponent two in a row", k=2)
    print("RAG Query Results:")
    for r in res:
        print(f"Title: {r['title']} | Score: {r['similarity_score']}")
        print(f"Text: {r['text']}\n")
