import os
import pytest
from rag import StrategyRAG


def test_rag_knowledge_base_loading():
    rag = StrategyRAG(kb_path="knowledge_base/strategies.txt")
    assert len(rag.passages) >= 5, "RAG should load multiple strategy sections from strategies.txt"


def test_rag_query_retrieval():
    rag = StrategyRAG(kb_path="knowledge_base/strategies.txt")
    results = rag.query("blocking opponent immediate win threat", k=2)

    assert len(results) > 0
    assert "title" in results[0]
    assert "text" in results[0]
    assert "similarity_score" in results[0]
    assert results[0]["similarity_score"] > 0.0


def test_rag_empty_or_missing_file():
    rag = StrategyRAG(kb_path="non_existent_file.txt")
    results = rag.query("center control")
    assert results == []
