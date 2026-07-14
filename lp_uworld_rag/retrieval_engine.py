"""Shared retrieval primitives -- the config-agnostic building blocks both retrieval engines stand
on: ``index.py`` (the docs side: two functional/technical collections, citation resolve) and
``direct_index.py`` (the code side: one repo's "index"-mode Chroma store). Each engine keeps its own
ranking *policy* (quota, normalize/rerank order, hint boost, priority reorder, citation join); only
the *plumbing* lives here, so "load the embed model / open Chroma / build the fusion retriever /
rerank" has one definition instead of two byte-identical copies.

These take plain arguments (not a config object) precisely so both a ``RagConfig`` caller and an
``IndexConfig`` caller can share them. Heavy ML imports stay lazy (inside each function) so the
CLI's ``--help`` and config loading keep working without llama-index/chromadb installed.
"""
from __future__ import annotations

import os


def pool_size(configured: int, top_k: int) -> int:
    """Candidate-pool size for retrieval -- never smaller than the caller's top_k."""
    return max(configured, top_k)


def round_score(score) -> float | None:
    """A result chunk's score rounded for the JSON payload (None passes through)."""
    return round(float(score), 6) if score is not None else None


def resolve_device(device: str | None) -> str:
    """An explicit device string, or auto-detect (cuda if available, else cpu) when ``device`` is
    falsy or ``"auto"``."""
    if device and device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def build_embed_model(*, model: str, trust_remote_code: bool, query_prefix: str,
                      text_prefix: str, device: str | None):
    """A normalized HuggingFace embedding with asymmetric task prefixes -- the single constructor
    both engines' ``get_embed_model`` wrappers call."""
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(
        model_name=model,
        trust_remote_code=trust_remote_code,
        query_instruction=query_prefix,
        text_instruction=text_prefix,
        device=resolve_device(device),
        normalize=True,
    )


def open_chroma_client(persist_path: str):
    """The one place a Chroma ``PersistentClient`` is opened -- every persist dir is its own
    independent local database (see the docs' StoreConfig / each repo's own
    ``index.store.persistDir``). Callers that need to drop a collection (a ``--full`` re-ingest)
    hold the client; callers that only read a collection use ``open_chroma_collection`` below."""
    import chromadb

    return chromadb.PersistentClient(path=persist_path)


def open_chroma_collection(persist_path: str, collection: str):
    """Open the client and fetch (or create) one collection -- the read path's convenience over
    ``open_chroma_client``."""
    return open_chroma_client(persist_path).get_or_create_collection(collection)


def load_vector_index(persist_path: str, collection: str, embed_model):
    from llama_index.core import VectorStoreIndex
    from llama_index.vector_stores.chroma import ChromaVectorStore

    vstore = ChromaVectorStore(chroma_collection=open_chroma_collection(persist_path, collection))
    return VectorStoreIndex.from_vector_store(vstore, embed_model=embed_model)


def load_docstore(docstore_json_path: str):
    """The persisted ``llama_index`` docstore (BM25 corpus + every chunk's node), or None if this
    persist dir hasn't been ingested yet -- callers fall back to vector-only in that case."""
    from llama_index.core.storage.docstore import SimpleDocumentStore

    if not os.path.exists(docstore_json_path):
        return None
    return SimpleDocumentStore.from_persist_path(docstore_json_path)


def build_bm25(docstore, pool_size: int):
    from llama_index.retrievers.bm25 import BM25Retriever

    return BM25Retriever.from_defaults(docstore=docstore, similarity_top_k=pool_size)


def build_fusion_retriever(vector_retriever, bm25, *, top_k: int, fusion_mode: str,
                           num_queries: int):
    """Hybrid dense+BM25 RRF fusion -- or the plain vector retriever when ``bm25`` is None (a
    persist dir whose ingest hasn't written a docstore yet). ``num_queries=1`` means no query
    generation and therefore no real LLM call, hence the ``MockLLM``."""
    if bm25 is None:
        return vector_retriever

    from llama_index.core.llms import MockLLM
    from llama_index.core.retrievers import QueryFusionRetriever

    return QueryFusionRetriever(
        [vector_retriever, bm25],
        mode=fusion_mode,
        num_queries=num_queries,
        similarity_top_k=top_k,
        use_async=False,
        llm=MockLLM(),
    )


def rerank(question: str, results: list, *, reranker=None, model: str | None = None) -> list:
    """Cross-encoder rerank a candidate list. Pass a cached ``reranker`` (kept for a process's
    lifetime, sized to a generous pool) to reuse it; otherwise a one-shot reranker is built from
    ``model`` sized to exactly this call's result count. Callers gate on their own "rerank enabled"
    flag before calling; this only no-ops on an empty list."""
    if not results:
        return results
    from llama_index.core.schema import QueryBundle

    if reranker is None:
        from llama_index.core.postprocessor import SentenceTransformerRerank
        reranker = SentenceTransformerRerank(model=model, top_n=len(results))
    return reranker.postprocess_nodes(results, QueryBundle(query_str=question))
