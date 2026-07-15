"""Core layer -- reusable primitives that depend on nothing else in the package: the retrieval
engine (embed/Chroma/BM25/fusion/rerank), token counting, the L1 override loader, and ingest-time
store sync. Everything above (retrieval, ingest, eval, repo plug-in) builds on these."""
