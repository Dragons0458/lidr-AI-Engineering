"""Vector store — PostgreSQL + pgvector persistence (Session 8).

``models`` defines ``DocumentRow`` and ``ChunkRow``; semantic search lives in
``POST /embeddings/search``. HNSW/IVFFlat indexes are deferred to the live session.
"""
