"""
ingest.py
Reads every .txt file in data/legal_docs/, splits it into overlapping chunks,
embeds each chunk with a domain-tuned Indian-legal embedding model
(InLegalBERT), and stores the vectors in a local ChromaDB collection.

Run this once at setup, and again any time you add/change files in
data/legal_docs/.

Usage:
    python ingest.py
"""

import sys
import chromadb
from sentence_transformers import SentenceTransformer

import config


def load_embedding_model():
    """Load InLegalBERT; fall back to a general multilingual model if the
    domain model can't be loaded (e.g. no internet on first run)."""
    try:
        print(f"Loading embedding model: {config.EMBEDDING_MODEL_PRIMARY}")
        # InLegalBERT is a plain HF encoder (not a sentence-transformers model
        # by default), so we wrap it with mean pooling via SentenceTransformer's
        # generic Transformer + Pooling modules.
        from sentence_transformers import models

        word_embedding_model = models.Transformer(config.EMBEDDING_MODEL_PRIMARY, max_seq_length=256)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode="mean")
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        print("InLegalBERT loaded successfully.")
        return model
    except Exception as e:
        print(f"Could not load {config.EMBEDDING_MODEL_PRIMARY} ({e}).")
        print(f"Falling back to {config.EMBEDDING_MODEL_FALLBACK}")
        return SentenceTransformer(config.EMBEDDING_MODEL_FALLBACK)


def chunk_text(text: str, chunk_size: int, overlap: int):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def main():
    if not config.LEGAL_DOCS_DIR.exists() or not any(config.LEGAL_DOCS_DIR.glob("*.txt")):
        print(f"No .txt files found in {config.LEGAL_DOCS_DIR}. Add legal reference "
              f"documents there first.")
        sys.exit(1)

    embed_model = load_embedding_model()

    client = chromadb.PersistentClient(path=str(config.VECTORSTORE_DIR))
    # Fresh collection each run, so stale/removed docs don't linger
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(config.COLLECTION_NAME)

    ids, docs, metadatas = [], [], []
    txt_files = sorted(config.LEGAL_DOCS_DIR.glob("*.txt"))
    print(f"Found {len(txt_files)} legal document(s). Chunking + embedding...")

    for file_path in txt_files:
        text = file_path.read_text(encoding="utf-8")
        chunks = chunk_text(text, config.CHUNK_SIZE_WORDS, config.CHUNK_OVERLAP_WORDS)
        for i, chunk in enumerate(chunks):
            ids.append(f"{file_path.stem}__chunk{i}")
            docs.append(chunk)
            metadatas.append({"source": file_path.name, "chunk_index": i})
        print(f"  {file_path.name}: {len(chunks)} chunk(s)")

    print(f"Embedding {len(docs)} chunks total (this can take a minute on CPU)...")
    embeddings = embed_model.encode(docs, show_progress_bar=True, convert_to_numpy=True).tolist()

    collection.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)

    print(f"\nDone. Vector store saved to: {config.VECTORSTORE_DIR}")
    print(f"Collection '{config.COLLECTION_NAME}' has {collection.count()} chunks.")


if __name__ == "__main__":
    main()
