"""
bid_memory.py
-------------
Ingests past bid documents into a ChromaDB vector store using Voyage AI embeddings.
Provides semantic retrieval: given a query, returns the most relevant chunks
from your ingested documents.

Usage (as a script):
    python3 bid_memory.py

Usage (imported by another module):
    from bid_memory import ingest_bids, find_relevant_chunks

Dependencies:
    pip3 install voyageai chromadb python-dotenv --break-system-packages

Environment variables required in .env:
    VOYAGE_API_KEY=your_voyage_key
"""

import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import voyageai
import chromadb
from dotenv import load_dotenv

load_dotenv()

# ── Clients ────────────────────────────────────────────────────────────────────
voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

chroma_client = chromadb.Client()

# Collections are created per client_id on demand — see _get_collection()
_collections: dict[str, object] = {}
_COLLECTIONS_LOCK = threading.Lock()

# ── Concurrency controls ───────────────────────────────────────────────────────
_EMBED_SEMAPHORE = threading.Semaphore(3)
_CHROMA_LOCK     = threading.Lock()


def _get_collection(client_id: str):
    """Return (and cache) a ChromaDB collection scoped to this client."""
    with _COLLECTIONS_LOCK:
        if client_id not in _collections:
            _collections[client_id] = chroma_client.get_or_create_collection(
                name=f"{client_id}_bids",
                metadata={"hnsw:space": "cosine"},
            )
        return _collections[client_id]


# ── FUNCTION 1: chunk_text ─────────────────────────────────────────────────────

def chunk_text(text: str, size: int = 700) -> list[str]:
    """Split text into overlapping sentence-boundary chunks."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = []
    current_size = 0

    for sentence in sentences:
        slen = len(sentence)
        if current_size + slen > size and current:
            chunk_str = ' '.join(current)
            if chunk_str.strip():
                chunks.append(chunk_str.strip())
            # Keep last 2 sentences as overlap
            current = current[-2:] if len(current) > 2 else current
            current_size = sum(len(s) for s in current)
        current.append(sentence)
        current_size += slen + 1

    if current:
        chunk_str = ' '.join(current)
        if chunk_str.strip():
            chunks.append(chunk_str.strip())

    return chunks


# ── FUNCTION 2: helpers used by ingest_one_doc ─────────────────────────────────

def _read_file(filepath: str) -> str:
    """Extract text from a PDF or plain-text file."""
    if filepath.lower().endswith(".pdf"):
        import fitz
        pages = []
        with fitz.open(filepath) as doc:
            for page in doc:
                t = page.get_text()
                if t.strip():
                    pages.append(t)
        return "\n\n".join(pages)
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()


def _embed_chunks(chunks: list[str]) -> list:
    """Call Voyage AI to embed a list of text chunks. Rate-limited by caller."""
    result = voyage_client.embed(
        texts=chunks,
        model="voyage-3-lite",
        input_type="document"
    )
    return result.embeddings


def _store_chunks(filepath: str, chunks: list[str], embeddings: list, client_id: str) -> None:
    """Write chunks + embeddings to the client's ChromaDB collection under a lock."""
    source_name = os.path.basename(filepath)
    ids        = [f"{source_name}_chunk_{i}" for i in range(len(chunks))]
    metadatas  = [{"source": source_name, "chunk_index": i, "total_chunks": len(chunks)}
                  for i in range(len(chunks))]

    with _CHROMA_LOCK:
        _get_collection(client_id).add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )


# ── FUNCTION 3: ingest_one_doc ─────────────────────────────────────────────────

def _ingest_one_doc(filepath: str, client_id: str) -> tuple:
    """
    Read → chunk → embed → store for a single file.
    Runs inside a thread. Returns (filepath, num_chunks, error_or_None).
    Errors are returned, not raised, so the thread pool can report them.
    """
    with _EMBED_SEMAPHORE:
        try:
            if not os.path.exists(filepath):
                return (filepath, 0, f"File not found: {filepath}")
            text       = _read_file(filepath)
            chunks     = chunk_text(text)
            embeddings = _embed_chunks(chunks)
            _store_chunks(filepath, chunks, embeddings, client_id)
            return (filepath, len(chunks), None)
        except Exception as e:
            return (filepath, 0, str(e))


# ── FUNCTION 4: ingest_bids ────────────────────────────────────────────────────

def ingest_bids(filepaths: list[str], client_id: str = "default", progress_callback=None) -> dict:
    """
    Ingest multiple bid documents in parallel into a client-scoped ChromaDB collection.

    Args:
        filepaths:         List of file paths (.pdf or .txt)
        client_id:         Scopes the ChromaDB collection — each client's bids are separate.
        progress_callback: Optional function(completed, total, filename) for UI progress updates.

    Returns:
        {"succeeded": int, "failed": [(path, error), ...], "total_chunks": int}
    """
    total        = len(filepaths)
    succeeded    = 0
    failed       = []
    total_chunks = 0
    completed    = 0

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_path = {
            executor.submit(_ingest_one_doc, fp, client_id): fp
            for fp in filepaths
        }

        for future in as_completed(future_to_path):
            filepath = future_to_path[future]
            completed += 1
            try:
                path, num_chunks, error = future.result()
                if error:
                    failed.append((path, error))
                    print(f"  ✗ {os.path.basename(path)}: {error}")
                else:
                    succeeded += 1
                    total_chunks += num_chunks
                    print(f"  ✓ {os.path.basename(path)} — {num_chunks} chunks")
            except Exception as e:
                failed.append((filepath, str(e)))

            if progress_callback:
                progress_callback(completed, total, os.path.basename(filepath))

    print(f"\nIngestion complete. {succeeded}/{total} documents. "
          f"{total_chunks} total chunks. {len(failed)} failed.")
    return {"succeeded": succeeded, "failed": failed, "total_chunks": total_chunks}


# ── FUNCTION 5: find_relevant_chunks ──────────────────────────────────────────

def find_relevant_chunks(query: str, client_id: str = "default", n: int = 3) -> list[dict]:
    """
    Embed a query and return the n most similar chunks from this client's ChromaDB collection.
    """
    query_result = voyage_client.embed(
        texts=[query],
        model="voyage-3-lite",
        input_type="query"
    )
    query_embedding = query_result.embeddings[0]

    results = _get_collection(client_id).query(
        query_embeddings=[query_embedding],
        n_results=n,
        include=["documents", "metadatas", "distances"]
    )

    return [
        {
            "text":        results["documents"][0][i],
            "source":      results["metadatas"][0][i]["source"],
            "chunk_index": results["metadatas"][0][i]["chunk_index"],
            "distance":    round(results["distances"][0][i], 4),
        }
        for i in range(len(results["documents"][0]))
    ]


# ── MAIN: smoke test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = ingest_bids(["past_bids/past_bid_1.txt",
                           "past_bids/past_bid_2.txt",
                           "past_bids/past_bid_3.txt"])
    print(results)

    for query in ["safety management", "Passive House certification", "BIM coordination"]:
        chunks = find_relevant_chunks(query, n=2)
        print(f"\n'{query}':")
        for c in chunks:
            print(f"  [{c['source']}] dist={c['distance']} — {c['text'][:120]}…")
