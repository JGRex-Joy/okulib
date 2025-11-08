from pathlib import Path
import json
import faiss
import numpy as np
from config import DATA_DIR
from services.embedder import embed_text

def search(book_name: str, query: str, top_k : int = 5):
    book_dir = Path(DATA_DIR) / book_name
    index_path = book_dir / "index.faiss"
    chunks_path = book_dir / "chunks.json"
    
    index = faiss.read_index(str(index_path))
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    query_vec = embed_text(query).reshape(1, -1).astype("float32")
    distances, indices = index.search(query_vec, top_k)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        chunk = chunks[idx]
        results.append({"id": chunk["id"], "text": chunk["text"], "score": {float(dist)}})
        
    return results