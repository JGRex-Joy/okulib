from sentence_transformers import SentenceTransformer
from pathlib import Path
from config import EMBEDDING_MODEL

model = SentenceTransformer(EMBEDDING_MODEL)

def embed_text(texts):
    if isinstance(texts, str):
        texts = [texts]  
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
