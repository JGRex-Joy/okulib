from sentence_transformers import SentenceTransformer
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from config import EMBEDDING_MODEL

model = SentenceTransformer(EMBEDDING_MODEL)

def embed_text(texts):
    """
    texts: list[str]
    возвращает numpy array shape=(len(texts), dim)
    """
    if isinstance(texts, str):
        texts = [texts]  # на всякий случай, если передали одну строку
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
