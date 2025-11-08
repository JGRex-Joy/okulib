import os
import json
from pathlib import Path
import faiss
import sys
import numpy as np
import uuid

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from services.embedder import embed_text
from services.data_services.chunker import chunk_book
import config

RAW_DATA_DIR = config.RAW_DATA_DIR
DATA_DIR = config.DATA_DIR

def prepare_book_data(book_path: str):
    book_path = Path(book_path)
    
    if not book_path.exists():
        raise FileNotFoundError(f"File {book_path} is not founded")
    
    book_name = book_path.stem
    book_data_dir = Path(DATA_DIR) / book_name
    os.makedirs(book_data_dir, exist_ok=True)
    
    # Разбиваем книгу на чанки
    chunks = chunk_book(book_path)
    
    # Создаём список объектов с уникальным id
    chunk_objects = []
    for i, text in enumerate(chunks):
        chunk_id = str(i)  # или uuid.uuid4() для глобально уникальных
        chunk_objects.append({"id": chunk_id, "text": text})
    
    # Сохраняем JSON с id + текст
    with open(book_data_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunk_objects, f, ensure_ascii=False, indent=2)
    
    # Создаём FAISS векторы для каждого чанка
    texts = [c["text"] for c in chunk_objects]
    vectors = embed_text(texts)
    vectors = np.array(vectors).astype("float32")
    
    dim = vectors.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 200
    
    # Добавляем каждый вектор отдельно, чтобы id соответствовали JSON
    index.add(vectors)
    
    # Сохраняем FAISS индекс
    faiss.write_index(index, str(book_data_dir / "index.faiss"))
    
    print(f'Index is created for {book_name}: {len(chunks)} chunks. Dim size is {dim}')
    print(f'Files are saved in {book_data_dir}')

# Пример запуска для нескольких книг
book_list = ["manas.txt", "semetey.txt", "syngan_kylych.txt", "uzak_jol.txt"]
for book in book_list:
    prepare_book_data(Path(RAW_DATA_DIR) / book)
