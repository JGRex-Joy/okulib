from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent  

RAW_DATA_DIR = BASE_DIR / "raw_data"
DATA_DIR = BASE_DIR / "data"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
API_KEY = "AIzaSyCBIv1XuzNquouPBB4XEnpCE5xQK3O-T9Q"
