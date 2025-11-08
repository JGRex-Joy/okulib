from config import CHUNK_SIZE, CHUNK_OVERLAP
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

def chunk_book(file_path: str):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} is not founded")
    
    text = file_path.read_text(encoding="utf-8")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n", 
            "\n", 
            r"\. ", 
            r"\! ", 
            r"\? ", 
            ", ", 
            " "
        ]
    )
    
    chunks = splitter.split_text(text)
    return chunks