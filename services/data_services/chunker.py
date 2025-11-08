from config import CHUNK_SIZE, CHUNK_OVERLAP
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_book(file_path: str):
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