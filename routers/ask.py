from fastapi import APIRouter
from services.app_services.llm import answer_query

router = APIRouter()

@router.get("/ask")
async def ask(book_name: str, query: str):
    result = answer_query(book_name, query)
    
    answer = result.get("answer")
    chunks = result.get("chunks", [])
    
    print("\n [INFO] Retrieved chunks: ")
    for i, c in enumerate(chunks, 1):
        print(f"{i}. Score={c['score']}, Text={c['text'][:100]}...")
        
    return answer
    