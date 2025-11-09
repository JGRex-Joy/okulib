from fastapi import APIRouter
from services.app_services.llm import answer_query

router = APIRouter()

@router.get("/ask")
async def ask(book_name: str, query: str):
    answer = answer_query(book_name, query)
    return answer