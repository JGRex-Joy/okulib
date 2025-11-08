from fastapi import FastAPI
from services.app_services.llm import answer_query

app = FastAPI()

@app.get("/ask")
def ask(book_name: str, query: str):
    answer = answer_query(book_name, query)
    return answer