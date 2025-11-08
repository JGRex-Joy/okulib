from services.app_services.search import search
from services.app_services.llm_client import llm_generate

def answer_query(book_name: str, query: str, top_k: int = 5):
    retrieved_chunks = search(book_name, query, top_k)
    
    context = "\n\n".join([c["text"] for c in retrieved_chunks])
    
    prompt = f"""
        Ты помощник. Отвечай строго по указанному контексту. 
        Если ответа в тексте нет — скажи "Ответ не найден в тексте".

        Контекст:
        {context}

        Вопрос: {query}
        Ответ:
    """.strip()
    
    return llm_generate(prompt)