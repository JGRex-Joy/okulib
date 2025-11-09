from services.app_services.search import search
from services.app_services.llm_client import llm_generate

def answer_query(book_name: str, query: str, top_k: int = 7):
    retrieved_chunks = search(book_name, query, top_k)
    
    context = "\n\n".join([c["text"] for c in retrieved_chunks])
    
    prompt = f"""
        Сен — акылдуу жардамчысың, адистешкен кыргыз адабияты боюнча суроолорго жооп берүүдө.

        Колдонуучунун суроосу төмөндө берилет.
        Сенде ошол китептин айрым бөлүктөрүнөн (чанктардан) алынган контекст бар.

        Тапшырмаң:
        1. Суроого жооп бер, бирок **контексттеги маалыматка гана таян**.
        2. Эгер контекстте жооп жок болсо, **“Бул суроого так жооп контекстте жок”** деп айт.
        3. Жоопту кыргыз тилинде түшүнүктүү, табигый жана адабий стилде бер.
        4. Жооп кыска, бирок так болсун (3–5 сүйлөм жетиштүү)
        
        **Контекст:**
        {context}

        **Суроо:**
        {query}

        **Жооп:**

    """.strip()
    
    return llm_generate(prompt)