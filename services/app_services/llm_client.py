from google import genai
from config import API_KEY

client = genai.Client(api_key=API_KEY)

def llm_generate(prompt: str) -> str:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    
    return response.text.strip()