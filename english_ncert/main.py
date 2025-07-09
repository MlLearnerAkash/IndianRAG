from fastapi import FastAPI, HTTPException
import requests
import spacy
from pydantic import BaseModel

app = FastAPI()

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# API configurations
FIRST_API_URL = "http://10.4.16.40:4554/search/api/word_search/"
SECOND_API_URL = "http://localhost:8001/ask"
CSRF_TOKEN = "v4IlXw6ZCxiO0Ogno7lggZdIAoTHNaAtleVazecfGnQ2057aQ90vSTrlnp0qpUCW"

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

def extract_keyword(question: str) -> str:
    """Extract the most relevant noun/proper noun from the question"""
    doc = nlp(question)
    
    # Prefer nouns/proper nouns that aren't stop words
    keywords = [token.text for token in doc 
               if token.pos_ in ("NOUN", "PROPN") and not token.is_stop]
    
    # Fallback to nouns/proper nouns including stop words
    if not keywords:
        keywords = [token.text for token in doc if token.pos_ in ("NOUN", "PROPN")]
    
    # Final fallback: use the first content word
    if not keywords:
        content_words = [token.text for token in doc if not token.is_stop]
        return content_words[0] if content_words else question.split()[0]
    
    return keywords[0]

@app.post("/integrated_ask")
async def integrated_ask(request: QueryRequest):
    # Step 1: Extract keyword
    keyword = extract_keyword(request.question)
    
    # Step 2: Call first API to get context
    headers_api1 = {
        "accept": "application/json",
        "X-CSRFTOKEN": CSRF_TOKEN
    }
    try:
        response1 = requests.get(
            FIRST_API_URL,
            params={"q": keyword},
            headers=headers_api1,
            timeout=1000
        )
        response1.raise_for_status()
        context_data = response1.json()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Keyword search API failed: {str(e)}"
        )

    # Step 3: Call second API to get answer
    headers_api2 = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    payload = {
        "question": request.question,
        "context": context_data,
        "top_k": request.top_k
    }
    try:
        response2 = requests.post(
            SECOND_API_URL,
            json=payload,
            headers=headers_api2,
            timeout=10000
        )
        response2.raise_for_status()
        answer_data = response2.json()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Answer generation API failed: {str(e)}"
        )

    # Step 4: Return combined response
    return {
        "extracted_keyword": keyword,
        "context": context_data,
        "answer": answer_data
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)