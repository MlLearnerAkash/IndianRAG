#@Author: Akash Manna, IIT-Hyd
#@Date:12/05/25
#@ref: https://www.analyticsvidhya.com/blog/2024/11/rag-pipeline-for-hindi-documents/

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import httpx
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import requests
import re
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    keyword: str

class ResponseModel(BaseModel):
    answer: str
    context_pages: List[str]

# Global variables for LLM
model = None
tokenizer = None
SEARCH_API_URL = "http://10.4.16.40:6556/search/api/word_search/"

@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    
    # Initialize LLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "sarvamai/sarvam-1"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,  
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16
    )

def create_prompt(context: str, question: str) -> str:

    # Format prompt
    prompt_template = '''ਤੁਸੀਂ ਇੱਕ ਵੱਡਾ ਭਾਸ਼ਾਈ ਮਾਡਲ ਹੋ ਜੋ ਦਿੱਤੇ ਗਏ ਸੰਦਰਭ ਦੇ ਆਧਾਰ 'ਤੇ ਸਵਾਲਾਂ ਦੇ ਉਤਰ ਦਿੰਦਾ ਹੈ। ਹੇਠਾਂ ਦਿੱਤੇ ਗਿਆ ਹੁਕਮਾਨੁਸਾਰ ਢਾਂਚਾ ਫਾਲੋ ਕਰੋ:

                    1. **ਸਵਾਲ ਪੜ੍ਹੋ**:
                        - ਦਿੱਤਾ ਗਿਆ ਸਵਾਲ ਧਿਆਨ ਨਾਲ ਪੜ੍ਹੋ ਅਤੇ ਸਮਝੋ।

                    2. **ਸੰਦਰਭ ਪੜ੍ਹੋ**:
                        - ਹੇਠਾਂ ਦਿੱਤਾ ਗਿਆ ਸੰਦਰਭ ਧਿਆਨ ਨਾਲ ਪੜ੍ਹੋ ਅਤੇ ਸਮਝੋ।

                    3. **ਜਾਣਕਾਰੀ ਤਿਆਰ ਕਰੋ**:
                        - ਸੰਦਰਭ ਦੀ ਵਰਤੋਂ ਕਰਦਿਆਂ, ਸਵਾਲ ਦਾ ਵਿਸਥਾਰਪੂਰਕ ਅਤੇ ਸਾਫ਼ ਉਤਰ ਤਿਆਰ ਕਰੋ।
                        - ਇਹ ਯਕੀਨੀ ਬਣਾਓ ਕਿ ਉਤਰ ਸਿੱਧਾ, ਸਮਝਣ ਵਿੱਚ ਆਸਾਨ ਅਤੇ ਤੱਥਾਂ 'ਤੇ ਆਧਾਰਿਤ ਹੋਵੇ।

                    ### ਉਦਾਹਰਣ:

                    **ਸੰਦਰਭ**:
                        "ਨਵੀਂ ਦਿੱਲੀ ਭਾਰਤ ਦੀ ਰਾਜਧਾਨੀ ਹੈ ਅਤੇ ਇਹ ਦੇਸ਼ ਦਾ ਪ੍ਰਮੁੱਖ ਰਾਜਨੀਤਿਕ ਅਤੇ ਪ੍ਰਸ਼ਾਸਕੀ ਕੇਂਦਰ ਹੈ। ਇਹ ਸ਼ਹਿਰ ਇਤਿਹਾਸਕ ਸਮਾਰਕਾਂ, ਅਦਿਆਰਕਾਲੈਡਿਆ ਅਤੇ ਬਹੁਰੰਗੀ ਸਭਿਆਚਾਰ ਲਈ ਮੰਨਿਆ ਜਾਂਦਾ ਹੈ।"

                    **ਸਵਾਲ**:
                        "ਭਾਰਤ ਦੀ ਰਾਜਧਾਨੀ ਕੀ ਹੈ ਅਤੇ ਇਹ ਕਿਉਂ ਮਹੱਤਵਪੂਰਨ ਹੈ?"

                    **ਉਮੀਦ ਕੀਤਾ ਗਿਆ ਉਤਰ**:
                        "ਭਾਰਤ ਦੀ ਰਾਜਧਾਨੀ ਨਵੀਂ ਦਿੱਲੀ ਹੈ। ਇਹ ਦੇਸ਼ ਦਾ ਪ੍ਰਮੁੱਖ ਰਾਜਨੀਤਿਕ ਅਤੇ ਪ੍ਰਸ਼ਾਸਕੀ ਕੇਂਦਰ ਹੈ ਅਤੇ ਇਤਿਹਾਸਕ ਸਮਾਰਕਾਂ, ਅਦਿਆਰਕਾਲੈਡਿਆ ਅਤੇ ਬਹੁਰੰਗੀ ਸਭਿਆਚਾਰ ਲਈ ਮੰਨਿਆ ਜਾਂਦਾ ਹੈ।"

                    ### ਹੁਕਮ:

                    ਹੁਣ, ਦਿੱਤਾ ਗਿਆ ਸੰਦਰਭ ਅਤੇ ਸਵਾਲ ਵਰਤਦੇ ਹੋਏ ਉਤਰ ਦਿਓ:

                    **ਸੰਦਰਭ**:
                    {docs}

                    **ਸਵਾਲ**:
                    {query}

                    ਉਤਰ:'''    

    # Keep your existing prompt template here
    formatted_prompt = prompt_template.format(
        docs="\n".join(context),
        query=question
    )
    return formatted_prompt

async def fetch_keyword_context(keyword: str) -> List[str]:
    print(f"Searching for keyword: {keyword}")
    try:
        response = requests.get(
            SEARCH_API_URL,
            headers={
                "accept": "application/json",
                "X-CSRFTOKEN": "IoWOF580TbPZnGZrVHUWvp8nKyETwjmLfHh2RSb1cW3vY7ziKKvQM0F4QiKRvH36"
            },
            params={"q": keyword},
            timeout=1000.0,
            verify=False  # Add this if SSL verification is needed
        )

        # Check for HTTP errors
        response.raise_for_status()
        
        data = response.json()
        print(f"API response received: {data.keys()}")
        
        return [" ".join(page["full_text"]) for page in data.get("pages", [])]

    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=500, detail=f"Search API HTTP error: {str(e)}")
    
    except requests.exceptions.JSONDecodeError:
        print(f"Invalid JSON response: {response.text}")
        raise HTTPException(status_code=500, detail="Invalid JSON response from search API")
    
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search API connection error: {str(e)}")
    
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search API error: {str(e)}")

def generate_answer(prompt: str) -> str:
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        
        with torch.inference_mode():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=1024,
                do_sample=False
            )
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_response.split("ਉਤਰ:")[-1].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

def create_prompt_with_chat_format(messages, bos="<s>", eos="</s>", add_bos=True):
    formatted_text = ""
    for message in messages:
        if message["role"] == "system":
            formatted_text += "<|system|>\n" + message["content"] + "\n"
        elif message["role"] == "user":
            formatted_text += "<|user|>\n" + message["content"] + "\n"
        elif message["role"] == "assistant":
            formatted_text += "<|assistant|>\n" + message["content"].strip() + eos + "\n"
    formatted_text += "<|assistant|>\n"
    return bos + formatted_text if add_bos else formatted_text

@app.post("/ask", response_model=ResponseModel)
async def ask_question(request: QueryRequest):
    try:
        # Step 1: Get context from search API
        context_pages = await fetch_keyword_context(request.keyword)
        
        # Combine first 5 pages to avoid context overflow
        combined_context = " ".join(context_pages[:5])[:4000]  # Limit context size
        combined_context = re.sub(r'[\x00-\x1F\x7F]+', ' ', combined_context).strip()
        
        # Step 2: Create LLM prompt
        formatted_prompt = create_prompt(combined_context, request.question)

        # Prepare input
        messages = [{"role": "user", "content": formatted_prompt}]
        input_prompt = create_prompt_with_chat_format(messages, add_bos=False)

        # Step 3: Generate answer
        answer = generate_answer(input_prompt)
        
        return {
            "answer": answer,
            "context_pages": context_pages[:5]  # Return first 5 pages used
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)