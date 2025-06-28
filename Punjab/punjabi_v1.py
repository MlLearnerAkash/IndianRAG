##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#@Author: Akash Manna, IIT-Hyd
#@Date:28/06/25
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
import os
import asyncio
from gensim.models import KeyedVectors
from transformers import pipeline
from langdetect import detect
from indian_languages import FASTTEXT_MODEL_MAP, FASTTEXT_CODE_TO_NAME



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

###################Punjab Search Engine ###############################################
SEARCH_API_URL = "http://10.4.16.40:6556/search/api/word_search/"
FASTTEXT_MODELS_DIR = "/home/akash/ws/fasttext_models/"


# Create the directory if it doesn't exist
os.makedirs(FASTTEXT_MODELS_DIR, exist_ok=True)

# Cache for loaded FastText models
_loaded_fasttext_models = {}


##Query expansion

def load_fasttext_model(lang_code: str):
    if lang_code in _loaded_fasttext_models:
        return _loaded_fasttext_models[lang_code]

    model_info = None
    for lang_name, info in FASTTEXT_MODEL_MAP.items():
        if info["code"] == lang_code:
            model_info = info
            break

    if not model_info:
        print(f"No FastText model information found for language code: {lang_code}")
        return None

    # Extract filename from the link
    model_filename_gz = model_info["link"].split("/")[-1]
    model_filename = model_filename_gz.replace(".gz", "")
    model_path = os.path.join(FASTTEXT_MODELS_DIR, model_filename)

    if not os.path.exists(model_path):
        print(f"FastText model for {lang_code} not found at {model_path}. Please download it first.")
        return None

    print(f"Loading FastText model for {lang_code} from {model_path}...")
    try:
        model = KeyedVectors.load_word2vec_format(model_path, binary=False)
        _loaded_fasttext_models[lang_code] = model
        print(f"FastText model for {lang_code} loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading FastText model for {lang_code}: {e}")
        return None

async def expand_query(query):
    return "expanded query"

# async def fetch_keyword_context(keyword: str) -> List[str]:
#     print(f"Searching for keyword: {keyword}")
#     try:
#         response = requests.get(
#             SEARCH_API_URL,
#             headers={
#                 "accept": "application/json",
#                 "X-CSRFTOKEN": "IoWOF580TbPZnGZrVHUWvp8nKyETwjmLfHh2RSb1cW3vY7ziKKvQM0F4QiKRvH36"
#             },
#             params={"q": keyword},
#             timeout=1000.0,
#             verify=False  # Add this if SSL verification is needed
#         )

#         # Check for HTTP errors
#         response.raise_for_status()
        
#         data = response.json()
#         print(f"API response received: {data.keys()}")
        
#         return [" ".join(page["full_text"]) for page in data.get("pages", [])]

#     except requests.exceptions.HTTPError as e:
#         print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
#         raise HTTPException(status_code=500, detail=f"Search API HTTP error: {str(e)}")
    
#     except requests.exceptions.JSONDecodeError:
#         print(f"Invalid JSON response: {response.text}")
#         raise HTTPException(status_code=500, detail="Invalid JSON response from search API")
    
#     except requests.exceptions.RequestException as e:
#         print(f"Request failed: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Search API connection error: {str(e)}")
    
#     except Exception as e:
#         print(f"Unexpected error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Search API error: {str(e)}")

async def fetch_keyword_context(keyword: str) -> List[str]:
    print(f"Searching for keyword: {keyword}")
    
    queries_to_search = [keyword] # Start with the original keyword

    # Detect language of the keyword
    detected_lang_code = None
    try:
        detected_lang_code = detect(keyword)
        print(f"Detected language for keyword \'{keyword}\': {detected_lang_code}")
    except Exception as e:
        print(f"Could not detect language for \'{keyword}\': {e}")

    # Check if the detected language is one of the supported Indian languages with a FastText model
    if detected_lang_code and detected_lang_code in FASTTEXT_CODE_TO_NAME:
        model = load_fasttext_model(detected_lang_code)
        if model:
            if keyword in model.key_to_index:
                print(f"Applying Gensim expansion for {FASTTEXT_CODE_TO_NAME[detected_lang_code]} keyword: {keyword}")
                try:
                    similar_words = [word for word, _ in model.most_similar(keyword, topn=5)]
                    queries_to_search.extend(similar_words)
                    print(f"Gensim expanded queries: {similar_words}")
                except Exception as e:
                    print(f"Error during Gensim expansion for {keyword}: {e}")
            else:
                print(f"Keyword \'{keyword}\' not found in {FASTTEXT_CODE_TO_NAME[detected_lang_code]} vocabulary. Skipping Gensim expansion.")
        else:
            print(f"FastText model for {FASTTEXT_CODE_TO_NAME[detected_lang_code]} not loaded. Skipping Gensim expansion.")
    else:
        print(f"Skipping Gensim expansion: Detected language ({detected_lang_code}) is not a supported Indian language with a FastText model.")

    # Remove duplicates and maintain order
    queries_to_search = list(dict.fromkeys(queries_to_search))
    print(f"Final queries to search: {queries_to_search}")

    all_results = []

    for query in queries_to_search:
        print(f"Fetching context for query: {query}")
        try:
            response = requests.get(
                SEARCH_API_URL,
                headers={
                    "accept": "application/json",
                    "X-CSRFTOKEN": "IoWOF580TbPZnGZrVHUWvp8nKyETwjmLfHh2RSb1cW3vY7ziKKvQM0F4QiKRvH36"
                },
                params={"q": query},
                timeout=1000.0,
                verify=False  # Add this if SSL verification is needed
            )

            # Check for HTTP errors
            response.raise_for_status()
            
            data = response.json()
            print(f"API response received for {query}: {data.keys()}")
            
            all_results.extend([" ".join(page["full_text"]) for page in data.get("pages", [])])

        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error for {query}: {e.response.status_code} - {e.response.text}")
        
        except requests.exceptions.JSONDecodeError:
            print(f"Invalid JSON response for {query}: {response.text}")
        
        except requests.exceptions.RequestException as e:
            print(f"Request failed for {query}: {str(e)}")
        
        except Exception as e:
            print(f"Unexpected error for {query}: {str(e)}")

    if not all_results:
        raise HTTPException(status_code=500, detail="No context found for any of the queries.")

    return all_results

    
####################PB Search Engine##########################################################


class ExpansionLLM:
    def __init__(self, model_name="ai4bharat/IndicBART-XXEN"):
        self.generator = pipeline(
            "text2text-generation",
            model=model_name,
            device=-1  # Use CPU
        )
        self.lang_map = {
            'en': 'en_XX',
            'hi': 'hi_IN',
            'ta': 'ta_IN',
            'te': 'te_IN',
            'kn': 'kn_IN',
            'ml': 'ml_IN',
            'mr': 'mr_IN',
            'bn': 'bn_IN',
            'gu': 'gu_IN',
            'pa': 'pa_IN',
            'or': 'or_IN'
        }
        self.instructions = {
            'en_XX': (
                "Expand and enrich this Indian context with cultural explanations: "
                "Original Context: '{context}'\n"
                "Related Question: '{question}'\n"
                "Add regional terms in brackets, explain local references, "
                "and provide necessary background. "
                "DO NOT ANSWER THE QUESTION - ONLY EXPAND THE CONTEXT.\n"
                "Expanded Context:"
            ),
            'hi_IN': (
                "सन्दर्भ को भारतीय सांस्कृतिक संदर्भ के साथ विस्तृत करें: "
                "मूल सन्दर्भ: '{context}'\n"
                "सम्बंधित प्रश्न: '{question}'\n"
                "स्थानीय शब्दों की व्याख्या करें, सांस्कृतिक पृष्ठभूमि जोड़ें। "
                "प्रश्न का उत्तर न दें - केवल सन्दर्भ का विस्तार करें।\n"
                "विस्तृत सन्दर्भ:"
            ),
            'pa_IN': (
                "ਭਾਰਤੀ ਸੱਭਿਆਚਾਰਕ ਪਿਛੋਕੜ ਨਾਲ ਸੰਦਰਭ ਦਾ ਵਿਸਤਾਰ ਕਰੋ: "
                "ਮੂਲ ਸੰਦਰਭ: '{context}'\n"
                "ਸਬੰਧਤ ਪ੍ਰਸ਼ਨ: '{question}'\n"
                "ਸਥਾਨਕ ਸ਼ਬਦਾਂ ਦੀ ਵਿਆਖਿਆ ਕਰੋ, ਸੱਭਿਆਚਾਰਕ ਪਿਛੋਕੜ ਸ਼ਾਮਲ ਕਰੋ। "
                "ਪ੍ਰਸ਼ਨ ਦਾ ਜਵਾਬ ਨਾ ਦਿਓ - ਕੇਵਲ ਸੰਦਰਭ ਦਾ ਵਿਸਤਾਰ ਕਰੋ।\n"
                "ਵਿਸਤ੍ਰਿਤ ਸੰਦਰਭ:"
            ),
            'default': (
                "Expand and enrich this Indian context with cultural explanations: "
                "Original Context: '{context}'\n"
                "Related Question: '{question}'\n"
                "Add regional terms in brackets, explain local references. "
                "DO NOT ANSWER THE QUESTION - ONLY EXPAND THE CONTEXT.\n"
                "Expanded Context:"
            )
        }
        # Chunk size in characters (conservative estimate for token limits)
        self.chunk_size = 1000  # Characters per chunk
        self.overlap_size = 0  # Character overlap between chunks

    async def expand(self, context: str, question: str) -> str:
        # Detect input language
        try:
            lang_code = detect(context) if context else detect(question)
            lang_token = self.lang_map.get(lang_code, 'en_XX')
        except:
            lang_token = 'en_XX'
            
        # Get instruction template
        instruction = self.instructions.get(lang_token, self.instructions['default'])
        
        # Process in chunks if context is large
        if len(context) > self.chunk_size:
            return await self._expand_in_chunks(context, question, lang_token, instruction)
        
        # Process single chunk
        return await self._expand_chunk(context, question, lang_token, instruction)

    async def _expand_in_chunks(self, context: str, question: str, 
                              lang_token: str, instruction: str) -> str:
        """Process long context using sliding window approach"""
        expanded_parts = []
        start = 0
        
        # Split context into overlapping chunks
        while start < len(context):
            end = start + self.chunk_size
            # Extend chunk to next sentence boundary if possible
            next_boundary = context.find('|', end) + 1
            if next_boundary > 0 and next_boundary < len(context):
                end = next_boundary
                
            chunk = context[start:end].strip()
            
            # Expand the chunk
            expanded = await self._expand_chunk(chunk, question, lang_token, instruction)
            expanded_parts.append(expanded)
            
            # Move to next chunk with overlap
            start = end - self.overlap_size
            if start < 0:
                start = 0
                
        # Combine expanded parts
        return " ".join(expanded_parts)

    async def _expand_chunk(self, context: str, question: str, 
                          lang_token: str, instruction: str) -> str:
        """Expand a single context chunk"""
        prompt = self._create_prompt(context, question, lang_token, instruction)
        return await self._generate_expansion(prompt, lang_token)

    def _create_prompt(self, context: str, question: str, 
                     lang_token: str, instruction: str) -> str:
        """Create prompt for a single chunk"""
        formatted_instruction = instruction.format(
            context=context,
            question=question
        )
        return f"[{lang_token}] {formatted_instruction}"

    async def _generate_expansion(self, prompt: str, lang_token: str) -> str:
        """Generate expansion for a single chunk"""
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, 
            lambda: self.generator(
                prompt,
                max_length=1024,
                num_beams=4,
                no_repeat_ngram_size=10,
                early_stopping=False
            )
        )
        expanded = result[0]['generated_text'].replace(f"[{lang_token}]", "").strip()
        return expanded

###################End: Expanded context########################################
@app.on_event("startup")
async def startup_event():
    global model, tokenizer, expansion_llm
    
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
    
    # Initialize expansion model
    expansion_llm = ExpansionLLM()

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
        
        # Combine first 3 pages to avoid context overflow
        combined_context = " ".join(context_pages[:3])[:3000]  # Reduced limit
        combined_context = re.sub(r'[\x00-\x1F\x7F]+', ' ', combined_context).strip()
        
        # Step 2: Expand context using async method
        print("Starting expanding context")
        print(">>>>>>>original context>>>>>", combined_context)
        expanded_context = await expansion_llm.expand(combined_context, request.question)
        print("?????Expaned context>>>>>>", expanded_context)
        print("Finished expanding context")
        
        # Step 3: Create LLM prompt
        formatted_prompt = create_prompt(expanded_context, request.question)

        # Prepare input
        messages = [{"role": "user", "content": formatted_prompt}]
        input_prompt = create_prompt_with_chat_format(messages, add_bos=False)

        # Step 4: Generate answer
        answer = generate_answer(input_prompt)
        
        return {
            "answer": answer,
            "context_pages": context_pages[:3]  # Return first 3 pages used
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)