from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import chromadb
from text_splitter import split_text_into_chunks
from chromadb.utils import embedding_functions
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
import glob
import os
import spacy
import logging
import requests
from bert_score import score
import matplotlib.pyplot as plt
from fastapi.responses import FileResponse



# from huggingface_hub import login
# login()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()
# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API configurations
FIRST_API_URL = "http://10.4.16.40:4554/search/api/word_search/"
CSRF_TOKEN = "v4IlXw6ZCxiO0Ogno7lggZdIAoTHNaAtleVazecfGnQ2057aQ90vSTrlnp0qpUCW"
nlp = None
class QueryRequest(BaseModel):
    question: str
    top_k: int

class ResponseModel(BaseModel):
    answer: str
    meta_context: list  # Returning context documents
    context: list

# Global variables
chroma_client = None
collection = None
model = None
tokenizer = None

#Get context from API
def get_context_from_api(keyword: str, top_k: int) -> list:
    """Get context from the external API using the extracted keyword"""
    headers = {
        "accept": "application/json",
        "X-CSRFTOKEN": CSRF_TOKEN
    }
    params = {"q": keyword}
    
    try:
        response = requests.get(
            FIRST_API_URL,
            headers=headers,
            params=params,
            timeout=10  # 10 seconds timeout
        )
        response.raise_for_status()
        context_data = response.json()
        
        # Handle different response structures
        if isinstance(context_data, list):
            return context_data[:top_k]
        elif isinstance(context_data, dict):
            # Try to find likely context fields
            for field in ["results", "documents", "context", "data"]:
                if field in context_data and isinstance(context_data[field], list):
                    return context_data[field][:top_k]
            # If no list found, return values as list
            return list(context_data.values())[:top_k]
        else:
            return [str(context_data)]
            
    except Exception as e:
        logger.error(f"API call failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Context API error: {str(e)}"
        )

#Extract keyword
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

@app.on_event("startup")
async def startup_event():
    global chroma_client, collection, model, tokenizer,nlp

    #Initialize spaCy for keyword extraction
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("spaCy model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading spaCy model: {e}")
        # If model not found, download it
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
    
    # Initialize ChromaDB
    chroma_client = chromadb.Client()
    print("Chromadb initialized!")
    # Initialize embedding function (multilingual)
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name= "intfloat/e5-base-v2"#"intfloat/multilingual-e5-base"
    )
    print("setence transformer initialized.")
    # Create or get collection
    print(">>collection initalizing")
    collection = chroma_client.get_or_create_collection(
        name="ncert_class10_history",
        embedding_function=sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"}
    )
    print("collection intialized!")
    # Directory containing .txt files
    data_dir = "/home/akash/ws/dataset/icvgip/ISE/easy_ocr/lehs101"
    
    # Read and split all text files into chunks
    all_chunks = []
    metadatas = []
    ids = []
    idx = 0
    for filepath in glob.glob(os.path.join(data_dir, "*.txt")):
        chunks = split_text_into_chunks(filepath)
        for chunk in chunks:
            all_chunks.append(chunk)
            # store metadata: source file and header or chunk index
            metadatas.append({"source": os.path.basename(filepath), "header": chunk[:50]})
            ids.append(f"chunk_{idx}")
            idx += 1
    
    # Ingest into Chroma
    collection.add(
        documents=all_chunks,
        metadatas=metadatas,
        ids=ids
    )
    
    # Initialize model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"#"meta-llama/Meta-Llama-3-8B-Instruct"#"CoRover/BharatGPT-3B-Indic"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map = "auto"
    )#.to(device)


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


def generate_answer(query: str, top_k: int) -> dict:
    try:
        docs = collection.query(
            query_texts=[query],
            n_results=top_k
        )
        context_docs = docs['documents'][0]
        
        prompt_template = '''You are a large language model that answers questions based on the provided context. 
                                Please follow these steps:

                                    1. **Read the Question**  
                                    - Carefully read and understand the given question.

                                    2. **Read the Context**  
                                    - Carefully read and understand the context provided below.

                                    3. **Generate the Answer**  
                                    - Using the context, produce a clear and detailed answer to the question.  
                                    - Ensure the answer is direct, easy to understand, and fact-based.

                                    ### Now, using the following context and question, provide your answer:

                                    **Context**:  
                                    {docs}

                                    **Question**:  
                                    {query}

                                    Answer:''' 
               
        formatted_prompt = prompt_template.format(
            docs="\n".join(context_docs), #
            query=query
        )
        
        messages = [{"role": "user", "content": formatted_prompt}]
        input_prompt = create_prompt_with_chat_format(messages, add_bos=False)
        
        inputs = tokenizer(input_prompt, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=1024,
                do_sample=False
            )
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_response.split("Answer:")[-1].strip()
        
        return {"answer": answer, "context": context_docs}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


@app.post("/ask", response_model=ResponseModel)
async def ask_question(request: QueryRequest):
    try:
        st_time = time.time()
        # Step 1: Extract keyword
        keyword = extract_keyword(request.question)
        print("The extracted keyword is:", keyword)
        
        # Step 2: Get meta context from API (e.g., retrieved documents or related content)
        meta_context = get_context_from_api(keyword=keyword, top_k=request.top_k)
        
        # Step 3: Generate answer along with associated contexts
        result = generate_answer(request.question, request.top_k)
        answer = result["answer"]
        contexts = result["context"]
        
        print(f"Total time taken: {time.time() - st_time}")

        # Step 4: Measure BERT score for the answer against each context
        bert_scores = []
        for context in contexts:
            # Compute BERT score between the answer and the current context.
            P, R, F1 = score([answer], [context], lang="en", verbose=False)
            bert_scores.append({
                "context": context,
                "P": float(P),
                "R": float(R),
                "F1": float(F1)
            })
        vect_top3 = sorted(bert_scores, key=lambda x: x["F1"], reverse=False)#[:3]
        vect_f1 = [item["F1"] for item in vect_top3]

        search_f1 = []
        for ctx in meta_context[2][:3]:
            P, R, F1 = score([answer], [ctx["full_text"]], lang="en", verbose=False)
            search_f1.append(float(F1))
        search_f1 = sorted(search_f1,  reverse=False)
        # search_f1    = [item["F1"] for item in search_top3]
        
        # Step 5: Rank contexts based on the F1 score in descending order
        ranked_contexts = sorted(bert_scores, key=lambda x: x["F1"], reverse=True)
        print("Ranked contexts based on F1 score:")
        for idx, item in enumerate(ranked_contexts):
            print(f"Rank {idx+1}: F1 score = {item['F1']}")

        #Step-6
        labels = [f"#{i+1}" for i in range(3)]
        x = list(range(3))
        width = 0.35

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([xi - width/2 for xi in x], vect_f1, width, label="VectorDB")
        ax.bar([xi + width/2 for xi in x], search_f1, width, label="SearchAPI")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Top‑3 contexts")
        ax.set_ylabel("BERT‑F1 Score")
        ax.set_title("Answer vs. Retrieved Contexts")
        ax.legend()
        plt.tight_layout()

        # 6) Save figure
        plot_path = "/tmp/bert_score_comparison.png"
        fig.savefig(plot_path)
        plt.close(fig)

        
        
        # Return answer along with meta context and ranked contexts with their BERT scores
        return {"answer": answer, "meta_context": meta_context, "context": ranked_contexts}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/plots/bert_comparison")
async def get_bert_plot():
    return FileResponse("/tmp/bert_score_comparison.png", media_type="image/png")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="10.10.16.13", port=8000)

