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
import json

from huggingface_hub import login
login()


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
    top_k: int

class ResponseModel(BaseModel):
    answer: str
    context: list  # Returning context documents

# Global variables
chroma_client = None
collection = None
model = None
tokenizer = None

# Configuration
CHROMA_DB_PATH = "./chroma_db"
DATA_DIR = "/home/akash/dataset/loksabha_rqa/txt_files"
COLLECTION_NAME = "loksabha_debates"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"



@app.on_event("startup")
async def startup_event():
    global chroma_client, collection, model, tokenizer
    
    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    print(f"Chromadb initialized at {CHROMA_DB_PATH}")
    
    # Initialize embedding function
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    
    # Check if collection exists
    try:
        collection = chroma_client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=sentence_transformer_ef
        )
        print("Loaded existing collection")
    except ValueError:
        print("Collection not found, initializing new database...")
        chroma_client, collection = initialize_database()

    # Initialize LLM model and tokenizer
    print("Initializing language model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    print("Model initialized!")


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
        # Retrieve relevant context from ChromaDB
        docs = collection.query(
            query_texts=[query],
            n_results=top_k
        )
        context_docs = docs['documents'][0]
        
        # Create RAG prompt
        prompt_template = '''You are a large language model that answers questions based on the provided context. Please follow these steps:

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
            docs="\n".join(context_docs),
            query=query
        )
        
        # Format for the model
        messages = [{"role": "user", "content": formatted_prompt}]
        input_prompt = create_prompt_with_chat_format(messages, add_bos=False)
        
        # Generate response
        inputs = tokenizer(input_prompt, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Process response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_response.split("Answer:")[-1].strip()
        
        return {"answer": answer, "context": context_docs}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


@app.post("/ask", response_model=ResponseModel)
async def ask_question(request: QueryRequest):
    try:
        st_time = time.time()
        result = generate_answer(request.question, request.top_k)
        print(f"Total time taken: {time.time() - st_time:.2f} seconds")
        return {"answer": result["answer"], "context": result["context"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)