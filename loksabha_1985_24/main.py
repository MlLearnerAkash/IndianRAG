from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import chromadb
from chromadb.utils import embedding_functions
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
import os



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


@app.on_event("startup")
async def startup_event():
    global chroma_client, collection, model, tokenizer
    
    # Initialize ChromaDB (load persistent client)
    chroma_db_path = "./chroma_db" # Path where the database is stored
    print(f"Loading ChromaDB from {chroma_db_path}...")
    chroma_client = chromadb.PersistentClient(path=chroma_db_path)
    print("ChromaDB client initialized!")

    # Initialize embedding function (multilingual) - MUST be the same as used during creation
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("Sentence transformer initialized.")

    # Get the existing collection
    print(">> Getting collection...")
    try:
        collection = chroma_client.get_collection(
            name="loksabha_debates",
            embedding_function=sentence_transformer_ef
        )
        print("Collection loaded successfully!")
    except Exception as e:
        print(f"Error loading collection: {e}")
        print("Please ensure the ChromaDB has been created using create_db.py.")
        exit(1) # Exit if collection cannot be loaded
    



def generate_answer(query: str, top_k: int) -> dict:
    try:
        docs = collection.query(
            query_texts=[query],
            n_results=top_k
        )
        context_docs = docs["documents"][0]
        
        # Placeholder for answer generation as language model is removed
        answer = f"This is a dummy answer for your question: '{query}'. The context retrieved is: {context_docs}"
        
        return {"answer": answer, "context": context_docs}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")


@app.post("/ask", response_model=ResponseModel)
async def ask_question(request: QueryRequest):
    try:
        st_time = time.time()
        result = generate_answer(request.question, request.top_k)
        print(f"Total time taken: {time.time() - st_time}")
        return {"answer": result["answer"], "context": result["context"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


