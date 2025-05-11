from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.documents import Document
from typing import List
import numpy as np
import faiss
from transformers import AutoModel, AutoTokenizer, pipeline
import torch
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from text_splitter import split_text_into_chunks
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings


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

class ResponseModel(BaseModel):
    answer: str
    context: str  # Add context field to response model

# Initialize components once at startup
@app.on_event("startup")
async def startup_event():
    global qa_pipeline, vector_store
    
    # Load embedding model
    path = 'google/muril-large-cased'
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path, output_hidden_states=True)
    # embedding_fn = MurilEmbeddings(model, tokenizer)
    embedding_fn = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # Create vector store
    chunks = split_text_into_chunks("bangladesh_issue.txt")
    documents = [Document(page_content=chunk) for chunk in chunks]
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embedding_fn
    )

    # Initialize QA pipeline
    qa_pipeline = pipeline(
        "question-answering",
        model="deepset/xlm-roberta-large-squad2",
        tokenizer="deepset/xlm-roberta-large-squad2",
    )

class MurilEmbeddings(Embeddings):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[0, 0].cpu().numpy()
            embeddings.append(cls_embedding.tolist())
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

def get_answer(query: str) -> dict:
    # Retrieve relevant documents
    relevant_docs = vector_store.similarity_search(query, k=300)
    
    # Prepare context
    context = "\n".join([doc.page_content for doc in relevant_docs])

    # Get answer from QA model
    result = qa_pipeline(
        question=query,
        context=context,
        max_answer_len=500,
        handle_impossible_answer=True
    )
    
    return {
        "answer": result['answer'] if result['score'] > 0.13 else "दस्तावेज़ में उत्तर नहीं मिला",
        "context": context
    }

@app.post("/ask", response_model=ResponseModel)
async def ask_question(request: QueryRequest):
    try:
        response = get_answer(request.question)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)