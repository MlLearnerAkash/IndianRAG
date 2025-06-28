# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware
# from langchain_core.documents import Document
# from typing import List
# import numpy as np
# import faiss
# from transformers import AutoModel, AutoTokenizer, pipeline
# import torch
# from langchain_community.vectorstores import FAISS
# from langchain_core.embeddings import Embeddings
# from text_splitter import split_text_into_chunks
# from sentence_transformers import SentenceTransformer
# from langchain_community.embeddings import HuggingFaceEmbeddings


# app = FastAPI()

# # Configure CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class QueryRequest(BaseModel):
#     question: str

# class ResponseModel(BaseModel):
#     answer: str
#     context: str  # Add context field to response model

# # Initialize components once at startup
# @app.on_event("startup")
# async def startup_event():
#     global qa_pipeline, vector_store
    
#     # Load embedding model
#     path = 'google/muril-large-cased'
#     tokenizer = AutoTokenizer.from_pretrained(path)
#     model = AutoModel.from_pretrained(path, output_hidden_states=True)
#     # embedding_fn = MurilEmbeddings(model, tokenizer)
#     embedding_fn = HuggingFaceEmbeddings(
#         model_name="l3cube-pune/indic-sentence-bert-nli",
#         model_kwargs={'device': 'cpu'}
#     )

#     # Create vector store
#     chunks = split_text_into_chunks("/home/akash/ws/IndianRAG/narendra_modi_hindi.txt")
#     documents = [Document(page_content=chunk) for chunk in chunks]
#     vector_store = FAISS.from_documents(
#         documents=documents,
#         embedding=embedding_fn
#     )

#     # Initialize QA pipeline
#     qa_pipeline = pipeline(
#         "question-answering",
#         model="deepset/xlm-roberta-large-squad2",
#         tokenizer="deepset/xlm-roberta-large-squad2",
#     )

# class MurilEmbeddings(Embeddings):
#     def __init__(self, model, tokenizer):
#         self.model = model
#         self.tokenizer = tokenizer
        
#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         embeddings = []
#         for text in texts:
#             inputs = self.tokenizer(
#                 text,
#                 return_tensors="pt",
#                 max_length=512,
#                 truncation=True,
#                 padding=True
#             )
#             with torch.no_grad():
#                 outputs = self.model(**inputs)
#             cls_embedding = outputs.last_hidden_state[0, 0].cpu().numpy()
#             embeddings.append(cls_embedding.tolist())
#         return embeddings

#     def embed_query(self, text: str) -> List[float]:
#         return self.embed_documents([text])[0]

# def get_answer(query: str) -> dict:
#     # Retrieve relevant documents
#     relevant_docs = vector_store.similarity_search(query, k=30)
    
#     # Prepare context
#     context = "\n".join([doc.page_content for doc in relevant_docs])

#     # Get answer from QA model
#     result = qa_pipeline(
#         question=query,
#         context=context,
#         max_answer_len=500,
#         handle_impossible_answer=True
#     )
#     print('result : ')
#     return {
#         "answer": result['answer'] if result['score'] > 0.13 else "दस्तावेज़ में उत्तर नहीं मिला",
#         "context": context
#     }

# @app.post("/ask", response_model=ResponseModel)
# async def ask_question(request: QueryRequest):
#     try:
#         response = get_answer(request.question)
#         return response
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.documents import Document
from typing import List
import numpy as np
import faiss
from transformers import AutoModel, AutoTokenizer, pipeline, AutoModelForCausalLM
import torch
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from text_splitter import split_text_into_chunks
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
    context: str

# Global variables for model components
generator = None
vector_store = None
tokenizer = None

@app.on_event("startup")
async def startup_event():
    global generator, vector_store, tokenizer
    
    # Load embedding model
    embedding_fn = HuggingFaceEmbeddings(
        model_name="l3cube-pune/indic-sentence-bert-nli",
        model_kwargs={'device': 'cpu'}
    )

    # Create vector store
    chunks = split_text_into_chunks("/home/akash/ws/IndianRAG/narendra_modi_hindi.txt")
    documents = [Document(page_content=chunk) for chunk in chunks]
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embedding_fn
    )

    # Initialize DeepSeek model
    model_name = "deepseek-ai/deepseek-llm-7b-chat"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )

def get_answer(query: str) -> dict:
    # Retrieve relevant documents
    relevant_docs = vector_store.similarity_search(query, k=5)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    # Prepare prompt in Hindi
    prompt = f"""नीचे दिए गए संदर्भ का उपयोग करके प्रश्न का उत्तर दें। यदि संदर्भ में उत्तर नहीं है, तो कहें 'दस्तावेज़ में उत्तर नहीं मिला'।
संदर्भ: {context}
प्रश्न: {query}
उत्तर:"""
    
    try:
        # Generate answer
        response = generator(
            prompt,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=True,
            truncation=True,
            max_length=4096
        )
        # Extract answer after the last 'उत्तर:' marker
        full_response = response[0]['generated_text']
        answer = full_response.split("उत्तर:")[-1].strip()
        
        # Fallback for irrelevant answers
        if "नहीं मिला" not in answer and len(answer) < 3:
            answer = "दस्तावेज़ में उत्तर नहीं मिला"
    except Exception as e:
        answer = "त्रुटि: उत्तर प्राप्त नहीं किया जा सका।"
        print(f"Generation error: {str(e)}")

    return {
        "answer": answer,
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