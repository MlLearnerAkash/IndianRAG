from langchain_core.documents import Document
from typing_extensions import List, TypedDict
import numpy as np
import faiss
from transformers import AutoModel, AutoTokenizer
import torch
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document



from langchain_core.embeddings import Embeddings

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


path = 'google/muril-base-cased'
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModel.from_pretrained(path, output_hidden_states=True)


embedding_fn = MurilEmbeddings(model, tokenizer)


from text_splitter import split_text_into_chunks
chunks = split_text_into_chunks("bangladesh_issue.txt")

documents = [Document(page_content=chunk) for chunk in chunks]

vector_store = FAISS.from_documents(
    documents=documents,
    embedding=embedding_fn
)


# 4. Create QA pipeline
from langchain.chains import RetrievalQA
from transformers import pipeline

# Create multilingual QA pipeline
qa_pipeline = pipeline(
    "question-answering",
    model="deepset/xlm-roberta-large-squad2",
    tokenizer="deepset/xlm-roberta-large-squad2",
    

)


def answer_query(query: str) -> str:
    # Step 1: Retrieve relevant documents
    relevant_docs = vector_store.similarity_search(query, k=100)
    
    # Step 2: Prepare context
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    # Step 3: Get answer from QA model
    result = qa_pipeline(
        question=query,
        context=context,
        max_answer_len=100,
        handle_impossible_answer=True
    )
    print(result)
    
    return result['answer'] if result['score'] > 0.13 else "Answer not found in document"

# Usage example
user_query = "सेना प्रमुख ने क्या कहा?"
answer = answer_query(user_query)
print(f"Question: {user_query}")
print(f"Answer: {answer}")

