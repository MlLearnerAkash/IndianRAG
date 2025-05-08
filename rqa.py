import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.vectorstores import InMemoryVectorStore
import numpy as np
import faiss
from transformers import AutoModel, AutoTokenizer
import torch
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
# Load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# all_splits = text_splitter.split_documents(docs)


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
chunks = split_text_into_chunks("test.txt")

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
    relevant_docs = vector_store.similarity_search(query, k=30)
    
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
    
    return result['answer'] if result['score'] > 0.3 else "Answer not found in document"

# Usage example
user_query = "आराज यूनिवर्सिटी में भूख हड़ताल के दौरान क्यों धारा 144 लगाई गई है?"
answer = answer_query(user_query)
print(f"Question: {user_query}")
print(f"Answer: {answer}")

#--------------------------------
# path = 'google/muril-base-cased'
# tokenizer = AutoTokenizer.from_pretrained(path)
# model = AutoModel.from_pretrained(path, output_hidden_states=True)

# from text_splitter import split_text_into_chunks
# chunks = split_text_into_chunks("test.txt")

# embeddings = []
# for chunk in chunks:
#     input_encoded = tokenizer.encode_plus(chunk, return_tensors="pt")
#     with torch.no_grad():
#         states = model(**input_encoded).hidden_states
#     # Get last layer, first token ([CLS])
#     cls_embedding = states[-1][0][0].numpy()  # (hidden_size,)
#     embeddings.append(cls_embedding)

# embeddings = np.stack(embeddings).astype('float32')  # shape: (num_chunks, hidden_size)

# vector_store = InMemoryVectorStore(model)

# documents = [Document(page_content=chunk) for chunk in chunks]

# vector_store = FAISS.from_embeddings(
#     text_embeddings=list(zip(chunks, embeddings)),
#     embedding=model.get_input_embeddings(),  # Or create dummy embedding function
#     metadatas=[{}]*len(chunks)
# )
# # Index chunks
# # _ = vector_store.add_documents(documents=documents)

# # Define prompt for question-answering
# prompt = hub.pull("rlm/rag-prompt")


# # Define state for application
# class State(TypedDict):
#     question: str
#     context: List[Document]
#     answer: str


# # Define application steps
# def retrieve(state: State):
#     retrieved_docs = vector_store.similarity_search(state["question"])
#     return {"context": retrieved_docs}


# def generate(state: State):
#     docs_content = "\n\n".join(doc.page_content for doc in state["context"])
#     messages = prompt.invoke({"question": state["question"], "context": docs_content})
#     response = llm.invoke(messages)
#     return {"answer": response.content}


# # Compile application and test
# graph_builder = StateGraph(State).add_sequence([retrieve, generate])
# graph_builder.add_edge(START, "retrieve")
# graph = graph_builder.compile()


# response = graph.invoke({"question": "What is Task Decomposition?"})
# print(response["answer"])



# #--------------------------