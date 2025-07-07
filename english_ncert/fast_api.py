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


@app.on_event("startup")
async def startup_event():
    global chroma_client, collection, model, tokenizer
    
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
    data_dir = "/home/akash/dataset/ncert/v_10_version/lehs101/out"
    
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
        result = generate_answer(request.question, request.top_k)
        print(f"Total time taken: {time.time() - st_time}")
        return {"answer": result["answer"], "context": result["context"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
