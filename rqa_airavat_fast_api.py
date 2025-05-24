#@Author: Akash Manna, IIT-Hyd
#@Date:12/05/25
#@ref: https://www.analyticsvidhya.com/blog/2024/11/rag-pipeline-for-hindi-documents/

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import chromadb
from text_splitter import split_text_into_chunks
from chromadb.utils import embedding_functions
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time

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
    
    # Initialize embedding function
    #Base: 768, large: 1024
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-base"
    )
    
    # Create collection
    collection = chroma_client.create_collection(
        name="modi_speech_hindi", 
        embedding_function=sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Ingest documents
    chunks = split_text_into_chunks("/home/akash/ws/IndianRAG/raw_data/all_data.txt")
    collection.add(
        documents=[section for section in chunks],
        metadatas=[{'header': section} for section in chunks],
        ids=[str(i) for i in range(len(chunks))]
    )
    
    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "ai4bharat/Airavata" #"deepseek-ai/DeepSeek-R1"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,  
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16
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


def generate_answer(query: str, top_k:int) -> dict:
    try:
        # Retrieve documents:
        #NOTE: hyperparameter:n_results
        docs = collection.query(
            query_texts=[query],
            n_results=top_k
        )
        context_docs = docs['documents'][0]
        
        # Format prompt
        prompt_template = '''आप एक बड़े भाषा मॉडल हैं जो दिए गए संदर्भ के आधार पर सवालों का उत्तर देते हैं। नीचे दिए गए निर्देशों का पालन करें:

                        1. **प्रश्न पढ़ें**:
                            - दिए गए सवाल को ध्यान से पढ़ें और समझें।

                        2. **संदर्भ पढ़ें**:
                            - नीचे दिए गए संदर्भ को ध्यानपूर्वक पढ़ें और समझें।

                        3. **सूचना उत्पन्न करना**:
                            - संदर्भ का उपयोग करते हुए, प्रश्न का विस्तृत और स्पष्ट उत्तर तैयार करें।
                            - यह सुनिश्चित करें कि उत्तर सीधा, समझने में आसान और तथ्यों पर आधारित हो।

                        ### उदाहरण:

                        **संदर्भ**:
                            "नई दिल्ली भारत की राजधानी है और यह देश का प्रमुख राजनीतिक और प्रशासनिक केंद्र है। यह शहर ऐतिहासिक स्मारकों, संग्रहालयों और विविध संस्कृति के लिए जाना जाता है।"

                        **प्रश्न**:
                            "भारत की राजधानी क्या है और यह क्यों महत्वपूर्ण है?"

                        **प्रत्याशित उत्तर**:
                            "भारत की राजधानी नई दिल्ली है। यह देश का प्रमुख राजनीतिक और प्रशासनिक केंद्र है और ऐतिहासिक स्मारकों, संग्रहालयों और विविध संस्कृति के लिए जाना जाता है।"

                        ### निर्देश:

                        अब, दिए गए संदर्भ और प्रश्न का उपयोग करके उत्तर दें:

                        **संदर्भ**:
                        {docs}

                        **प्रश्न**:
                        {query}

                        उत्तर:'''
        # Keep your existing prompt template here
        formatted_prompt = prompt_template.format(
            docs="\n".join(context_docs),
            query=query
        )
        
        # Prepare input
        messages = [{"role": "user", "content": formatted_prompt}]
        input_prompt = create_prompt_with_chat_format(messages, add_bos=False)
        
        # Generate response
        inputs = tokenizer(input_prompt, return_tensors="pt", padding=True).to(model.device)
        
        with torch.inference_mode():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=1024,
                do_sample=False
            )
        
        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_response.split("उत्तर:")[-1].strip()
        
        return {
            "answer": answer,
            "context": context_docs
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


@app.post("/ask", response_model=ResponseModel)
async def ask_question(request: QueryRequest):
    try:
        st_time= time.time()
        result = generate_answer(request.question, request.top_k)
        print(f"Total time taken: {time.time()- st_time}")
        
        return {
            "answer": result["answer"],
            "context": result["context"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)