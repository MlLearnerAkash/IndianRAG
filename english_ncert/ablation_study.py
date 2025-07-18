# import os
# import json
# import time
# import glob
# import logging
# import requests
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware
# import chromadb
# from chromadb.utils import embedding_functions
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# import spacy
# from bert_score import score
# from huggingface_hub import login
# from typing import Optional, List, Dict, Any
# from enum import Enum

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# app = FastAPI()

# # Configure CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # API configurations
# FIRST_API_URL = "http://10.4.16.40:4554/search/api/word_search/"
# CSRF_TOKEN = "v4IlXw6ZCxiO0Ogno7lggZdIAoTHNaAtleVazecfGnQ2057aQ90vSTrlnp0qpUCW"
# nlp = None

# # Experiment configuration
# class ChunkConfig(BaseModel):
#     size: int = 512
#     overlap: int = 50

# class SimilarityMetric(str, Enum):
#     COSINE = "cosine"
#     EUCLIDEAN = "l2"
#     DOT = "ip"

# class RetrievalMode(str, Enum):
#     VECTOR_ONLY = "vector_only"
#     SEARCH_ONLY = "search_only"
#     HYBRID = "hybrid"

# class CorpusSize(str, Enum):
#     SMALL = "small"
#     LARGE = "large"

# class QueryRequest(BaseModel):
#     question: str
#     top_k: int = 5
#     use_context: bool = True
#     chunk_config: Optional[ChunkConfig] = None
#     similarity_metric: SimilarityMetric = SimilarityMetric.COSINE
#     retrieval_mode: RetrievalMode = RetrievalMode.VECTOR_ONLY
#     hybrid_ratio: float = 0.5
#     llm_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
#     corpus_size: CorpusSize = CorpusSize.LARGE

# class ExperimentResult(BaseModel):
#     answer: str
#     meta_context: list
#     vector_contexts: list
#     latency: dict
#     metrics: dict
#     ablation_config: dict

# # Global variables
# chroma_client = None
# collections = {}  # Stores collections for different chunking strategies
# model = None
# tokenizer = None
# current_llm = ""
# nlp = None
# experiment_history = []
# ground_truth = {}  # Placeholder for ground truth data

# # Initialize ChromaDB collections for different chunking strategies
# def initialize_chroma_collections():
#     global chroma_client, collections
#     chroma_client = chromadb.PersistentClient()
#     logger.info("Chromadb initialized!")
    
#     sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
#         model_name="intfloat/e5-base-v2"
#     )
#     logger.info("Sentence transformer initialized.")
    
#     # Define corpus directories
#     corpus_dirs = {
#         CorpusSize.SMALL: "/home/akash/ws/dataset/icvgip/ISE/easy_ocr/lehs101/small",
#         CorpusSize.LARGE: "/home/akash/ws/dataset/icvgip/ISE/easy_ocr/lehs101/large"
#     }
    
#     # Define chunk configurations
#     chunk_configs = [
#         {"size": 256, "overlap": 25},
#         {"size": 512, "overlap": 50},
#         {"size": 1024, "overlap": 100},
#     ]
    
#     for corpus_size, data_dir in corpus_dirs.items():
#         for config in chunk_configs:
#             config_id = f"{corpus_size.value}_{config['size']}_{config['overlap']}"
#             collection = chroma_client.get_or_create_collection(
#                 name=f"ncert_history_{config_id}",
#                 embedding_function=sentence_transformer_ef,
#                 metadata={"hnsw:space": "cosine"}
#             )
            
#             # Only ingest if collection is empty
#             if collection.count() == 0:
#                 logger.info(f"Ingesting data for {config_id}...")
#                 all_chunks = []
#                 metadatas = []
#                 ids = []
#                 idx = 0
                
#                 for filepath in glob.glob(os.path.join(data_dir, "*.txt")):
#                     with open(filepath, 'r') as f:
#                         text = f.read()
                    
#                     # Simple text splitter (replace with your actual implementation)
#                     chunks = split_text_into_chunks(text, config["size"], config["overlap"])
                    
#                     for chunk in chunks:
#                         all_chunks.append(chunk)
#                         metadatas.append({"source": os.path.basename(filepath)})
#                         ids.append(f"chunk_{idx}")
#                         idx += 1
                
#                 collection.add(
#                     documents=all_chunks,
#                     metadatas=metadatas,
#                     ids=ids
#                 )
#                 logger.info(f"Ingested {len(all_chunks)} chunks for {config_id}")
            
#             collections[config_id] = collection
#             logger.info(f"Collection {config_id} ready")

# # Initialize LLM
# def initialize_llm(model_name: str):
#     global model, tokenizer, current_llm
#     if model_name == current_llm and model is not None:
#         return
    
#     logger.info(f"Loading model: {model_name}")
    
#     tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
#     tokenizer.pad_token = tokenizer.eos_token
    
#     quantization_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.float16,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_use_double_quant=True
#         )
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         quantization_config=quantization_config,
#         torch_dtype=torch.bfloat16,
#         device_map="auto"
#     )
#     current_llm = model_name
#     logger.info(f"Model {model_name} loaded successfully")

# # Custom text splitter implementation
# def split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
#     """Split text into chunks with specified size and overlap"""
#     chunks = []
#     start = 0
#     text_length = len(text)
    
#     while start < text_length:
#         end = min(start + chunk_size, text_length)
#         chunks.append(text[start:end])
#         start = end - chunk_overlap if end - chunk_overlap > start else end
    
#     return chunks

# # Create prompt with chat format
# def create_prompt_with_chat_format(messages, bos="<s>", eos="</s>", add_bos=True):
#     formatted_text = ""
#     for message in messages:
#         if message["role"] == "system":
#             formatted_text += "<|system|>\n" + message["content"] + "\n"
#         elif message["role"] == "user":
#             formatted_text += "<|user|>\n" + message["content"] + "\n"
#         elif message["role"] == "assistant":
#             formatted_text += "<|assistant|>\n" + message["content"].strip() + eos + "\n"
#     formatted_text += "<|assistant|>\n"
#     return bos + formatted_text if add_bos else formatted_text

# @app.on_event("startup")
# async def startup_event():
#     global nlp
#     try:
#         nlp = spacy.load("en_core_web_sm")
#         logger.info("spaCy model loaded successfully")
#     except Exception as e:
#         logger.error(f"Error loading spaCy model: {e}")
#         import subprocess
#         subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
#         nlp = spacy.load("en_core_web_sm")
    
#     # Load ground truth data (placeholder - replace with actual implementation)
#     global ground_truth
#     try:
#         with open("ground_truth.json", "r") as f:
#             ground_truth = json.load(f)
#         logger.info("Ground truth data loaded")
#     except:
#         logger.warning("No ground truth data found")
    
#     initialize_chroma_collections()
#     initialize_llm("mistralai/Mistral-7B-Instruct-v0.2")
#     logger.info("Startup complete")

# # Get context from API
# def get_context_from_api(keyword: str, top_k: int) -> list:
#     """Get context from the external API"""
#     headers = {"accept": "application/json", "X-CSRFTOKEN": CSRF_TOKEN}
#     params = {"q": keyword}
    
#     try:
#         response = requests.get(FIRST_API_URL, headers=headers, params=params, timeout=10)
#         response.raise_for_status()
#         context_data = response.json()
        
#         if isinstance(context_data, list):
#             return context_data[:top_k]
#         elif isinstance(context_data, dict):
#             for field in ["results", "documents", "context", "data"]:
#                 if field in context_data and isinstance(context_data[field], list):
#                     return context_data[field][:top_k]
#             return list(context_data.values())[:top_k]
#         else:
#             return [str(context_data)]
            
#     except Exception as e:
#         logger.error(f"API call failed: {e}")
#         raise HTTPException(status_code=500, detail=f"Context API error: {str(e)}")

# # Extract keyword
# def extract_keyword(question: str) -> str:
#     """Extract the most relevant noun/proper noun from the question"""
#     doc = nlp(question)
    
#     # Prefer nouns/proper nouns that aren't stop words
#     keywords = [token.text for token in doc 
#                if token.pos_ in ("NOUN", "PROPN") and not token.is_stop]
    
#     # Fallback to nouns/proper nouns including stop words
#     if not keywords:
#         keywords = [token.text for token in doc if token.pos_ in ("NOUN", "PROPN")]
    
#     # Final fallback: use the first content word
#     if not keywords:
#         content_words = [token.text for token in doc if not token.is_stop]
#         return content_words[0] if content_words else question.split()[0]
    
#     return keywords[0]

# # Generate answer
# def generate_answer(query: str, contexts: List[str], llm_model: str) -> str:
#     initialize_llm(llm_model)
    
#     prompt_template = '''You are a large language model that answers questions based on the provided context. 
# Please follow these steps:

# 1. **Read the Question**  
# - Carefully read and understand the given question.

# 2. **Read the Context**  
# - Carefully read and understand the context provided below.

# 3. **Generate the Answer**  
# - Using the context, produce a clear and detailed answer to the question.  
# - Ensure the answer is direct, easy to understand, and fact-based.

# ### Now, using the following context and question, provide your answer:

# **Context**:  
# {context}

# **Question**:  
# {query}

# Answer:'''        
    
#     # Handle case with no context
#     if not contexts:
#         contexts = ["No specific context provided. Answer using your general knowledge."]
    
#     formatted_prompt = prompt_template.format(
#         context="\n".join(contexts),
#         query=query
#     )
    
#     messages = [{"role": "user", "content": formatted_prompt}]
#     input_prompt = create_prompt_with_chat_format(messages, add_bos=False)
    
#     inputs = tokenizer(input_prompt, return_tensors="pt", padding=True).to(model.device)
#     with torch.inference_mode():
#         outputs = model.generate(
#             inputs.input_ids,
#             max_new_tokens=1024,
#             do_sample=False
#         )
    
#     full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return full_response.split("Answer:")[-1].strip()

# # Calculate BERT score
# def calculate_bert_score(answer: str, contexts: List[str]) -> List[Dict]:
#     results = []
#     for context in contexts:
#         if not context.strip():
#             results.append({"context": context, "P": 0.0, "R": 0.0, "F1": 0.0})
#             continue
            
#         P, R, F1 = score([answer], [context], lang="en", verbose=False)
#         results.append({
#             "context": context,
#             "P": float(P.mean().item()),
#             "R": float(R.mean().item()),
#             "F1": float(F1.mean().item())
#         })
#     return results

# # Calculate factuality score
# def calculate_factuality(answer: str, question: str) -> float:
#     """Calculate factuality score using ground truth or LLM-as-judge"""
#     # Placeholder implementation - replace with actual method
#     if question in ground_truth:
#         # Simple string matching for demonstration
#         gt_answer = ground_truth[question]
#         return 1.0 if gt_answer.lower() in answer.lower() else 0.0
    
#     # Fallback to BERT score against question
#     P, R, F1 = score([answer], [question], lang="en", verbose=False)
#     return float(F1.mean().item())

# # Run ablation experiment
# def run_ablation_experiment(request: QueryRequest) -> Dict[str, Any]:
#     experiment_config = {
#         "use_context": request.use_context,
#         "chunk_size": request.chunk_config.size if request.chunk_config else 512,
#         "chunk_overlap": request.chunk_config.overlap if request.chunk_config else 50,
#         "similarity_metric": request.similarity_metric.value,
#         "retrieval_mode": request.retrieval_mode.value,
#         "hybrid_ratio": request.hybrid_ratio,
#         "llm_model": request.llm_model,
#         "corpus_size": request.corpus_size.value,
#         "question": request.question
#     }
    
#     latency = {}
#     metrics = {}
#     start_time = time.time()
    
#     # 1. Context Retrieval
#     contexts = []
#     meta_context = []
#     vector_contexts = []
    
#     if request.use_context:
#         # Keyword extraction
#         keyword_start = time.time()
#         keyword = extract_keyword(request.question)
#         latency["keyword_extraction"] = time.time() - keyword_start
        
#         # Retrieve contexts based on ablation config
#         retrieval_start = time.time()
        
#         # Vector-based retrieval
#         if request.retrieval_mode in [RetrievalMode.VECTOR_ONLY, RetrievalMode.HYBRID]:
#             config_id = f"{request.corpus_size.value}_{request.chunk_config.size}_{request.chunk_config.overlap}" if request.chunk_config else f"{request.corpus_size.value}_512_50"
#             collection = collections.get(config_id)
            
#             if not collection:
#                 raise HTTPException(
#                     status_code=404,
#                     detail=f"Collection {config_id} not found"
#                 )
            
#             # Update similarity metric
#             if request.similarity_metric != SimilarityMetric.COSINE:
#                 collection.metadata = {"hnsw:space": request.similarity_metric.value}
            
#             # Query vector DB
#             vector_results = collection.query(
#                 query_texts=[request.question],
#                 n_results=request.top_k
#             )
#             vector_contexts = vector_results['documents'][0]
        
#         # Search engine retrieval
#         if request.retrieval_mode in [RetrievalMode.SEARCH_ONLY, RetrievalMode.HYBRID]:
#             meta_context = get_context_from_api(keyword, request.top_k)
        
#         # Combine contexts based on hybrid ratio
#         if request.retrieval_mode == RetrievalMode.HYBRID:
#             vector_count = int(request.top_k * (1 - request.hybrid_ratio))
#             search_count = request.top_k - vector_count
#             contexts = vector_contexts[:vector_count] + meta_context[:search_count]
#         elif request.retrieval_mode == RetrievalMode.VECTOR_ONLY:
#             contexts = vector_contexts
#         else:
#             contexts = meta_context
        
#         latency["retrieval"] = time.time() - retrieval_start
#     else:
#         # No context case
#         contexts = []
#         meta_context = []
#         vector_contexts = []
#         latency["retrieval"] = 0.0
#         latency["keyword_extraction"] = 0.0
    
#     # 2. Answer Generation
#     generation_start = time.time()
#     answer = generate_answer(request.question, contexts, request.llm_model)
#     latency["generation"] = time.time() - generation_start
    
#     # 3. Evaluation Metrics
#     bert_scores = calculate_bert_score(answer, contexts) if contexts else []
#     metrics["bert_score"] = {
#         "avg_p": sum(score["P"] for score in bert_scores) / len(bert_scores) if bert_scores else 0.0,
#         "avg_r": sum(score["R"] for score in bert_scores) / len(bert_scores) if bert_scores else 0.0,
#         "avg_f1": sum(score["F1"] for score in bert_scores) / len(bert_scores) if bert_scores else 0.0
#     }
    
#     metrics["factuality"] = calculate_factuality(answer, request.question)
#     latency["total"] = time.time() - start_time
    
#     # Store experiment results
#     result = {
#         "question": request.question,
#         "answer": answer,
#         "meta_context": meta_context,
#         "vector_contexts": vector_contexts,
#         "contexts": contexts,
#         "bert_scores": bert_scores,
#         "latency": latency,
#         "metrics": metrics,
#         "config": experiment_config
#     }
#     experiment_history.append(result)
    
#     return result

# # Endpoint to run experiments
# @app.post("/run-experiment", response_model=ExperimentResult)
# async def run_experiment(request: QueryRequest):
#     try:
#         result = run_ablation_experiment(request)
#         return {
#             "answer": result["answer"],
#             "meta_context": result["meta_context"],
#             "vector_contexts": result["vector_contexts"],
#             "latency": result["latency"],
#             "metrics": result["metrics"],
#             "ablation_config": result["config"]
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # Analysis endpoints
# @app.get("/analyze/factuality")
# async def analyze_factuality():
#     if not experiment_history:
#         return {"error": "No experiments conducted yet"}
    
#     # Create new columns from nested data
#     df = pd.DataFrame(experiment_history)
#     df['retrieval_mode'] = df['config'].apply(lambda x: x['retrieval_mode'])
#     df['factuality'] = df['metrics'].apply(lambda x: x['factuality'])
    
#     plt.figure(figsize=(10, 6))
#     sns.boxplot(
#         x="retrieval_mode", 
#         y="factuality", 
#         data=df
#     )
#     plt.title("Factuality by Retrieval Mode")
#     plt.ylabel("Factuality Score")
#     plt.xlabel("Retrieval Mode")
#     plt.savefig("factuality_by_retrieval.png")
    
#     # Factuality by LLM
#     plt.figure(figsize=(10, 6))
#     sns.boxplot(
#         x="llm_model", 
#         y="factuality", 
#         data=df
#     )
#     plt.title("Factuality by LLM Model")
#     plt.ylabel("Factuality Score")
#     plt.xlabel("LLM Model")
#     plt.xticks(rotation=45)
#     plt.savefig("factuality_by_llm.png")
    
#     return {
#         "plots": [
#             "factuality_by_retrieval.png",
#             "factuality_by_llm.png"
#         ]
#     }

# @app.get("/analyze/latency")
# async def analyze_latency():
#     if not experiment_history:
#         return {"error": "No experiments conducted yet"}
    
#     df = pd.DataFrame(experiment_history)
    
#     # Total latency by retrieval mode
#     plt.figure(figsize=(10, 6))
#     sns.boxplot(
#         x="retrieval_mode", 
#         y="latency.total", 
#         data=df
#     )
#     plt.title("Total Latency by Retrieval Mode")
#     plt.ylabel("Latency (seconds)")
#     plt.xlabel("Retrieval Mode")
#     plt.savefig("latency_by_retrieval.png")
    
#     # Latency breakdown
#     latency_df = df["latency"].apply(pd.Series)
#     latency_df = pd.concat([df["config"], latency_df], axis=1)
#     latency_melted = latency_df.melt(
#         id_vars=["config.retrieval_mode", "config.llm_model"], 
#         value_vars=["keyword_extraction", "retrieval", "generation"],
#         var_name="component",
#         value_name="latency"
#     )
    
#     plt.figure(figsize=(12, 8))
#     sns.boxplot(
#         x="config.retrieval_mode", 
#         y="latency", 
#         hue="component",
#         data=latency_melted
#     )
#     plt.title("Latency Breakdown by Component and Retrieval Mode")
#     plt.ylabel("Latency (seconds)")
#     plt.xlabel("Retrieval Mode")
#     plt.legend(title="Component")
#     plt.savefig("latency_breakdown.png")
    
#     return {
#         "plots": [
#             "latency_by_retrieval.png",
#             "latency_breakdown.png"
#         ]
#     }

# @app.get("/analyze/context-relevance")
# async def analyze_context_relevance():
#     if not experiment_history:
#         return {"error": "No experiments conducted yet"}
    
#     df = pd.DataFrame(experiment_history)
    
#     # BERT F1 by chunk size
#     if "config.chunk_size" in df.columns:
#         plt.figure(figsize=(10, 6))
#         sns.boxplot(
#             x="config.chunk_size", 
#             y="metrics.bert_score.avg_f1", 
#             data=df
#         )
#         plt.title("Context Relevance (BERT F1) by Chunk Size")
#         plt.ylabel("Average BERT F1 Score")
#         plt.xlabel("Chunk Size")
#         plt.savefig("relevance_by_chunk_size.png")
    
#     # BERT F1 by corpus size
#     plt.figure(figsize=(10, 6))
#     sns.boxplot(
#         x="config.corpus_size", 
#         y="metrics.bert_score.avg_f1", 
#         data=df
#     )
#     plt.title("Context Relevance (BERT F1) by Corpus Size")
#     plt.ylabel("Average BERT F1 Score")
#     plt.xlabel("Corpus Size")
#     plt.savefig("relevance_by_corpus_size.png")
    
#     return {
#         "plots": [
#             "relevance_by_chunk_size.png",
#             "relevance_by_corpus_size.png"
#         ]
#     }

# @app.get("/export-results")
# async def export_results():
#     if not experiment_history:
#         return {"error": "No experiments conducted yet"}
    
#     # Flatten the data for CSV export
#     flat_data = []
#     for exp in experiment_history:
#         flat_exp = {
#             "question": exp["question"],
#             "answer": exp["answer"],
#             "num_meta_contexts": len(exp["meta_context"]),
#             "num_vector_contexts": len(exp["vector_contexts"]),
#             "bert_avg_p": exp["metrics"]["bert_score"]["avg_p"],
#             "bert_avg_r": exp["metrics"]["bert_score"]["avg_r"],
#             "bert_avg_f1": exp["metrics"]["bert_score"]["avg_f1"],
#             "factuality": exp["metrics"]["factuality"],
#             "total_latency": exp["latency"]["total"],
#             "keyword_extraction_latency": exp["latency"].get("keyword_extraction", 0.0),
#             "retrieval_latency": exp["latency"].get("retrieval", 0.0),
#             "generation_latency": exp["latency"]["generation"],
#         }
#         # Add config parameters
#         flat_exp.update({f"config_{k}": v for k, v in exp["config"].items()})
#         flat_data.append(flat_exp)
    
#     df = pd.DataFrame(flat_data)
#     df.to_csv("experiment_results.csv", index=False)
    
#     # Generate summary report
#     summary = {
#         "total_experiments": len(df),
#         "avg_factuality": df["factuality"].mean(),
#         "avg_bert_f1": df["bert_avg_f1"].mean(),
#         "avg_latency": df["total_latency"].mean(),
#         "best_factuality": {
#             "value": df["factuality"].max(),
#             "config": df.loc[df["factuality"].idxmax()]["config_retrieval_mode"]
#         },
#         "fastest_experiment": {
#             "latency": df["total_latency"].min(),
#             "config": df.loc[df["total_latency"].idxmin()]["config_retrieval_mode"]
#         }
#     }
    
#     return {
#         "data": "experiment_results.csv",
#         "summary": summary
#     }

# @app.get("/clear-results")
# async def clear_results():
#     global experiment_history
#     experiment_history = []
#     return {"message": "Experiment history cleared"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



import os
import json
import time
import glob
import logging
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import chromadb
from chromadb.utils import embedding_functions
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import spacy
from bert_score import score
from huggingface_hub import login
from typing import Optional, List, Dict, Any
from enum import Enum

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

# Experiment configuration
class ChunkConfig(BaseModel):
    size: int = 512
    overlap: int = 50

class SimilarityMetric(str, Enum):
    COSINE = "cosine"
    EUCLIDEAN = "l2"
    DOT = "ip"

class RetrievalMode(str, Enum):
    VECTOR_ONLY = "vector_only"
    SEARCH_ONLY = "search_only"
    HYBRID = "hybrid"

class CorpusSize(str, Enum):
    SMALL = "small"
    LARGE = "large"

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    use_context: bool = True
    chunk_config: Optional[ChunkConfig] = None
    similarity_metric: SimilarityMetric = SimilarityMetric.COSINE
    retrieval_mode: RetrievalMode = RetrievalMode.VECTOR_ONLY
    hybrid_ratio: float = 0.5
    llm_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    corpus_size: CorpusSize = CorpusSize.LARGE

class ExperimentResult(BaseModel):
    answer: str
    meta_context: list
    vector_contexts: list
    latency: dict
    metrics: dict
    ablation_config: dict

# Global variables
chroma_client = None
collections = {}  # Stores collections for different chunking strategies
model = None
tokenizer = None
current_llm = ""
nlp = None
experiment_history = []
ground_truth = {}  # Placeholder for ground truth data

# Initialize ChromaDB collections for different chunking strategies
def initialize_chroma_collections():
    global chroma_client, collections
    chroma_client = chromadb.PersistentClient()
    logger.info("Chromadb initialized!")
    
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/e5-base-v2"
    )
    logger.info("Sentence transformer initialized.")
    
    # Define corpus directories
    corpus_dirs = {
        CorpusSize.SMALL: "/home/akash/ws/dataset/icvgip/ISE/easy_ocr/lehs101/small",
        CorpusSize.LARGE: "/home/akash/ws/dataset/icvgip/ISE/easy_ocr/lehs101/large"
    }
    
    # Define chunk configurations
    chunk_configs = [
        {"size": 256, "overlap": 25},
        {"size": 512, "overlap": 50},
        {"size": 1024, "overlap": 100},
    ]
    
    for corpus_size, data_dir in corpus_dirs.items():
        for config in chunk_configs:
            config_id = f"{corpus_size.value}_{config['size']}_{config['overlap']}"
            collection = chroma_client.get_or_create_collection(
                name=f"ncert_history_{config_id}",
                embedding_function=sentence_transformer_ef,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Only ingest if collection is empty
            if collection.count() == 0:
                logger.info(f"Ingesting data for {config_id}...")
                all_chunks = []
                metadatas = []
                ids = []
                idx = 0
                
                for filepath in glob.glob(os.path.join(data_dir, "*.txt")):
                    with open(filepath, "r") as f:
                        text = f.read()
                    
                    # Simple text splitter (replace with your actual implementation)
                    chunks = split_text_into_chunks(text, config["size"], config["overlap"])
                    
                    for chunk in chunks:
                        all_chunks.append(chunk)
                        metadatas.append({"source": os.path.basename(filepath)})
                        ids.append(f"chunk_{idx}")
                        idx += 1
                
                collection.add(
                    documents=all_chunks,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"Ingested {len(all_chunks)} chunks for {config_id}")
            
            collections[config_id] = collection
            logger.info(f"Collection {config_id} ready")

# Initialize LLM
def initialize_llm(model_name: str):
    global model, tokenizer, current_llm
    if model_name == current_llm and model is not None:
        return
    
    logger.info(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    current_llm = model_name
    logger.info(f"Model {model_name} loaded successfully")

# Custom text splitter implementation
def split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text into chunks with specified size and overlap"""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        start = end - chunk_overlap if end - chunk_overlap > start else end
    
    return chunks

# Create prompt with chat format
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

@app.on_event("startup")
async def startup_event():
    global nlp
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("spaCy model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading spaCy model: {e}")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
    
    # Load ground truth data (placeholder - replace with actual implementation)
    global ground_truth
    try:
        with open("ground_truth.json", "r") as f:
            ground_truth = json.load(f)
        logger.info("Ground truth data loaded")
    except:
        logger.warning("No ground truth data found")
    
    initialize_chroma_collections()
    initialize_llm("mistralai/Mistral-7B-Instruct-v0.2")
    logger.info("Startup complete")

# Get context from API
def get_context_from_api(keyword: str, top_k: int) -> list:
    """Get context from the external API"""
    headers = {"accept": "application/json", "X-CSRFTOKEN": CSRF_TOKEN}
    params = {"q": keyword}
    
    try:
        response = requests.get(FIRST_API_URL, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        context_data = response.json()
        
        if isinstance(context_data, list):
            return context_data[:top_k]
        elif isinstance(context_data, dict):
            for field in ["results", "documents", "context", "data"]:
                if field in context_data and isinstance(context_data[field], list):
                    return context_data[field][:top_k]
            return list(context_data.values())[:top_k]
        else:
            return [str(context_data)]
            
    except Exception as e:
        logger.error(f"API call failed: {e}")
        raise HTTPException(status_code=500, detail=f"Context API error: {str(e)}")

# Extract keyword
def extract_keyword(question: str) -> str:
    """Extract the most relevant noun/proper noun from the question"""
    doc = nlp(question)
    
    # Prefer nouns/proper nouns that aren\"t stop words
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

# Generate answer
def generate_answer(query: str, contexts: List[str], llm_model: str) -> str:
    initialize_llm(llm_model)
    
    prompt_template = """You are a large language model that answers questions based on the provided context. 
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
{context}

**Question**:  
{query}

Answer:"""        
    
    # Handle case with no context
    if not contexts:
        contexts = ["No specific context provided. Answer using your general knowledge."]
    
    formatted_prompt = prompt_template.format(
        context="\n".join(contexts),
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
    return full_response.split("Answer:")[-1].strip()

# Calculate BERT score
def calculate_bert_score(answer: str, contexts: List[str]) -> List[Dict]:
    results = []
    for context in contexts:
        if not context.strip():
            results.append({"context": context, "P": 0.0, "R": 0.0, "F1": 0.0})
            continue
            
        P, R, F1 = score([answer], [context], lang="en", verbose=False)
        results.append({
            "context": context,
            "P": float(P.mean().item()),
            "R": float(R.mean().item()),
            "F1": float(F1.mean().item())
        })
    return results

# Calculate factuality score
def calculate_factuality(answer: str, question: str) -> float:
    """Calculate factuality score using ground truth or LLM-as-judge"""
    # Placeholder implementation - replace with actual method
    if question in ground_truth:
        # Simple string matching for demonstration
        gt_answer = ground_truth[question]
        return 1.0 if gt_answer.lower() in answer.lower() else 0.0
    
    # Fallback to BERT score against question
    P, R, F1 = score([answer], [question], lang="en", verbose=False)
    return float(F1.mean().item())

# Run ablation experiment
def run_ablation_experiment(request: QueryRequest) -> Dict[str, Any]:
    experiment_config = {
        "use_context": request.use_context,
        "chunk_size": request.chunk_config.size if request.chunk_config else 512,
        "chunk_overlap": request.chunk_config.overlap if request.chunk_config else 50,
        "similarity_metric": request.similarity_metric.value,
        "retrieval_mode": request.retrieval_mode.value,
        "hybrid_ratio": request.hybrid_ratio,
        "llm_model": request.llm_model,
        "corpus_size": request.corpus_size.value,
        "question": request.question
    }
    
    latency = {}
    metrics = {}
    start_time = time.time()
    
    # 1. Context Retrieval
    contexts = []
    meta_context = []
    vector_contexts = []
    
    if request.use_context:
        # Keyword extraction
        keyword_start = time.time()
        keyword = extract_keyword(request.question)
        latency["keyword_extraction"] = time.time() - keyword_start
        
        # Retrieve contexts based on ablation config
        retrieval_start = time.time()
        
        # Vector-based retrieval
        if request.retrieval_mode in [RetrievalMode.VECTOR_ONLY, RetrievalMode.HYBRID]:
            config_id = f"{request.corpus_size.value}_{request.chunk_config.size}_{request.chunk_config.overlap}" if request.chunk_config else f"{request.corpus_size.value}_512_50"
            collection = collections.get(config_id)
            
            if not collection:
                raise HTTPException(
                    status_code=404,
                    detail=f"Collection {config_id} not found"
                )
            
            # Update similarity metric
            if request.similarity_metric != SimilarityMetric.COSINE:
                collection.metadata = {"hnsw:space": request.similarity_metric.value}
            
            # Query vector DB
            vector_results = collection.query(
                query_texts=[request.question],
                n_results=request.top_k
            )
            vector_contexts = vector_results["documents"][0]
        
        # Search engine retrieval
        if request.retrieval_mode in [RetrievalMode.SEARCH_ONLY, RetrievalMode.HYBRID]:
            meta_context = get_context_from_api(keyword, request.top_k)
        
        # Combine contexts based on hybrid ratio
        if request.retrieval_mode == RetrievalMode.HYBRID:
            vector_count = int(request.top_k * (1 - request.hybrid_ratio))
            search_count = request.top_k - vector_count
            contexts = vector_contexts[:vector_count] + meta_context[:search_count]
        elif request.retrieval_mode == RetrievalMode.VECTOR_ONLY:
            contexts = vector_contexts
        else:
            contexts = meta_context
        
        latency["retrieval"] = time.time() - retrieval_start
    else:
        # No context case
        contexts = []
        meta_context = []
        vector_contexts = []
        latency["retrieval"] = 0.0
        latency["keyword_extraction"] = 0.0
    
    # 2. Answer Generation
    generation_start = time.time()
    answer = generate_answer(request.question, contexts, request.llm_model)
    latency["generation"] = time.time() - generation_start
    
    # 3. Evaluation Metrics
    bert_scores = calculate_bert_score(answer, contexts) if contexts else []
    metrics["bert_score"] = {
        "avg_p": sum(score["P"] for score in bert_scores) / len(bert_scores) if bert_scores else 0.0,
        "avg_r": sum(score["R"] for score in bert_scores) / len(bert_scores) if bert_scores else 0.0,
        "avg_f1": sum(score["F1"] for score in bert_scores) / len(bert_scores) if bert_scores else 0.0
    }
    
    metrics["factuality"] = calculate_factuality(answer, request.question)
    latency["total"] = time.time() - start_time
    
    # Store experiment results
    result = {
        "question": request.question,
        "answer": answer,
        "meta_context": meta_context,
        "vector_contexts": vector_contexts,
        "contexts": contexts,
        "bert_scores": bert_scores,
        "latency": latency,
        "metrics": metrics,
        "config": experiment_config
    }
    experiment_history.append(result)
    
    return result

# Endpoint to run experiments
@app.post("/run-experiment", response_model=ExperimentResult)
async def run_experiment(request: QueryRequest):
    try:
        result = run_ablation_experiment(request)
        return {
            "answer": result["answer"],
            "meta_context": result["meta_context"],
            "vector_contexts": result["vector_contexts"],
            "latency": result["latency"],
            "metrics": result["metrics"],
            "ablation_config": result["config"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class BatchQueryRequest(BaseModel):
    experiments: List[QueryRequest]

@app.post("/run-batch-experiments")
async def run_batch_experiments(batch_request: BatchQueryRequest):
    results = []
    for request in batch_request.experiments:
        try:
            result = run_ablation_experiment(request)
            results.append({
                "answer": result["answer"],
                "meta_context": result["meta_context"],
                "vector_contexts": result["vector_contexts"],
                "latency": result["latency"],
                "metrics": result["metrics"],
                "ablation_config": result["config"]
            })
        except Exception as e:
            logger.error(f"Error running experiment for question {request.question}: {e}")
            results.append({"error": str(e), "question": request.question})
    return results

# Analysis endpoints
@app.get("/analyze/factuality")
async def analyze_factuality():
    if not experiment_history:
        return {"error": "No experiments conducted yet"}
    
    df = pd.DataFrame(experiment_history)
    
    # Flatten 'config' and 'metrics' columns
    df_config = pd.json_normalize(df['config'])
    df_metrics = pd.json_normalize(df['metrics'])
    df = pd.concat([df.drop(columns=['config', 'metrics']), df_config, df_metrics], axis=1)

    # Factuality by retrieval mode
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="retrieval_mode", 
        y="factuality", 
        data=df,
        errorbar=None
    )
    plt.title("Factuality by Retrieval Mode")
    plt.ylabel("Factuality Score")
    plt.xlabel("Retrieval Mode")
    plt.tight_layout()
    plt.savefig("factuality_by_retrieval.png")
    plt.close()
    
    # Factuality by LLM
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="llm_model", 
        y="factuality", 
        data=df,
        errorbar=None
    )
    plt.title("Factuality by LLM Model")
    plt.ylabel("Factuality Score")
    plt.xlabel("LLM Model")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("factuality_by_llm.png")
    plt.close()
    
    return {
        "plots": [
            "factuality_by_retrieval.png",
            "factuality_by_llm.png"
        ]
    }

@app.get("/analyze/latency")
async def analyze_latency():
    if not experiment_history:
        return {"error": "No experiments conducted yet"}
    
    df = pd.DataFrame(experiment_history)

    # Flatten 'config' and 'latency' columns
    df_config = pd.json_normalize(df['config'])
    df_latency = pd.json_normalize(df['latency'])
    df = pd.concat([df.drop(columns=['config', 'latency']), df_config, df_latency], axis=1)
    
    # Total latency by retrieval mode
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="retrieval_mode", 
        y="total", 
        data=df,
        errorbar=None
    )
    plt.title("Total Latency by Retrieval Mode")
    plt.ylabel("Latency (seconds)")
    plt.xlabel("Retrieval Mode")
    plt.tight_layout()
    plt.savefig("latency_by_retrieval.png")
    plt.close()
    
    # Latency breakdown
    latency_melted = df.melt(
        id_vars=["retrieval_mode", "llm_model"], 
        value_vars=["keyword_extraction", "retrieval", "generation"],
        var_name="component",
        value_name="latency"
    )
    
    plt.figure(figsize=(12, 8))
    sns.barplot(
        x="retrieval_mode", 
        y="latency", 
        hue="component",
        data=latency_melted,
        errorbar=None
    )
    plt.title("Latency Breakdown by Component and Retrieval Mode")
    plt.ylabel("Latency (seconds)")
    plt.xlabel("Retrieval Mode")
    plt.legend(title="Component")
    plt.tight_layout()
    plt.savefig("latency_breakdown.png")
    plt.close()
    
    return {
        "plots": [
            "latency_by_retrieval.png",
            "latency_breakdown.png"
        ]
    }

@app.get("/analyze/context-relevance")
async def analyze_context_relevance():
    if not experiment_history:
        return {"error": "No experiments conducted yet"}
    
    df = pd.DataFrame(experiment_history)

    # Flatten 'config' and 'metrics' columns
    df_config = pd.json_normalize(df['config'])
    df_metrics = pd.json_normalize(df['metrics'])
    df = pd.concat([df.drop(columns=['config', 'metrics']), df_config, df_metrics], axis=1)
    
    # BERT F1 by chunk size
    if "chunk_size" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x="chunk_size", 
            y="bert_score.avg_f1", 
            data=df,
            errorbar=None
        )
        plt.title("Context Relevance (BERT F1) by Chunk Size")
        plt.ylabel("Average BERT F1 Score")
        plt.xlabel("Chunk Size")
        plt.tight_layout()
        plt.savefig("relevance_by_chunk_size.png")
        plt.close()
    
    # BERT F1 by corpus size
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="corpus_size", 
        y="bert_score.avg_f1", 
        data=df,
        errorbar=None
    )
    plt.title("Context Relevance (BERT F1) by Corpus Size")
    plt.ylabel("Average BERT F1 Score")
    plt.xlabel("Corpus Size")
    plt.tight_layout()
    plt.savefig("relevance_by_corpus_size.png")
    plt.close()
    
    return {
        "plots": [
            "relevance_by_chunk_size.png" if "chunk_size" in df.columns else None,
            "relevance_by_corpus_size.png"
        ]
    }

@app.get("/analyze/chunking-strategy")
async def analyze_chunking_strategy():
    if not experiment_history:
        return {"error": "No experiments conducted yet"}
    
    df = pd.DataFrame(experiment_history)
    df_config = pd.json_normalize(df["config"])
    df_metrics = pd.json_normalize(df["metrics"])
    df_latency = pd.json_normalize(df["latency"])
    df = pd.concat([df.drop(columns=["config", "metrics", "latency"]), df_config, df_metrics, df_latency], axis=1)

    plots = []

    if "chunk_size" in df.columns and "chunk_overlap" in df.columns:
        # Factuality by Chunk Size and Overlap
        plt.figure(figsize=(12, 7))
        sns.barplot(x='chunk_size', y='factuality', hue='chunk_overlap', data=df, errorbar=None)
        plt.title('Factuality by Chunk Size and Overlap')
        plt.ylabel('Factuality Score')
        plt.xlabel('Chunk Size')
        plt.legend(title='Chunk Overlap')
        plt.tight_layout()
        plt.savefig('factuality_by_chunking.png')
        plt.close()
        plots.append('factuality_by_chunking.png')

        # BERT F1 by Chunk Size and Overlap
        plt.figure(figsize=(12, 7))
        sns.barplot(x='chunk_size', y='bert_score.avg_f1', hue='chunk_overlap', data=df, errorbar=None)
        plt.title('Context Relevance (BERT F1) by Chunk Size and Overlap')
        plt.ylabel('Average BERT F1 Score')
        plt.xlabel('Chunk Size')
        plt.legend(title='Chunk Overlap')
        plt.tight_layout()
        plt.savefig('relevance_by_chunking.png')
        plt.close()
        plots.append('relevance_by_chunking.png')

        # Total Latency by Chunk Size and Overlap
        plt.figure(figsize=(12, 7))
        sns.barplot(x='chunk_size', y='total', hue='chunk_overlap', data=df, errorbar=None)
        plt.title('Total Latency by Chunk Size and Overlap')
        plt.ylabel('Latency (seconds)')
        plt.xlabel('Chunk Size')
        plt.legend(title='Chunk Overlap')
        plt.tight_layout()
        plt.savefig('latency_by_chunking.png')
        plt.close()
        plots.append('latency_by_chunking.png')

    return {"plots": plots}

@app.get("/analyze/retrieval-llm-comparison")
async def analyze_retrieval_llm_comparison():
    if not experiment_history:
        return {"error": "No experiments conducted yet"}
    
    df = pd.DataFrame(experiment_history)
    df_config = pd.json_normalize(df["config"])
    df_metrics = pd.json_normalize(df["metrics"])
    df_latency = pd.json_normalize(df["latency"])
    df = pd.concat([df.drop(columns=["config", "metrics", "latency"]), df_config, df_metrics, df_latency], axis=1)

    plots = []

    # Factuality by Retrieval Mode and LLM Model
    plt.figure(figsize=(14, 8))
    sns.barplot(x='retrieval_mode', y='factuality', hue='llm_model', data=df, errorbar=None)
    plt.title('Factuality by Retrieval Mode and LLM Model')
    plt.ylabel('Factuality Score')
    plt.xlabel('Retrieval Mode')
    plt.legend(title='LLM Model')
    plt.tight_layout()
    plt.savefig('factuality_retrieval_llm.png')
    plt.close()
    plots.append('factuality_retrieval_llm.png')

    # BERT F1 by Retrieval Mode and LLM Model
    plt.figure(figsize=(14, 8))
    sns.barplot(x='retrieval_mode', y='bert_score.avg_f1', hue='llm_model', data=df, errorbar=None)
    plt.title('Context Relevance (BERT F1) by Retrieval Mode and LLM Model')
    plt.ylabel('Average BERT F1 Score')
    plt.xlabel('Retrieval Mode')
    plt.legend(title='LLM Model')
    plt.tight_layout()
    plt.savefig('relevance_retrieval_llm.png')
    plt.close()
    plots.append('relevance_retrieval_llm.png')

    # Total Latency by Retrieval Mode and LLM Model
    plt.figure(figsize=(14, 8))
    sns.barplot(x='retrieval_mode', y='total', hue='llm_model', data=df, errorbar=None)
    plt.title('Total Latency by Retrieval Mode and LLM Model')
    plt.ylabel('Latency (seconds)')
    plt.xlabel('Retrieval Mode')
    plt.legend(title='LLM Model')
    plt.tight_layout()
    plt.savefig('latency_retrieval_llm.png')
    plt.close()
    plots.append('latency_retrieval_llm.png')

    return {"plots": plots}

@app.get("/export-results")
async def export_results():
    if not experiment_history:
        return {"error": "No experiments conducted yet"}
    
    # Flatten the data for CSV export
    flat_data = []
    for exp in experiment_history:
        flat_exp = {
            "question": exp["question"],
            "answer": exp["answer"],
            "num_meta_contexts": len(exp["meta_context"]),
            "num_vector_contexts": len(exp["vector_contexts"]),
            "bert_avg_p": exp["metrics"]["bert_score"]["avg_p"],
            "bert_avg_r": exp["metrics"]["bert_score"]["avg_r"],
            "bert_avg_f1": exp["metrics"]["bert_score"]["avg_f1"],
            "factuality": exp["metrics"]["factuality"],
            "total_latency": exp["latency"]["total"],
            "keyword_extraction_latency": exp["latency"].get("keyword_extraction", 0.0),
            "retrieval_latency": exp["latency"].get("retrieval", 0.0),
            "generation_latency": exp["latency"]["generation"],
        }
        # Add config parameters
        flat_exp.update({f"config_{k}": v for k, v in exp["config"].items()})
        flat_data.append(flat_exp)
    
    df = pd.DataFrame(flat_data)
    df.to_csv("experiment_results.csv", index=False)
    
    # Generate summary report
    summary = {
        "total_experiments": len(df),
        "avg_factuality": df["factuality"].mean(),
        "avg_bert_f1": df["bert_avg_f1"].mean(),
        "avg_latency": df["total_latency"].mean(),
        "best_factuality": {
            "value": df["factuality"].max(),
            "config": df.loc[df["factuality"].idxmax()]["config_retrieval_mode"]
        },
        "fastest_experiment": {
            "latency": df["total_latency"].min(),
            "config": df.loc[df["total_latency"].idxmin()]["config_retrieval_mode"]
        }
    }
    
    return {
        "data": "experiment_results.csv",
        "summary": summary
    }

@app.get("/clear-results")
async def clear_results():
    global experiment_history
    experiment_history = []
    return {"message": "Experiment history cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


