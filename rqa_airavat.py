#@Author: Akash Manna
#@Date:11/05/25
#@ref: https://www.analyticsvidhya.com/blog/2024/11/rag-pipeline-for-hindi-documents/

import chromadb
from text_splitter import split_text_into_chunks

chroma_client = chromadb.Client()

from chromadb.utils import embedding_functions

#initializing embedding model
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="intfloat/multilingual-e5-base")

#creating a collection
collection = chroma_client.create_collection(name="modi_speech_hindi", embedding_function= sentence_transformer_ef, metadata={"hnsw:space": "cosine"})


#Document Infistion
#ingestion documents 


#apparently we need to pass some ids to documents in chroma db, hence using id
chunks = split_text_into_chunks("/home/akash/ws/IndianRAG/narendra_modi_hindi.txt")
collection.add(
    documents=[section for section in chunks], 
    metadatas = [{'header': section} for section in chunks],
    ids=[str(i) for i in range(len(chunks))]
)

docs = collection.query(
    query_texts=["डिजिटल इंडिया अभियान के तहत सरकार ने कौन-कौन सी योजनाएँ लागू की हैं"],
    n_results=3
)
print(docs)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# it should print Cuda


model_name = "ai4bharat/Airavata"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(model_name,  quantization_config=quantization_config, torch_dtype=torch.bfloat16)

def create_prompt_with_chat_format(messages, bos="<s>", eos="</s>", add_bos=True):
    formatted_text = ""
    for message in messages:
        if message["role"] == "system":
            formatted_text += "<|system|>\n" + message["content"] + "\n"
        elif message["role"] == "user":
            formatted_text += "<|user|>\n" + message["content"] + "\n"
        elif message["role"] == "assistant":
            formatted_text += "<|assistant|>\n" + message["content"].strip() + eos + "\n"
        else:
            raise ValueError(
                "Tulu chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(
                    message["role"]
                )
            )
    formatted_text += "<|assistant|>\n"
    formatted_text = bos + formatted_text if add_bos else formatted_text
    return formatted_text

def inference(input_prompts, model, tokenizer):
    input_prompts = [
        create_prompt_with_chat_format([{"role": "user", "content": input_prompt}], add_bos=False)
        for input_prompt in input_prompts
    ]

    encodings = tokenizer(input_prompts, padding=True, return_tensors="pt")
    encodings = encodings.to(device)

    with torch.inference_mode():
        outputs = model.generate(encodings.input_ids, do_sample=False, max_new_tokens=1024)

    output_texts = tokenizer.batch_decode(outputs.detach(), skip_special_tokens=True)

    input_prompts = [
        tokenizer.decode(tokenizer.encode(input_prompt), skip_special_tokens=True) for input_prompt in input_prompts
    ]
    output_texts = [output_text[len(input_prompt) :] for input_prompt, output_text in zip(input_prompts, output_texts)]
    return output_texts


prompt ='''आप एक बड़े भाषा मॉडल हैं जो दिए गए संदर्भ के आधार पर सवालों का उत्तर देते हैं। नीचे दिए गए निर्देशों का पालन करें:

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


def generate_answer(query):
  docs =  collection.query(
    query_texts=[query],
    n_results=3
) #taking top 3 results 
  docs = [doc for doc in docs['documents'][0]]
  docs = "\n".join(docs)
  formatted_prompt = prompt.format(docs = docs,query = query)
  answers = inference([formatted_prompt], model, tokenizer)
  return answers[0]


questions = [
    "सरकार ने चंडीगढ़ की समस्या को क्यों जटिल बना दिया है?"
]

for question in questions:
    answer = generate_answer(question)
    print(f"Question: {question}\nAnswer: {answer}\n")