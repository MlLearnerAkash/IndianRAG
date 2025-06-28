# #@Author: Akaash Manna
# #@Date: 06/05/25
# #@Ack:used LLM when needed

# """
# This code is meant to split the text and then intergrate to vector DB
# """
# import sys
# import os
# import numpy as np
# import faiss
# from langchain_community.vectorstores import FAISS


# path = 'google/muril-base-cased'


# from transformers import AutoModel, AutoTokenizer
# import torch
# tokenizer = AutoTokenizer.from_pretrained(path)
# model = AutoModel.from_pretrained(path,output_hidden_states=True)


# from text_splitter import split_text_into_chunks

# chunks = split_text_into_chunks("test.txt")


# for chunk in chunks:
#     print(tokenizer.convert_ids_to_tokens(tokenizer.encode(chunk)))
#     input_encoded = tokenizer.encode_plus(chunk, return_tensors="pt")
#     with torch.no_grad():
#             states = model(**input_encoded).hidden_states
#     output = torch.stack([states[i] for i in range(len(states))])
#     output = output.squeeze()
    
    
#     # Create FAISS index
#     embedding_dim = output.shape[1]
#     faiss_index = faiss.IndexFlatL2(embedding_dim)
#     faiss_index.add(output)

#     vector_store = FAISS(
#     faiss_index=faiss_index,
#     embedding_function=None,  # Not needed; you already have embeddings
#     docstore=None             # No text/documents
# )





import numpy as np
import faiss
from transformers import AutoModel, AutoTokenizer
import torch
from langchain_community.vectorstores import FAISS

path = 'google/muril-base-cased'
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModel.from_pretrained(path, output_hidden_states=True)


from text_splitter import split_text_into_chunks
chunks = split_text_into_chunks("test.txt")

embeddings = []
for chunk in chunks:
    input_encoded = tokenizer.encode_plus(chunk, return_tensors="pt")
    with torch.no_grad():
        states = model(**input_encoded).hidden_states
    # Get last layer, first token ([CLS])
    cls_embedding = states[-1][0][0].numpy()  # (hidden_size,)
    embeddings.append(cls_embedding)

embeddings = np.stack(embeddings).astype('float32')  # shape: (num_chunks, hidden_size)
embedding_dim = embeddings.shape[1]

faiss_index = faiss.IndexFlatL2(embedding_dim)
faiss_index.add(embeddings)


print(faiss_index)

vector_store = FAISS(
faiss_index=faiss_index,
embedding_function=None,  # Not needed; you already have embeddings
docstore=None             # No text/documents
)