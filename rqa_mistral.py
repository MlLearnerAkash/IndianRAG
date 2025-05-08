from langchain.document_loaders import TextLoader

file_path = "/home/opervu-user/opervu/RAG/test.txt"
loader = TextLoader(file_path)
pages = loader.load()
print(len(pages))  # Check the number of pages loaded


from langchain.text_splitter import RecursiveCharacterTextSplitter

chunk_size = 256
chunk_overlap = 50
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

splits = text_splitter.split_documents(pages)
print(len(splits))  # Number of chunks

from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")


from langchain.vectorstores import Chroma

persist_directory = '/path/to/save/db'

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=persist_directory
)