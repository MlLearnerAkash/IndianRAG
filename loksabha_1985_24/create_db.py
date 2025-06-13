import chromadb
from chromadb.utils import embedding_functions
import glob
import os
from text_splitter import split_text_into_chunks # Assuming text_splitter.py is available

def create_and_persist_chromadb(data_dir: str, db_path: str = "./chroma_db"):
    """
    Creates and persists a ChromaDB collection from text files in a given directory.

    Args:
        data_dir (str): Path to the directory containing .txt files.
        db_path (str): Path to store the persistent ChromaDB.
    """
    print(f"Initializing ChromaDB at {db_path}...")
    chroma_client = chromadb.PersistentClient(path=db_path)
    print("ChromaDB initialized!")

    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("Sentence transformer initialized.")

    print(">> Collection initializing...")
    collection = chroma_client.get_or_create_collection(
        name="loksabha_debates",
        embedding_function=sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"}
    )
    print("Collection initialized!")

    all_chunks = []
    metadatas = []
    ids = []
    idx = 0

    print(f"Reading text files from {data_dir}...")
    for filepath in glob.glob(os.path.join(data_dir, "*.txt")):
        print(f"Processing {filepath}...")
        chunks = split_text_into_chunks(filepath)
        for chunk in chunks:
            all_chunks.append(chunk)
            metadatas.append({"source": os.path.basename(filepath), "header": chunk[:50]})
            ids.append(f"chunk_{idx}")
            idx += 1
    print(f"Found {len(all_chunks)} chunks.")

    if all_chunks:
        print("Ingesting into ChromaDB...")
        collection.add(
            documents=all_chunks,
            metadatas=metadatas,
            ids=ids
        )
        print("Ingestion complete!")
    else:
        print("No chunks to ingest.")

if __name__ == "__main__":
    # Example usage:
    # Make sure to replace with your actual data directory
    data_directory = "/home/akash/dataset/loksabha_rqa/txt_files"  # Placeholder: User needs to provide this path
    chroma_db_path = "./chroma_db" # Default path for persistent ChromaDB

    # Create a dummy directory and file for demonstration if they don't exist
    os.makedirs(data_directory, exist_ok=True)
    with open(os.path.join(data_directory, "sample.txt"), "w") as f:
        f.write("This is a sample text file for testing ChromaDB. It contains some information about the Lok Sabha debates.")

    # Assuming text_splitter.py is in the same directory or accessible in PYTHONPATH
    # You might need to create a dummy text_splitter.py if it's not provided
    # For now, let's assume it exists and has a split_text_into_chunks function
    try:
        from text_splitter import split_text_into_chunks
    except ImportError:
        print("Error: text_splitter.py not found. Please provide or create it.")
        print("Creating a dummy text_splitter.py for demonstration.")
        with open("text_splitter.py", "w") as f:
            f.write("def split_text_into_chunks(filepath):\n    with open(filepath, 'r') as f:\n        content = f.read()\n    # Simple split for demonstration, you'd have a more sophisticated one\n    return [content[i:i+200] for i in range(0, len(content), 200)]")
        from text_splitter import split_text_into_chunks

    create_and_persist_chromadb(data_directory, chroma_db_path)


