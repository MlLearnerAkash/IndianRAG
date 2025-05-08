from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text_into_chunks(file_path, chunk_size=300, chunk_overlap=50, verbose=False):
   
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        chunks = splitter.split_text(text)
        
        if verbose:
            for i, chunk in enumerate(chunks):
                print(f"Chunk {i+1}:\n{chunk}\n")
        
        return chunks
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return []
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return []