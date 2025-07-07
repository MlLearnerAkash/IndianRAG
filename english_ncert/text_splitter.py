from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text_into_chunks(file_path, chunk_size=300, chunk_overlap=50, verbose=False):
   
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,

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

# """
# Text Splitter for RAG Pipeline
# Author: Akash
# Date: May 24, 2025
# """

# def split_text_into_chunks(file_path, chunk_size=200, overlap=50):
#     """
#     Split text from a file into overlapping chunks.
    
#     Args:
#         file_path (str): Path to the text file
#         chunk_size (int): Size of each chunk in characters
#         overlap (int): Overlap between chunks in characters
        
#     Returns:
#         list: List of text chunks
#     """
#     try:
#         with open(file_path, 'r') as file:
#             text = file.read()
#     except Exception as e:
#         print(f"Error reading file: {str(e)}")
#         return []
    
#     # Split text into chunks
#     chunks = []
#     start = 0
    
#     while start < len(text):
#         end = min(start + chunk_size, len(text))
        
#         # Try to find a sentence or paragraph break for cleaner chunks
#         if end < len(text):
#             # Look for paragraph break
#             paragraph_break = text.rfind('\n\n', start, end)
#             if paragraph_break != -1 and paragraph_break > start + chunk_size // 2:
#                 end = paragraph_break + 2
#             else:
#                 # Look for sentence break
#                 sentence_breaks = [text.rfind('. ', start, end), 
#                                   text.rfind('। ', start, end),  # Hindi sentence break
#                                   text.rfind('? ', start, end),
#                                   text.rfind('! ', start, end)]
#                 best_break = max(sentence_breaks)
#                 if best_break != -1 and best_break > start + chunk_size // 2:
#                     end = best_break + 2
        
#         # Add chunk to list
#         chunks.append(text[start:end].strip())
        
#         # Move start position for next chunk, considering overlap
#         start = end - overlap
    
#     return chunks
