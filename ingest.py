import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Paths
DATA_PATH = "data/"
DB_PATH = "vectorstore_db"

def build_brain():
    print("1. Loading PDFs...")
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    
    print(f"2. Splitting {len(docs)} pages into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    chunks = splitter.split_documents(docs)
    
    print("3. Creating Embeddings (Downloading model)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("4. Saving to Database...")
    Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_PATH)
    print("SUCCESS! Brain is ready.")

if __name__ == "__main__":
    build_brain()