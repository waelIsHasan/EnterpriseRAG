import os
import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Configure logging instead of print() for production visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 2. Define chunking strategy constraints
# 800 characters provides enough context for the LLM. 
# 150 characters overlap ensures concepts split across chunks remain connected.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

def load_documents(data_dir: str = "data") -> List[Document]:
    """
    Scans the data directory and loads raw text files.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        logger.warning(f"Created missing data directory at '{data_dir}'. Please add files.")
        return []
        
    if not os.listdir(data_dir):
        logger.warning(f"Data directory '{data_dir}' is empty.")
        return []

    # Using DirectoryLoader to batch-process files.
    # glob="**/*.txt" targets text files. You can expand this to PDFs later.
    loader = DirectoryLoader(
        data_dir, 
        glob="**/*.txt", 
        loader_cls=TextLoader,
        show_progress=True
    )
    
    logger.info(f"Loading documents from {data_dir}...")
    documents = loader.load()
    logger.info(f"Successfully loaded {len(documents)} document(s).")
    
    return documents

def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Splits raw documents into overlapping semantic chunks.
    """
    if not documents:
        return []

    # RecursiveCharacterTextSplitter tries to split by double newline \n\n (paragraphs) first, 
    # then single \n, then spaces, ensuring we don't break thoughts mid-sentence.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    
    logger.info("Chunking documents...")
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} documents into {len(chunks)} independent chunks.")
    
    return chunks

def run_ingestion_pipeline(data_dir: str = "data") -> List[Document]:
    """
    Orchestrates the loading and chunking process.
    This function will be called by our API layer later.
    """
    raw_docs = load_documents(data_dir)
    processed_chunks = chunk_documents(raw_docs)
    return processed_chunks