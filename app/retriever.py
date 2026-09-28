import os
import logging
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)

# مسار مجلد حفظ المتجهات محلياً واسم نموذج التضمين
VECTOR_DB_DIR = "vector_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_embeddings_model() -> HuggingFaceEmbeddings:
    """
    تهيئة نموذج التضمين لتحويل النصوص إلى متجهات رياضية.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def get_vectorstore() -> Chroma:
    """
    تهيئة أو استدعاء قاعدة بيانات ChromaDB المحفوظة محلياً لضمان عدم ضياع البيانات.
    """
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    return Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=get_embeddings_model()
    )

def add_documents_to_db(chunks: List[Document]) -> bool:
    """
    إضافة الكتل النصية (Chunks) إلى قاعدة البيانات المتجهية وحفظها.
    """
    if not chunks:
        logger.warning("No chunks provided to insert into the database.")
        return False
        
    logger.info(f"Adding {len(chunks)} chunks to ChromaDB at '{VECTOR_DB_DIR}'...")
    vectorstore = get_vectorstore()
    
    # Chroma ستقوم تلقائياً بتوليد الـ Embeddings وحفظها في مجلد vector_db
    vectorstore.add_documents(documents=chunks)
    logger.info("Successfully stored documents in the vector database.")
    
    return True

def get_retriever(k: int = 3):
    """
    إرجاع واجهة البحث (Retriever) لاستخدامها لاحقاً مع النموذج اللغوي (k=3 تعني جلب أفضل 3 مطابقات).
    """
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k})