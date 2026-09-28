import pytest
from langchain_core.documents import Document
from app import retriever

def test_retriever_pipeline(tmp_path, monkeypatch):
    """
    اختبار متكامل يتأكد من القدرة على حفظ المتجهات واسترجاعها بدقة،
    مع عزل قاعدة البيانات عن بيئة الإنتاج الحقيقية.
    """
    # 1. العزل (Isolation): تغيير مسار الحفظ ليكون في مجلد مؤقت خاص بالاختبار فقط
    test_db_dir = str(tmp_path / "test_vector_db")
    monkeypatch.setattr(retriever, "VECTOR_DB_DIR", test_db_dir)
    
    # 2. تجهيز البيانات الوهمية (Mock Data)
    dummy_chunks = [
        Document(page_content="The Project Titan backend is built using FastAPI and Python.", metadata={"source": "doc1"}),
        Document(page_content="ChromaDB is used as the primary vector database for storing embeddings.", metadata={"source": "doc2"}),
        Document(page_content="The frontend is written in React, but it is not part of this API.", metadata={"source": "doc3"}),
    ]
    
    # 3. اختبار عملية التخزين (Insertion)
    success = retriever.add_documents_to_db(dummy_chunks)
    assert success is True, "عملية إدراج البيانات في ChromaDB يجب أن تنجح"
    
    # 4. اختبار عملية البحث الدلالي (Semantic Search)
    # نحدد k=1 لجلب أدق نتيجة واحدة فقط
    retriever_engine = retriever.get_retriever(k=1)
    
    # نسأل سؤالاً يجب أن يطابق المستند الثاني
    results = retriever_engine.invoke("What vector database is used?")
    
    # 5. التحقق من الدقة (Assertions)
    assert len(results) == 1, "يجب إرجاع مستند واحد فقط كما حددنا في k"
    assert "ChromaDB" in results[0].page_content, "البحث الدلالي فشل في جلب المستند الصحيح!"