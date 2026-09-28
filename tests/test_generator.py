import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from app import generator

def test_format_docs_utility():
    """
    Unit test to ensure documents are formatted with double newlines.
    """
    dummy_docs = [
        Document(page_content="First sentence."),
        Document(page_content="Second sentence.")
    ]
    result = generator.format_docs(dummy_docs)
    assert result == "First sentence.\n\nSecond sentence."

@patch("app.generator.get_llm")
@patch("app.generator.get_retriever")
def test_rag_chain_execution(mock_get_retriever, mock_get_llm):
    """
    Integration test for the LangChain pipeline (LCEL) execution.
    Mocks both the local ChromaDB and the Hugging Face inference API.
    """
    # 1. Mock the Retriever (Vector DB)
    mock_retriever_instance = MagicMock()
    mock_docs = [Document(page_content="The backend framework is FastAPI.")]
    mock_retriever_instance.invoke.return_value = mock_docs
    mock_retriever_instance.return_value = mock_docs  # Allows LCEL to call it directly
    mock_get_retriever.return_value = mock_retriever_instance

    # 2. Mock the LLM (Hugging Face)
    mock_llm_instance = MagicMock()
    fake_message = AIMessage(content="Based on the context, the framework is FastAPI.")
    mock_llm_instance.invoke.return_value = fake_message
    mock_llm_instance.return_value = fake_message  # Allows LCEL to call it directly
    mock_get_llm.return_value = mock_llm_instance

    # 3. Build and execute the chain
    chain = generator.build_rag_chain()
    response = chain.invoke("What framework is used?")

    # 4. Assertions
    assert mock_get_retriever.called, "Retriever must be initialized"
    assert mock_get_llm.called, "LLM must be initialized"
    assert "FastAPI" in response, "The output parser should extract the string content."