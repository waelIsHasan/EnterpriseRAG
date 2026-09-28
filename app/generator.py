import logging
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from app.retriever import get_retriever
from app.config import Config

logger = logging.getLogger(__name__)

def get_llm() -> ChatHuggingFace:
    """
    Initialize the LLM connection via Hugging Face Serverless Endpoints.
    """
    if not Config.HF_TOKEN:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set in environment variables.")

    # HuggingFaceEndpoint executes inference over the cloud API
    llm = HuggingFaceEndpoint(
        repo_id=Config.LLM_REPO_ID,
        task="text-generation",
        max_new_tokens=512,
        temperature=0.1,  
        repetition_penalty=1.03,
        huggingfacehub_api_token=Config.HF_TOKEN
    )
    
    # Wrap with ChatHuggingFace to utilize chat-style prompt templates
    return ChatHuggingFace(llm=llm)

def format_docs(docs):
    """Utility to format retrieved ChromaDB chunks into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain():
    """
    Constructs the LangChain Expression Language (LCEL) pipeline.
    """
    retriever = get_retriever(k=3)
    llm = get_llm()
    
    # Strict System Prompt Guardrails
    system_prompt = (
        "You are an enterprise support assistant. Use the following pieces of retrieved context "
        "to answer the user's question. If you don't know the answer or if the context doesn't "
        "contain the information, just say that you don't know. Do not guess or make up information.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])
    
    # LCEL Pipeline Assembly
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain