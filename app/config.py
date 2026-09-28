import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

class Config:
    # Strictly required for the serverless inference API
    HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    # LLM configuration with a fallback default
    LLM_REPO_ID = os.getenv("LLM_REPO_ID", "mistralai/Mistral-7B-Instruct-v0.3")