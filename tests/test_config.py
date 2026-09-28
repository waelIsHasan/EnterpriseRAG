import importlib
import pytest
import app.config

def test_config_loads_custom_env(monkeypatch):
    """
    Test that the config correctly reads custom environment variables.
    """
    # 1. Isolation: Inject fake environment variables
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "fake_token_123")
    monkeypatch.setenv("LLM_REPO_ID", "custom/Mistral-Test")
    
    # 2. Reload the module so it reads the patched environment
    importlib.reload(app.config)
    
    # 3. Assertions
    assert app.config.Config.HF_TOKEN == "fake_token_123"
    assert app.config.Config.LLM_REPO_ID == "custom/Mistral-Test"

def test_config_default_fallback(monkeypatch):
    """
    Test that the LLM falls back to the default Mistral model if not specified.
    """
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "fake_token_123")
    monkeypatch.delenv("LLM_REPO_ID", raising=False)
    
    importlib.reload(app.config)
    
    assert app.config.Config.HF_TOKEN == "fake_token_123"
    assert app.config.Config.LLM_REPO_ID == "mistralai/Mistral-7B-Instruct-v0.3"