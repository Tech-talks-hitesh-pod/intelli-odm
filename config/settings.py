"""Configuration settings for Intelli-ODM system."""

import os
from typing import Dict, Any, Optional, Literal
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # LLM Configuration
    llm_provider: Literal["ollama", "openai", "demo"] = Field(default="ollama", description="LLM provider to use")
    
    # Ollama Configuration
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama server URL")
    ollama_model: str = Field(default="llama3:8b", description="Ollama model to use")
    ollama_timeout: int = Field(default=300, description="Ollama request timeout in seconds")
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model to use")
    openai_temperature: float = Field(default=0.1, description="OpenAI temperature setting")
    
    # Vector Database
    vector_db_type: Literal["chromadb", "faiss"] = Field(default="chromadb", description="Vector database type")
    chromadb_persist_dir: str = Field(default="./data/chromadb", description="ChromaDB persistence directory", alias="chroma_persist_directory")
    chromadb_host: str = Field(default="localhost", description="ChromaDB host")
    chromadb_port: int = Field(default=8000, description="ChromaDB port")
    
    # Business Constraints
    default_budget: int = Field(default=1000000, description="Default budget in INR")
    default_moq: int = Field(default=200, description="Default minimum order quantity")
    default_pack_size: int = Field(default=20, description="Default pack size")
    default_lead_time_days: int = Field(default=30, description="Default lead time in days")
    default_safety_stock_factor: float = Field(default=1.2, description="Default safety stock factor")
    
    # Performance
    max_workers: int = Field(default=4, description="Maximum worker threads")
    embedding_batch_size: int = Field(default=32, description="Embedding batch size")
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: str = Field(default="logs/intelli_odm.log", description="Log file path")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format")
    
    # Development
    debug_mode: bool = Field(default=False, description="Enable debug mode", alias="demo_mode")
    enable_profiling: bool = Field(default=False, description="Enable performance profiling")
    
    # LangSmith / LangChain Observability
    langchain_tracing_v2: bool = Field(default=False, description="Enable LangSmith tracing")
    langchain_project: str = Field(default="intelli-odm", description="LangSmith project name")
    langchain_api_key: Optional[str] = Field(default=None, description="LangSmith API key")
    langchain_endpoint: str = Field(default="https://api.smith.langchain.com", description="LangSmith API endpoint")
    
    # Streamlit UI
    streamlit_port: int = Field(default=8501, description="Streamlit port")
    streamlit_host: str = Field(default="localhost", description="Streamlit host")
    
    @field_validator('llm_provider')
    @classmethod
    def validate_llm_provider(cls, v):
        # Convert 'demo' to 'ollama' for backward compatibility
        if v == "demo":
            return "ollama"
        if v not in ["ollama", "openai"]:
            raise ValueError("llm_provider must be 'ollama', 'openai', or 'demo'")
        return v
    
    @model_validator(mode='after')
    def validate_openai_key(self):
        """Validate that OpenAI API key is provided when using OpenAI provider."""
        if self.llm_provider == 'openai' and not self.openai_api_key:
            raise ValueError("openai_api_key is required when using OpenAI provider")
        return self
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration based on provider."""
        if self.llm_provider == "ollama":
            return {
                "provider": "ollama",
                "base_url": self.ollama_base_url,
                "model": self.ollama_model,
                "timeout": self.ollama_timeout
            }
        else:
            return {
                "provider": "openai",
                "api_key": self.openai_api_key,
                "model": self.openai_model,
                "temperature": self.openai_temperature
            }
    
    def get_business_constraints(self) -> Dict[str, Any]:
        """Get default business constraints."""
        return {
            "budget": self.default_budget,
            "MOQ": self.default_moq,
            "pack_size": self.default_pack_size,
            "lead_time_days": self.default_lead_time_days,
            "safety_stock_factor": self.default_safety_stock_factor
        }
    
    def get_langsmith_config(self) -> Dict[str, Any]:
        """Get LangSmith configuration for tracking."""
        return {
            "enabled": self.langchain_tracing_v2,
            "api_key": self.langchain_api_key,
            "project": self.langchain_project,
            "endpoint": self.langchain_endpoint
        }
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "populate_by_name": True,  # Allow both field name and alias
        "extra": "ignore"  # Ignore extra fields from environment
    }

# Global settings instance
settings = Settings()

def ensure_directories():
    """Ensure required directories exist."""
    directories = [
        "data",
        "data/chromadb", 
        "data/uploads",
        "data/exports",
        "logs",
        "temp",
        "cache"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)