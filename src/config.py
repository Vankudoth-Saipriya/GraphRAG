import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    
    env: str = os.getenv("ENV", "development")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    api_host: str = os.getenv("API_HOST", "127.0.0.1")
    
    vector_db_dir: str = str(BASE_DIR / "data" / "chroma_db")
    graph_storage_path: str = str(BASE_DIR / "data" / "knowledge_graph.json")
    docs_dataset_path: str = str(BASE_DIR / "data" / "synthetic_enterprise_docs.json")
    benchmark_dataset_path: str = str(BASE_DIR / "data" / "eval_benchmark_questions.json")
    
    crag_confidence_threshold: float = 0.65
    embedding_model: str = "gemini-embedding-001"
    llm_model: str = "gemini-2.0-flash"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
