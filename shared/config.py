from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_model: str = "qwen3:1.7b"
    ollama_host: str = "http://localhost:11434"

    orchestrator_port: int = 8000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
