from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_model: str = "qwen3:1.7b"
    ollama_host: str = "http://localhost:11434"

    orchestrator_port: int = 8000
    tool_policy_mode: str = "safe"
    tool_allowed_risks: str = "low,medium"
    tool_allowed: str = ""
    tool_denied: str = "system.list_processes,docker.list_containers,docker.list_images"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
