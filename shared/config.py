from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_model: str = "qwen3:1.7b"
    ollama_host: str = "http://localhost:11434"

    orchestrator_port: int = 8000
    tool_policy_mode: str = "safe"
    tool_allowed_risks: str = "low,medium"
    tool_allowed: str = ""
    tool_denied: str = "system.list_processes,docker.list_containers,docker.list_images"
    tool_approval_required_risks: str = "medium,high"
    tool_approval_ttl_seconds: int = 600
    run_history_db_path: str = ".data/run_history.sqlite3"
    command_allowed_executables: str = "python,ollama,ruff,pytest,docker"
    command_output_max_chars: int = 12000
    command_env_allowlist: str = (
        "PATH,Path,PATHEXT,SYSTEMROOT,SystemRoot,WINDIR,COMSPEC,ComSpec,"
        "TEMP,TMP,HOME,USERPROFILE,LOCALAPPDATA,APPDATA,"
        "PYTHONPATH,OLLAMA_HOST,NO_PROXY,no_proxy,PYTHONIOENCODING"
    )
    workspace_file_list_limit: int = 500
    workspace_max_file_bytes: int = 1_048_576
    workspace_max_file_chars: int = 40_000
    workspace_max_search_results: int = 100

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
