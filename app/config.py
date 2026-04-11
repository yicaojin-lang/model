from dataclasses import dataclass, field


@dataclass
class Settings:
    # Ollama local gateway — NEVER points to any external network
    ollama_base_url: str = "http://127.0.0.1:11434"

    # SQLite stored entirely on local disk
    database_url: str = "sqlite+aiosqlite:///./data/modelTool.db"

    # Keep-alive = 0: unload model from VRAM immediately after each inference.
    # This is mandatory on unified-memory hardware (Mac Mini) to prevent OOM.
    keep_alive: int = 0

    # API key for the RESTful API (used by external projects / LAN protection).
    # Change this to any secret string. Set to "" to disable auth entirely.
    api_key: str = "local-modelTool-key"

    app_title: str = "多模型盲测评估工具"
    debug: bool = False


settings = Settings()
