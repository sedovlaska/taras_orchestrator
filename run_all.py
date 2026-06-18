from orchestrator.server import app
from shared.config import settings


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.orchestrator_port)
