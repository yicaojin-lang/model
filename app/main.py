import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.config import settings
from app.database import init_db
from app.routers import (
    api_benchmarks,
    api_models,
    api_responses,
    api_runs,
    web,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Initialising local SQLite database …")
    await init_db()
    logger.info("Database ready. Starting server.")
    yield
    logger.info("Server shutting down.")


app = FastAPI(
    title=settings.app_title,
    description=(
        "多模型本地盲测与对比评估工具 — "
        "所有数据严格保存在本地 SQLite，绝不向外网发送任何测试数据。"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ─── Routers ──────────────────────────────────────────────────────────────────
# Web UI routes first (catch-all "/" must come before API prefixes)
app.include_router(web.router)

# RESTful API routes
app.include_router(api_benchmarks.router)
app.include_router(api_runs.router)
app.include_router(api_responses.router)
app.include_router(api_models.router)


# Convenience redirect: /api → /docs
@app.get("/api", include_in_schema=False)
async def api_root() -> RedirectResponse:
    return RedirectResponse(url="/docs")
