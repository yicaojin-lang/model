import logging
from typing import AsyncGenerator

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

logger = logging.getLogger(__name__)

engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    connect_args={"check_same_thread": False},
)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def _migrate_sqlite_columns(sync_conn: Connection) -> None:
    """Align existing SQLite files with newer ORM columns.

    SQLAlchemy ``create_all`` only creates missing tables; it does not add columns
    to tables that already exist. Old deployments keep hitting OperationalError until
    we ALTER TABLE explicitly.
    """
    if sync_conn.dialect.name != "sqlite":
        return

    insp = inspect(sync_conn)

    pending: list[str] = []

    if insp.has_table("test_cases"):
        cols = {c["name"] for c in insp.get_columns("test_cases")}
        if "images_json" not in cols:
            pending.append("ALTER TABLE test_cases ADD COLUMN images_json TEXT")

    if insp.has_table("evaluation_runs"):
        cols = {c["name"] for c in insp.get_columns("evaluation_runs")}
        if "context_mode" not in cols:
            pending.append(
                "ALTER TABLE evaluation_runs ADD COLUMN context_mode VARCHAR(50) "
                "NOT NULL DEFAULT 'full_history'"
            )

    if insp.has_table("model_responses"):
        cols = {c["name"] for c in insp.get_columns("model_responses")}
        if "context_mode" not in cols:
            pending.append(
                "ALTER TABLE model_responses ADD COLUMN context_mode VARCHAR(50) "
                "NOT NULL DEFAULT 'full_history'"
            )

    for ddl in pending:
        sync_conn.execute(text(ddl))
        logger.info("SQLite schema patched: %s", ddl.split("COLUMN", 1)[-1].strip()[:72])


async def init_db() -> None:
    from app.models import orm  # noqa: F401 — ensure all ORM models are registered

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_migrate_sqlite_columns)
