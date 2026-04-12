"""
Evaluation runner — executes a run in a background asyncio task.

Execution strategy (memory-safe for unified-memory hardware):
  • Outer loop: one model at a time (sequential)
  • Inner loop: test cases for that model (sequential)
  • keep_alive=0 on every request → model evicted from VRAM between calls

This prevents loading two large models simultaneously and triggering OOM.
"""
import asyncio
import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.database import AsyncSessionLocal
from app.models.orm import Benchmark, EvaluationRun, ModelResponse
from app.services.ollama import ollama_client

logger = logging.getLogger(__name__)


async def execute_run(run_id: int) -> None:
    """
    Background coroutine: drives the full lifecycle of one evaluation run.
    Uses its own database session (independent of the HTTP request session).
    """
    async with AsyncSessionLocal() as db:
        # ── Load run with related data ──────────────────────────────────────
        result = await db.execute(
            select(EvaluationRun)
            .where(EvaluationRun.id == run_id)
            .options(
                selectinload(EvaluationRun.benchmark).selectinload(
                    Benchmark.test_cases
                )
            )
        )
        run = result.scalar_one_or_none()
        if run is None:
            logger.error("execute_run: run_id=%d not found", run_id)
            return

        run.status = "running"
        await db.commit()

        model_names = run.model_names
        test_cases = sorted(
            run.benchmark.test_cases, key=lambda tc: tc.order_index
        )
        total = len(model_names) * len(test_cases)
        logger.info(
            "Run #%d started: %d model(s) × %d test case(s) = %d inferences",
            run_id,
            len(model_names),
            len(test_cases),
            total,
        )

        try:
            for model_name in model_names:
                for tc in test_cases:
                    logger.info(
                        "Run #%d | model=%s | tc#%d", run_id, model_name, tc.id
                    )
                    images = [tc.image_data] if tc.image_data else None
                    result_obj = await ollama_client.generate(
                        model_name, tc.prompt, images=images
                    )

                    response = ModelResponse(
                        run_id=run_id,
                        test_case_id=tc.id,
                        model_name=model_name,
                        response_text=result_obj.response_text or None,
                        latency_ms=result_obj.latency_ms,
                        prompt_tokens=result_obj.prompt_tokens,
                        completion_tokens=result_obj.completion_tokens,
                        error=result_obj.error,
                    )
                    db.add(response)
                    # Commit after each response so progress is visible in real time
                    await db.commit()

            run.status = "completed"
            run.completed_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
            await db.commit()
            logger.info("Run #%d completed successfully", run_id)

        except asyncio.CancelledError:
            run.status = "failed"
            await db.commit()
            logger.warning("Run #%d was cancelled", run_id)
            raise

        except Exception:
            logger.exception("Run #%d failed with an unexpected error", run_id)
            run.status = "failed"
            await db.commit()
