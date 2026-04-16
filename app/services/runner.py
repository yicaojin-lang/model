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
from app.models.orm import Benchmark, EvaluationRun, ModelResponse, TestCase
from app.services.ollama import ollama_client

logger = logging.getLogger(__name__)


def build_full_history_prompt(current_tc: TestCase, previous_responses: list[ModelResponse], max_turns: int = 3) -> str:
    lines = []
    
    # 核心逻辑：只截取最近的 max_turns 轮对话
    recent_responses = previous_responses[-max_turns:] if len(previous_responses) > max_turns else previous_responses
    
    # 如果有被截断的历史，可以加一句系统提示（可选）
    if len(previous_responses) > max_turns:
        lines.append("[系统提示：为节省内存，已省略部分早期历史对话...]")

    # 重新计算显示的序号（为了排版好看，可以继续用真实的轮次索引）
    start_idx = len(previous_responses) - len(recent_responses) + 1
    
    for idx, response in enumerate(recent_responses, start=start_idx):
        lines.append(f"Q{idx}: {response.test_case.prompt}")
        lines.append(f"A{idx}: {response.response_text or ''}")
        
    lines.append(f"Q{len(previous_responses) + 1}: {current_tc.prompt}")
    return "\n".join(lines)


async def execute_run(run_id: int, context_mode: str = "full_history") -> None:
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
            # Query existing responses only for this run
            existing_responses_result = await db.execute(
                select(ModelResponse)
                .where(ModelResponse.run_id == run_id)
                .options(selectinload(ModelResponse.test_case))
            )
            existing_responses = existing_responses_result.scalars().all()
            existing_response_map = {
                (resp.model_name, resp.test_case_id): resp
                for resp in existing_responses
            }

            for model_name in model_names:
                for idx, tc in enumerate(test_cases):
                    # Skip if response already exists
                    if (model_name, tc.id) in existing_response_map:
                        logger.info(
                            "Run #%d | model=%s | tc#%d | skipped (already exists)",
                            run_id, model_name, tc.id
                        )
                        continue

                    logger.info(
                        "Run #%d | model=%s | tc#%d", run_id, model_name, tc.id
                    )

                    previous_responses = []
                    if context_mode == "full_history" and idx > 0:
                        previous_ids = [tc_prior.id for tc_prior in test_cases[:idx]]
                        # Use existing responses from this run as history context
                        previous_responses = [
                            existing_response_map.get((model_name, tc_id))
                            for tc_id in previous_ids
                            if (model_name, tc_id) in existing_response_map
                        ]
                        previous_responses = [r for r in previous_responses if r]  # Filter None
                        previous_responses.sort(
                            key=lambda r: next(
                                (i for i, tc_prior in enumerate(test_cases) if tc_prior.id == r.test_case_id),
                                0,
                            ),
                        )

                    prompt = tc.prompt
                    if context_mode == "full_history" and previous_responses:
                        prompt = build_full_history_prompt(tc, previous_responses)

                    # Collect all base64 images from the test case
                    # images_list returns [{"data": "base64", "media_type": "image/png"}, ...]
                    images = []
                    try:
                        for img in tc.images_list:
                            if isinstance(img, dict) and "data" in img:
                                images.append(img["data"])
                    except Exception as e:
                        logger.error("Failed to extract images for tc#%d: %s", tc.id, e)
                    
                    images = images if images else None
                    
                    result_obj = await ollama_client.generate(
                        model_name, prompt, images=images
                    )

                    response = ModelResponse(
                        run_id=run_id,
                        test_case_id=tc.id,
                        model_name=model_name,
                        context_mode=context_mode,
                        response_text=result_obj.response_text or None,
                        latency_ms=result_obj.latency_ms,
                        prompt_tokens=result_obj.prompt_tokens,
                        completion_tokens=result_obj.completion_tokens,
                        error=result_obj.error,
                    )
                    db.add(response)
                    # Commit after each response so progress is visible in real time
                    await db.commit()

                    existing_response_map[(model_name, tc.id)] = response

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
