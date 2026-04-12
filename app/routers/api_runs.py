import asyncio
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.dependencies import verify_api_key
from app.models.orm import Benchmark, EvaluationRun, ModelResponse
from app.schemas.api import ModelResponseOut, ModelStats, RunCreate, RunOut, RunProgress, RunStats
from app.services.runner import execute_run

router = APIRouter(
    prefix="/api/runs",
    tags=["runs"],
    dependencies=[Depends(verify_api_key)],
)


def _build_run_out(run: EvaluationRun, total: int, completed: int) -> RunOut:
    return RunOut(
        id=run.id,
        benchmark_id=run.benchmark_id,
        name=run.name,
        model_names=run.model_names,
        status=run.status,
        created_at=run.created_at,
        completed_at=run.completed_at,
        progress=RunProgress(completed=completed, total=total),
    )


@router.get("", response_model=List[RunOut], summary="List all evaluation runs")
async def list_runs(db: AsyncSession = Depends(get_db)) -> List[RunOut]:
    result = await db.execute(
        select(EvaluationRun)
        .options(
            selectinload(EvaluationRun.benchmark).selectinload(Benchmark.test_cases),
            selectinload(EvaluationRun.responses),
        )
        .order_by(EvaluationRun.created_at.desc())
    )
    runs = result.scalars().all()
    out = []
    for run in runs:
        total = len(run.model_names) * len(run.benchmark.test_cases)
        out.append(_build_run_out(run, total, len(run.responses)))
    return out


@router.post(
    "",
    response_model=RunOut,
    status_code=201,
    summary="Create a run and immediately start it in the background",
)
async def create_run(
    payload: RunCreate,
    db: AsyncSession = Depends(get_db),
) -> RunOut:
    # Validate benchmark exists and has test cases
    bench_result = await db.execute(
        select(Benchmark)
        .where(Benchmark.id == payload.benchmark_id)
        .options(selectinload(Benchmark.test_cases))
    )
    benchmark = bench_result.scalar_one_or_none()
    if benchmark is None:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    if not benchmark.test_cases:
        raise HTTPException(
            status_code=400, detail="Benchmark has no test cases"
        )
    if not payload.model_names:
        raise HTTPException(status_code=400, detail="No models selected")

    run = EvaluationRun(
        benchmark_id=payload.benchmark_id,
        name=payload.name,
        status="pending",
    )
    run.model_names = payload.model_names
    db.add(run)
    await db.commit()
    await db.refresh(run)

    # Fire-and-forget background task using asyncio — has its own DB session
    asyncio.create_task(execute_run(run.id))

    total = len(payload.model_names) * len(benchmark.test_cases)
    return _build_run_out(run, total, 0)


@router.get("/{run_id}", response_model=RunOut, summary="Get a run's current status")
async def get_run(
    run_id: int,
    db: AsyncSession = Depends(get_db),
) -> RunOut:
    result = await db.execute(
        select(EvaluationRun)
        .where(EvaluationRun.id == run_id)
        .options(
            selectinload(EvaluationRun.benchmark).selectinload(Benchmark.test_cases),
            selectinload(EvaluationRun.responses),
        )
    )
    run = result.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    total = len(run.model_names) * len(run.benchmark.test_cases)
    return _build_run_out(run, total, len(run.responses))


@router.get(
    "/{run_id}/responses",
    response_model=List[ModelResponseOut],
    summary="Get all model responses for a run",
)
async def get_run_responses(
    run_id: int,
    db: AsyncSession = Depends(get_db),
) -> List[ModelResponseOut]:
    run_check = await db.get(EvaluationRun, run_id)
    if run_check is None:
        raise HTTPException(status_code=404, detail="Run not found")

    result = await db.execute(
        select(ModelResponse)
        .where(ModelResponse.run_id == run_id)
        .options(selectinload(ModelResponse.manual_score))
        .order_by(ModelResponse.test_case_id, ModelResponse.model_name)
    )
    return list(result.scalars().all())


@router.get(
    "/{run_id}/stats",
    response_model=RunStats,
    summary="Aggregate scoring statistics for a completed run",
)
async def get_run_stats(
    run_id: int,
    db: AsyncSession = Depends(get_db),
) -> RunStats:
    run_check = await db.get(EvaluationRun, run_id)
    if run_check is None:
        raise HTTPException(status_code=404, detail="Run not found")

    result = await db.execute(
        select(ModelResponse)
        .where(ModelResponse.run_id == run_id)
        .options(selectinload(ModelResponse.manual_score))
    )
    responses = result.scalars().all()

    # Aggregate per-model
    model_data: Dict[str, Dict] = {}
    for resp in responses:
        mn = resp.model_name
        if mn not in model_data:
            model_data[mn] = {
                "scores": [],
                "latencies": [],
                "tps": [],
                "count": 0,
            }
        model_data[mn]["count"] += 1
        if resp.latency_ms:
            model_data[mn]["latencies"].append(resp.latency_ms)
        if resp.completion_tokens and resp.latency_ms:
            model_data[mn]["tps"].append(
                resp.completion_tokens / (resp.latency_ms / 1000)
            )
        if resp.manual_score:
            model_data[mn]["scores"].append(resp.manual_score.score)

    def _avg(lst: list) -> Optional[float]:
        return round(sum(lst) / len(lst), 2) if lst else None

    model_stats = [
        ModelStats(
            model_name=mn,
            avg_score=_avg(data["scores"]),
            avg_latency_ms=_avg(data["latencies"]),
            avg_tokens_per_second=_avg(data["tps"]),
            response_count=data["count"],
            scored_count=len(data["scores"]),
        )
        for mn, data in model_data.items()
    ]
    model_stats.sort(key=lambda s: s.avg_score or 0, reverse=True)

    return RunStats(
        run_id=run_check.id,
        run_name=run_check.name,
        status=run_check.status,
        total_responses=len(responses),
        scored_responses=sum(1 for r in responses if r.manual_score),
        model_stats=model_stats,
    )


@router.delete("/{run_id}", status_code=204, summary="Delete a run and all its data")
async def delete_run(
    run_id: int,
    db: AsyncSession = Depends(get_db),
) -> None:
    run = await db.get(EvaluationRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    await db.delete(run)
    await db.commit()
