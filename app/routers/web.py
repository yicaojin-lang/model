from typing import Any, Dict

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.database import get_db
from app.models.orm import Benchmark, EvaluationRun, ModelResponse
from app.services.ollama import ollama_client

templates = Jinja2Templates(directory="app/templates")
router = APIRouter(tags=["web"])


def _ctx(request: Request, **kwargs: Any) -> Dict[str, Any]:
    """Build a template context that always includes the API key for JS fetch calls."""
    return {"request": request, "api_key": settings.api_key, **kwargs}


# ─── Dashboard ────────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def dashboard(
    request: Request, db: AsyncSession = Depends(get_db)
) -> HTMLResponse:
    bench_count = await db.scalar(select(func.count(Benchmark.id))) or 0
    run_count = await db.scalar(select(func.count(EvaluationRun.id))) or 0
    completed_count = (
        await db.scalar(
            select(func.count(EvaluationRun.id)).where(
                EvaluationRun.status == "completed"
            )
        )
        or 0
    )

    recent_result = await db.execute(
        select(EvaluationRun)
        .options(selectinload(EvaluationRun.benchmark))
        .order_by(EvaluationRun.created_at.desc())
        .limit(5)
    )
    recent_runs = recent_result.scalars().all()

    return templates.TemplateResponse(
        "index.html",
        _ctx(
            request,
            bench_count=bench_count,
            run_count=run_count,
            completed_count=completed_count,
            recent_runs=recent_runs,
        ),
    )


# ─── Benchmarks ───────────────────────────────────────────────────────────────

@router.get("/benchmarks", response_class=HTMLResponse)
async def benchmarks_list(
    request: Request, db: AsyncSession = Depends(get_db)
) -> HTMLResponse:
    result = await db.execute(
        select(Benchmark)
        .options(selectinload(Benchmark.test_cases))
        .order_by(Benchmark.created_at.desc())
    )
    benchmarks = result.scalars().all()
    return templates.TemplateResponse(
        "benchmarks/list.html", _ctx(request, benchmarks=benchmarks)
    )


@router.get("/benchmarks/new", response_class=HTMLResponse)
async def benchmark_create_form(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "benchmarks/create.html", _ctx(request)
    )


@router.get("/benchmarks/{benchmark_id}", response_class=HTMLResponse)
async def benchmark_detail(
    request: Request,
    benchmark_id: int,
    db: AsyncSession = Depends(get_db),
) -> HTMLResponse:
    result = await db.execute(
        select(Benchmark)
        .where(Benchmark.id == benchmark_id)
        .options(
            selectinload(Benchmark.test_cases),
            selectinload(Benchmark.runs),
        )
    )
    benchmark = result.scalar_one_or_none()
    if benchmark is None:
        return RedirectResponse("/benchmarks", status_code=302)
    return templates.TemplateResponse(
        "benchmarks/detail.html", _ctx(request, benchmark=benchmark)
    )


# ─── Runs ─────────────────────────────────────────────────────────────────────

@router.get("/runs", response_class=HTMLResponse)
async def runs_list(
    request: Request, db: AsyncSession = Depends(get_db)
) -> HTMLResponse:
    result = await db.execute(
        select(EvaluationRun)
        .options(
            selectinload(EvaluationRun.benchmark).selectinload(Benchmark.test_cases),
            selectinload(EvaluationRun.responses),
        )
        .order_by(EvaluationRun.created_at.desc())
    )
    runs = result.scalars().all()
    runs_data = [
        {
            "run": run,
            "total": len(run.model_names) * len(run.benchmark.test_cases),
            "completed": len(run.responses),
        }
        for run in runs
    ]
    return templates.TemplateResponse(
        "runs/list.html", _ctx(request, runs_data=runs_data)
    )


@router.get("/runs/new", response_class=HTMLResponse)
async def run_create_form(
    request: Request, db: AsyncSession = Depends(get_db)
) -> HTMLResponse:
    bench_result = await db.execute(
        select(Benchmark)
        .options(selectinload(Benchmark.test_cases))
        .order_by(Benchmark.created_at.desc())
    )
    benchmarks = bench_result.scalars().all()

    try:
        models = await ollama_client.list_models()
    except Exception:
        models = []

    return templates.TemplateResponse(
        "runs/new.html", _ctx(request, benchmarks=benchmarks, models=models)
    )


@router.get("/runs/{run_id}", response_class=HTMLResponse)
async def run_detail(
    request: Request,
    run_id: int,
    db: AsyncSession = Depends(get_db),
) -> HTMLResponse:
    result = await db.execute(
        select(EvaluationRun)
        .where(EvaluationRun.id == run_id)
        .options(
            selectinload(EvaluationRun.benchmark).selectinload(Benchmark.test_cases),
            selectinload(EvaluationRun.responses).selectinload(
                ModelResponse.manual_score
            ),
            selectinload(EvaluationRun.responses).selectinload(
                ModelResponse.test_case
            ),
        )
    )
    run = result.scalar_one_or_none()
    if run is None:
        return RedirectResponse("/runs", status_code=302)

    total = len(run.model_names) * len(run.benchmark.test_cases)
    completed = len(run.responses)
    return templates.TemplateResponse(
        "runs/detail.html",
        _ctx(request, run=run, total=total, completed=completed),
    )


@router.get("/runs/{run_id}/compare", response_class=HTMLResponse)
async def run_compare(
    request: Request,
    run_id: int,
    db: AsyncSession = Depends(get_db),
) -> HTMLResponse:
    result = await db.execute(
        select(EvaluationRun)
        .where(EvaluationRun.id == run_id)
        .options(
            selectinload(EvaluationRun.benchmark).selectinload(Benchmark.test_cases),
            selectinload(EvaluationRun.responses).selectinload(
                ModelResponse.manual_score
            ),
            selectinload(EvaluationRun.responses).selectinload(
                ModelResponse.test_case
            ),
        )
    )
    run = result.scalar_one_or_none()
    if run is None:
        return RedirectResponse("/runs", status_code=302)

    test_cases = sorted(run.benchmark.test_cases, key=lambda tc: tc.order_index)
    model_names = run.model_names

    # Blind labels: Model A, Model B, …
    label_map = {mn: chr(65 + i) for i, mn in enumerate(model_names)}

    # response_map[test_case_id][model_name] = ModelResponse | None
    response_map: Dict[int, Dict[str, Any]] = {}
    for resp in run.responses:
        response_map.setdefault(resp.test_case_id, {})[resp.model_name] = resp

    return templates.TemplateResponse(
        "runs/compare.html",
        _ctx(
            request,
            run=run,
            test_cases=test_cases,
            model_names=model_names,
            label_map=label_map,
            response_map=response_map,
        ),
    )
