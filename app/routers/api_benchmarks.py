from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.dependencies import verify_api_key
from app.models.orm import Benchmark, TestCase
from app.schemas.api import BenchmarkCreate, BenchmarkDetailOut, BenchmarkOut

router = APIRouter(
    prefix="/api/benchmarks",
    tags=["benchmarks"],
    dependencies=[Depends(verify_api_key)],
)


def _to_out(b: Benchmark) -> BenchmarkOut:
    return BenchmarkOut(
        id=b.id,
        name=b.name,
        description=b.description,
        created_at=b.created_at,
        test_case_count=len(b.test_cases),
    )


def _to_detail_out(b: Benchmark) -> BenchmarkDetailOut:
    return BenchmarkDetailOut(
        id=b.id,
        name=b.name,
        description=b.description,
        created_at=b.created_at,
        test_case_count=len(b.test_cases),
        test_cases=b.test_cases,  # type: ignore[arg-type]
    )


@router.get("", response_model=List[BenchmarkOut], summary="List all benchmarks")
async def list_benchmarks(
    db: AsyncSession = Depends(get_db),
) -> List[BenchmarkOut]:
    result = await db.execute(
        select(Benchmark)
        .options(selectinload(Benchmark.test_cases))
        .order_by(Benchmark.created_at.desc())
    )
    return [_to_out(b) for b in result.scalars().all()]


@router.post(
    "",
    response_model=BenchmarkDetailOut,
    status_code=201,
    summary="Create a new benchmark",
)
async def create_benchmark(
    payload: BenchmarkCreate,
    db: AsyncSession = Depends(get_db),
) -> BenchmarkDetailOut:
    benchmark = Benchmark(name=payload.name, description=payload.description)
    db.add(benchmark)
    await db.flush()

    for idx, tc_data in enumerate(payload.test_cases):
        tc = TestCase(
            benchmark_id=benchmark.id,
            prompt=tc_data.prompt,
            reference_answer=tc_data.reference_answer,
            order_index=tc_data.order_index if tc_data.order_index else idx,
            image_data=tc_data.image_data,
            image_media_type=tc_data.image_media_type,
        )
        db.add(tc)

    await db.commit()

    # Re-fetch with relationships to return complete data
    result = await db.execute(
        select(Benchmark)
        .where(Benchmark.id == benchmark.id)
        .options(selectinload(Benchmark.test_cases))
    )
    return _to_detail_out(result.scalar_one())


@router.get(
    "/{benchmark_id}",
    response_model=BenchmarkDetailOut,
    summary="Get a benchmark with its test cases",
)
async def get_benchmark(
    benchmark_id: int,
    db: AsyncSession = Depends(get_db),
) -> BenchmarkDetailOut:
    result = await db.execute(
        select(Benchmark)
        .where(Benchmark.id == benchmark_id)
        .options(selectinload(Benchmark.test_cases))
    )
    b = result.scalar_one_or_none()
    if b is None:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    return _to_detail_out(b)


@router.delete(
    "/{benchmark_id}",
    status_code=204,
    summary="Delete a benchmark and all its associated data",
)
async def delete_benchmark(
    benchmark_id: int,
    db: AsyncSession = Depends(get_db),
) -> None:
    result = await db.execute(
        select(Benchmark).where(Benchmark.id == benchmark_id)
    )
    b = result.scalar_one_or_none()
    if b is None:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    await db.delete(b)
    await db.commit()
