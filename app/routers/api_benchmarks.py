from typing import List
import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.dependencies import verify_api_key
from app.models.orm import Benchmark, TestCase
from app.schemas.api import (
    BenchmarkCreate,
    BenchmarkDetailOut,
    BenchmarkOut,
    TestCaseCreate,
    TestCaseOut,
)

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
        # Handle multiple images or fallback to single image
        images_json = None
        if tc_data.images:
            images_list = [{"data": img.data, "media_type": img.media_type} for img in tc_data.images]
            images_json = json.dumps(images_list, ensure_ascii=False)
        
        tc = TestCase(
            benchmark_id=benchmark.id,
            prompt=tc_data.prompt,
            reference_answer=tc_data.reference_answer,
            order_index=tc_data.order_index if tc_data.order_index else idx,
            _images_json=images_json,
            # Backward compatibility
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


@router.post(
    "/{benchmark_id}/test_cases",
    response_model=TestCaseOut,
    summary="Append a new test case to a benchmark",
)
async def append_test_case(
    benchmark_id: int,
    payload: TestCaseCreate,
    db: AsyncSession = Depends(get_db),
) -> TestCaseOut:
    benchmark = await db.get(Benchmark, benchmark_id)
    if benchmark is None:
        raise HTTPException(status_code=404, detail="Benchmark not found")

    result = await db.execute(
        select(func.max(TestCase.order_index)).where(TestCase.benchmark_id == benchmark_id)
    )
    max_index = result.scalar_one()
    order_index = payload.order_index
    if order_index == 0 and max_index is not None:
        order_index = max_index + 1

    # Handle multiple images or fallback to single image
    images_json = None
    if payload.images:
        images_list = [{"data": img.data, "media_type": img.media_type} for img in payload.images]
        images_json = json.dumps(images_list, ensure_ascii=False)

    test_case = TestCase(
        benchmark_id=benchmark_id,
        prompt=payload.prompt,
        reference_answer=payload.reference_answer,
        order_index=order_index,
        _images_json=images_json,
        # Backward compatibility
        image_data=payload.image_data,
        image_media_type=payload.image_media_type,
    )
    db.add(test_case)
    await db.commit()
    await db.refresh(test_case)
    return test_case


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
