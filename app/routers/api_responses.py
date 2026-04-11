from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.dependencies import verify_api_key
from app.models.orm import ManualScore, ModelResponse
from app.schemas.api import ModelResponseOut, ScoreCreate

router = APIRouter(
    prefix="/api/responses",
    tags=["responses"],
    dependencies=[Depends(verify_api_key)],
)


@router.post(
    "/{response_id}/score",
    response_model=ModelResponseOut,
    summary="Submit or update a manual score (1–5) for a model response",
)
async def score_response(
    response_id: int,
    payload: ScoreCreate,
    db: AsyncSession = Depends(get_db),
) -> ModelResponseOut:
    result = await db.execute(
        select(ModelResponse)
        .where(ModelResponse.id == response_id)
        .options(selectinload(ModelResponse.manual_score))
    )
    response = result.scalar_one_or_none()
    if response is None:
        raise HTTPException(status_code=404, detail="Response not found")

    if response.manual_score:
        # Update existing score in-place
        response.manual_score.score = payload.score
        response.manual_score.notes = payload.notes
    else:
        score = ManualScore(
            run_id=response.run_id,
            response_id=response_id,
            score=payload.score,
            notes=payload.notes,
        )
        db.add(score)

    await db.commit()

    # Re-fetch to return fresh state
    result = await db.execute(
        select(ModelResponse)
        .where(ModelResponse.id == response_id)
        .options(selectinload(ModelResponse.manual_score))
    )
    return result.scalar_one()
