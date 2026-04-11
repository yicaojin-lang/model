from typing import List

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import verify_api_key
from app.services.ollama import ollama_client

router = APIRouter(
    prefix="/api/models",
    tags=["models"],
    dependencies=[Depends(verify_api_key)],
)


@router.get("", response_model=List[str], summary="List locally available Ollama models")
async def list_models() -> List[str]:
    """
    Returns the names of all models currently installed in the local Ollama instance.
    Raises 503 if Ollama is not reachable.
    """
    try:
        return await ollama_client.list_models()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot reach local Ollama service: {exc}",
        )
