from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class RunStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


# ─── Benchmark ────────────────────────────────────────────────────────────────

class ImageData(BaseModel):
    """Single image for multimodal models"""
    data: str = Field(..., description="Base64-encoded image data (no data-URL prefix)")
    media_type: str = Field(..., description="MIME type, e.g. 'image/png', 'image/jpeg'")


class TestCaseCreate(BaseModel):
    prompt: str = Field(..., min_length=1)
    reference_answer: Optional[str] = None
    order_index: int = 0
    # Multiple images for vision models (e.g. llava)
    images: Optional[List[ImageData]] = None
    # Backward compatibility: single image support
    image_data: Optional[str] = None
    image_media_type: Optional[str] = None


class TestCaseOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    benchmark_id: int
    prompt: str
    reference_answer: Optional[str]
    order_index: int
    images_json: Optional[str] = None  # JSON array of images
    image_data: Optional[str] = None  # Backward compat
    image_media_type: Optional[str] = None  # Backward compat
    created_at: datetime


class BenchmarkCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    test_cases: List[TestCaseCreate] = Field(default_factory=list)


class BenchmarkOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: Optional[str]
    created_at: datetime
    test_case_count: int = 0


class BenchmarkDetailOut(BenchmarkOut):
    test_cases: List[TestCaseOut] = Field(default_factory=list)


# ─── Evaluation Run ───────────────────────────────────────────────────────────

class RunCreate(BaseModel):
    benchmark_id: int
    name: str = Field(..., min_length=1, max_length=255)
    model_names: List[str] = Field(..., min_length=1)
    context_mode: Optional[str] = Field(default="full_history")


class RunProgress(BaseModel):
    completed: int
    total: int


class RunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    benchmark_id: int
    name: str
    model_names: List[str]
    status: RunStatus
    created_at: datetime
    completed_at: Optional[datetime]
    progress: Optional[RunProgress] = None


class ManualQuestionCreate(BaseModel):
    prompt: str = Field(..., min_length=1)
    reference_answer: Optional[str] = None


class FollowupCreate(BaseModel):
    response_id: int


class FollowupSuggestionOut(BaseModel):
    suggested_question: str


# ─── Model Response ───────────────────────────────────────────────────────────

class ManualScoreOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    score: int
    notes: Optional[str]
    scored_at: datetime


class ModelResponseOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    run_id: int
    test_case_id: int
    model_name: str
    response_text: Optional[str]
    latency_ms: Optional[float]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    error: Optional[str]
    created_at: datetime
    manual_score: Optional[ManualScoreOut] = None


# ─── Score ────────────────────────────────────────────────────────────────────

class ScoreCreate(BaseModel):
    score: int = Field(..., ge=1, le=5)
    notes: Optional[str] = None


# ─── Statistics ───────────────────────────────────────────────────────────────

class ModelStats(BaseModel):
    model_name: str
    avg_score: Optional[float]
    avg_latency_ms: Optional[float]
    avg_tokens_per_second: Optional[float]
    response_count: int
    scored_count: int


class RunStats(BaseModel):
    run_id: int
    run_name: str
    status: RunStatus
    total_responses: int
    scored_responses: int
    model_stats: List[ModelStats]
