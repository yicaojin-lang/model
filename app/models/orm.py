import json
from datetime import datetime
from typing import List, Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Benchmark(Base):
    __tablename__ = "benchmarks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    test_cases: Mapped[List["TestCase"]] = relationship(
        back_populates="benchmark",
        cascade="all, delete-orphan",
        order_by="TestCase.order_index",
    )
    runs: Mapped[List["EvaluationRun"]] = relationship(
        back_populates="benchmark",
        cascade="all, delete-orphan",
    )


class TestCase(Base):
    __tablename__ = "test_cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    benchmark_id: Mapped[int] = mapped_column(
        ForeignKey("benchmarks.id"), nullable=False
    )
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    reference_answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    order_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # Optional image for multimodal models (e.g. llava).
    # Stored as raw base64 string (no data-URL prefix).
    image_data: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # MIME type of the image, e.g. "image/png", "image/jpeg"
    image_media_type: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    benchmark: Mapped["Benchmark"] = relationship(back_populates="test_cases")
    responses: Mapped[List["ModelResponse"]] = relationship(
        back_populates="test_case"
    )


class EvaluationRun(Base):
    """A single comparative evaluation run of N models against a benchmark."""

    __tablename__ = "evaluation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    benchmark_id: Mapped[int] = mapped_column(
        ForeignKey("benchmarks.id"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    # Stored as JSON array of model name strings, e.g. '["llama3", "gemma:7b"]'
    _model_names: Mapped[str] = mapped_column(
        "model_names", Text, nullable=False
    )
    # pending | running | completed | failed
    status: Mapped[str] = mapped_column(
        String(50), default="pending", nullable=False
    )
    context_mode: Mapped[str] = mapped_column(
        String(50), default="full_history", nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    benchmark: Mapped["Benchmark"] = relationship(back_populates="runs")
    responses: Mapped[List["ModelResponse"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    scores: Mapped[List["ManualScore"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )

    @property
    def model_names(self) -> List[str]:
        return json.loads(self._model_names)

    @model_names.setter
    def model_names(self, value: List[str]) -> None:
        self._model_names = json.dumps(value, ensure_ascii=False)


class ModelResponse(Base):
    """One inference result: a specific model's answer to a specific test case."""

    __tablename__ = "model_responses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("evaluation_runs.id"), nullable=False
    )
    test_case_id: Mapped[int] = mapped_column(
        ForeignKey("test_cases.id"), nullable=False
    )
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    context_mode: Mapped[str] = mapped_column(
        String(50), default="full_history", nullable=False
    )
    response_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    prompt_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    run: Mapped["EvaluationRun"] = relationship(back_populates="responses")
    test_case: Mapped["TestCase"] = relationship(back_populates="responses")
    manual_score: Mapped[Optional["ManualScore"]] = relationship(
        back_populates="response", uselist=False
    )


class ManualScore(Base):
    """Human evaluation score (1–5) assigned to a model response."""

    __tablename__ = "manual_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("evaluation_runs.id"), nullable=False
    )
    response_id: Mapped[int] = mapped_column(
        ForeignKey("model_responses.id"), nullable=False, unique=True
    )
    score: Mapped[int] = mapped_column(Integer, nullable=False)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    scored_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    run: Mapped["EvaluationRun"] = relationship(back_populates="scores")
    response: Mapped["ModelResponse"] = relationship(back_populates="manual_score")
