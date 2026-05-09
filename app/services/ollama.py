"""
Ollama local inference client.

Design constraints (Mac Mini unified-memory hardware):
  - keep_alive=0  →  unload model from VRAM immediately after every response,
                     preventing out-of-memory when multiple models are tested.
  - All traffic stays on 127.0.0.1 — no data ever leaves the machine.
"""
import time
from dataclasses import dataclass
from typing import List, Optional

import httpx

from app.config import settings


@dataclass
class GenerateResult:
    response_text: str
    latency_ms: float
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    error: Optional[str] = None

    @property
    def tokens_per_second(self) -> Optional[float]:
        if self.completion_tokens and self.latency_ms:
            return self.completion_tokens / (self.latency_ms / 1000)
        return None


class OllamaClient:
    def __init__(
        self,
        base_url: str = settings.ollama_base_url,
        keep_alive: int = settings.keep_alive,
        timeout: float = settings.ollama_timeout,
    ) -> None:
        self.base_url = base_url
        # keep_alive=0 ensures the model is evicted from VRAM after each call.
        self.keep_alive = keep_alive
        self.timeout = timeout

    async def list_models(self) -> List[str]:
        """Return names of all locally available Ollama models."""
        async with httpx.AsyncClient(timeout=self.timeout, proxy=None) as client:
            resp = await client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]

    async def generate(
        self,
        model: str,
        prompt: str,
        images: Optional[List[str]] = None,
    ) -> GenerateResult:
        """
        Send a non-streaming generation request to Ollama.

        Args:
            model:  Ollama model name, e.g. "llama3" or "llava:13b".
            prompt: Text prompt.
            images: Optional list of raw base64-encoded image strings (no
                    data-URL prefix). Passed directly to Ollama's ``images``
                    field — required for vision models such as llava / moondream.

        keep_alive=0 releases VRAM immediately upon completion.
        Timeout is set generously (10 min) for slow/large models.
        """
        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout, proxy=None) as client:
            payload: dict = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": self.keep_alive,
                "options": {
                    "num_ctx": 4096  # 限制模型最多只能使用 4096 个 Token 的上下文窗口
                }
            }
            if images:
                payload["images"] = images

            try:
                resp = await client.post(
                    f"{self.base_url}/api/generate", json=payload
                )
                resp.raise_for_status()
                data = resp.json()
                latency_ms = (time.perf_counter() - start) * 1000
                return GenerateResult(
                    response_text=data.get("response", ""),
                    latency_ms=round(latency_ms, 2),
                    prompt_tokens=data.get("prompt_eval_count"),
                    completion_tokens=data.get("eval_count"),
                )
            except Exception as exc:
                latency_ms = (time.perf_counter() - start) * 1000
                return GenerateResult(
                    response_text="",
                    latency_ms=round(latency_ms, 2),
                    prompt_tokens=None,
                    completion_tokens=None,
                    error=str(exc),
                )


# Module-level singleton — imported by routers and the runner service
ollama_client = OllamaClient()
