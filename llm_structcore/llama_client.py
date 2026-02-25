from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class LlamaChatConfig:
    base_url: str
    model: str
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2048
    timeout_s: int = 600
    json_mode: bool = False  # best-effort: send response_format={"type":"json_object"}


def _chat_url(base_url: str) -> str:
    u = base_url.rstrip("/")
    if u.endswith("/v1"):
        return f"{u}/chat/completions"
    if u.endswith("/v1/chat/completions"):
        return u
    return f"{u}/v1/chat/completions"


def chat_completions(cfg: LlamaChatConfig, messages: list[dict[str, str]]) -> str:
    url = _chat_url(cfg.base_url)

    payload: dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "max_tokens": cfg.max_tokens,
        "stream": False,
    }
    if cfg.json_mode:
        payload["response_format"] = {"type": "json_object"}

    r = requests.post(url, json=payload, timeout=cfg.timeout_s)
    if r.status_code >= 400 and cfg.json_mode:
        # Some llama.cpp OpenAI-compatible servers reject unknown fields like `response_format`.
        # Best-effort fallback: retry once without JSON mode.
        body = (r.text or "")[:4000].lower()
        if ("response_format" in body) or ("unknown field" in body) or ("unexpected field" in body):
            payload.pop("response_format", None)
            r = requests.post(url, json=payload, timeout=cfg.timeout_s)

    r.raise_for_status()
    data = r.json()

    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Unexpected response schema from llama-server: {json.dumps(data)[:2000]}") from e


def list_models(base_url: str, *, timeout_s: int = 20) -> dict:
    u = base_url.rstrip("/")
    if not u.endswith("/v1"):
        u = f"{u}/v1"
    url = f"{u}/models"
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    return r.json()
