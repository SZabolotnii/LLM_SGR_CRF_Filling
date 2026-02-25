#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

_SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = _SCRIPT_PATH.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _detect_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / ".git").exists() or (p / ".env").exists():
            return p
    return start.parents[-1]


REPO_ROOT = _detect_repo_root(_SCRIPT_PATH)
sys.path.insert(0, str(REPO_ROOT))

try:
    from openai import OpenAI  # type: ignore

    _OPENAI_AVAILABLE = True
except Exception:
    OpenAI = None  # type: ignore
    _OPENAI_AVAILABLE = False

try:
    import anthropic  # type: ignore

    _ANTHROPIC_AVAILABLE = True
except Exception:
    anthropic = None  # type: ignore
    _ANTHROPIC_AVAILABLE = False

from llm_structcore.stage1_sgr import (  # noqa: E402
    STAGE1_KEYS_7,
    STAGE1_KEYS_9,
    build_stage1_system_prompt,
    coerce_stage1_summary,
    extract_json_object,
    has_thinking_leak,
)

try:
    from llm_structcore.stage2_items import CRF_ITEMS_BY_CLUSTER  # noqa: E402

    _CRF_ITEM_ALLOWLIST_LOWER: set[str] = {
        str(item).strip().lower()
        for cluster_items in (CRF_ITEMS_BY_CLUSTER or {}).values()
        for item in (cluster_items or [])
        if str(item).strip()
    }
    _CRF_ITEMS_BY_CLUSTER_LOWER: dict[str, set[str]] = {
        str(cluster).strip(): {str(item).strip().lower() for item in (items or []) if str(item).strip()}
        for cluster, items in (CRF_ITEMS_BY_CLUSTER or {}).items()
        if str(cluster).strip()
    }
except Exception:
    _CRF_ITEM_ALLOWLIST_LOWER = set()
    _CRF_ITEMS_BY_CLUSTER_LOWER = {}

_UNKNOWN_VALUE_VARIANTS = {
    "",
    "unknown",
    "not stated",
    "n/a",
    "na",
    "none",
    "null",
    "not specified",
    "not mentioned",
    "not reported",
    "not available",
}

_LINE_KV_RE = re.compile(r"^\s*([^:]{1,160})\s*:\s*(.*?)\s*$")

_CLUSTER_LINE_CAP: dict[str, int] = {
    "DEMOGRAPHICS": 16,
    "VITALS": 18,
    "LABS": 36,
    "PROBLEMS": 24,
    "SYMPTOMS": 24,
    "MEDICATIONS": 20,
    "PROCEDURES": 24,
    "UTILIZATION": 16,
    "DISPOSITION": 12,
}

_APPENDIX_VITALS_KEYS_LOWER = {
    "level of consciousness",
    "respiratory rate",
    "heart rate",
    "blood pressure",
    "body temperature",
    "spo2",
}

_APPENDIX_LABS_KEYS_LOWER = {
    "ph",
    "pa02",
    "pac02",
    "hc03-",
    "lactates",
    "hemoglobin",
    "platelets",
    "leukocytes",
    "c-reactive protein",
    "blood sodium",
    "blood potassium",
    "blood glucose",
    "creatinine",
    "transaminases",
    "inr",
    "troponin",
    "bnp or nt-pro-bnp",
    "d-dimer",
}

_NEGATION_LEFT_CONTEXT_CUES = (
    "no ",
    "denies",
    "without",
    "not ",
    # IT
    "nega",
    "negato",
    "senza",
    "non ",
)


def _trim_text(text: str, *, max_chars: int, strategy: str) -> str:
    if max_chars <= 0:
        return text
    s = text or ""
    if len(s) <= max_chars:
        return s
    if strategy == "head":
        return s[:max_chars]
    if strategy == "tail":
        return s[-max_chars:]
    if strategy == "head_tail":
        half = max_chars // 2
        return s[:half] + "\n...\n" + s[-(max_chars - half) :]
    # default: middle
    start = max(0, (len(s) - max_chars) // 2)
    return s[start : start + max_chars]


def _merge_stage1_summaries(summaries: list[dict[str, str]], keys: list[str]) -> dict[str, str]:
    """
    Multi-slice merge:
    - Union "Key: Value" lines across slices per cluster
    - Dedupe case-insensitively
    - Keep sparse contract: "not stated" if empty after merge
    - Re-sanitize at the end (caps + sparse)
    """
    merged: dict[str, str] = {k: "not stated" for k in keys}
    per_cluster_lines: dict[str, list[str]] = {k: [] for k in keys}

    for s in summaries:
        for cluster in keys:
            raw = str((s or {}).get(cluster, "") or "").strip()
            if not raw or raw.lower() in _UNKNOWN_VALUE_VARIANTS:
                continue
            for line in raw.splitlines():
                ln = " ".join(line.strip().split())
                if not ln:
                    continue
                per_cluster_lines[cluster].append(ln)

    for cluster in keys:
        seen: set[str] = set()
        out_lines: list[str] = []
        for ln in per_cluster_lines[cluster]:
            norm = ln.lower()
            if norm in seen:
                continue
            seen.add(norm)
            out_lines.append(ln)
        merged[cluster] = "\n".join(out_lines).strip() if out_lines else "not stated"

    return _sanitize_stage1_summary(merged, keys)


def _inject_appendix_lines(summary: dict[str, str], appendix_lines: list[str], keys: list[str]) -> dict[str, str]:
    if not appendix_lines:
        return summary

    out = dict(summary)

    def append_line(cluster: str, line: str) -> None:
        if cluster not in out:
            return
        s = out.get(cluster, "not stated").strip()
        if s.lower() in _UNKNOWN_VALUE_VARIANTS:
            s = ""
        existing = {x.strip().lower() for x in s.splitlines() if x.strip()}
        if line.strip().lower() in existing:
            return
        out[cluster] = (s + ("\n" if s else "") + line.strip()).strip() if line.strip() else (s or "not stated")

    for line in appendix_lines:
        m = _LINE_KV_RE.match(str(line).strip())
        if not m:
            continue
        k = m.group(1).strip()
        if not k:
            continue
        kl = k.lower()
        v = m.group(2).strip()
        if not v:
            continue

        # Preserve existing routing heuristics first.
        if "MEDICATIONS" in keys and (kl.startswith("medication") or kl.startswith("home med") or kl.startswith("home medication")):
            append_line("MEDICATIONS", f"{k}: {v}")
            continue
        if kl in _APPENDIX_VITALS_KEYS_LOWER and "VITALS" in keys:
            append_line("VITALS", f"{k}: {v}")
            continue
        if kl in _APPENDIX_LABS_KEYS_LOWER and "LABS" in keys:
            append_line("LABS", f"{k}: {v}")
            continue
        if "PROCEDURES" in keys and ("," in kl or "abnormal" in kl or "scan" in kl or "rx" in kl or "ultrasound" in kl):
            append_line("PROCEDURES", f"{k}: {v}")
            continue

        # If the appendix line uses an exact CRF item name, route it to the correct Stage1 cluster.
        # This matters because Stage2 builds a doc-scoped ontology from "present clusters" (non-"not stated" values).
        routed = False
        for cluster in (
            "VITALS",
            "LABS",
            "PROBLEMS",
            "SYMPTOMS",
            "MEDICATIONS",
            "PROCEDURES",
            "UTILIZATION",
            "DISPOSITION",
            "DEMOGRAPHICS",
        ):
            if cluster not in keys:
                continue
            if kl in _CRF_ITEMS_BY_CLUSTER_LOWER.get(cluster, set()):
                append_line(cluster, f"{k}: {v}")
                routed = True
                break
        if routed:
            continue

        if "DEMOGRAPHICS" in keys:
            append_line("DEMOGRAPHICS", f"{k}: {v}")

    # Re-sanitize after injection to enforce caps + sparse rule.
    return _sanitize_stage1_summary(out, keys)


def _sanitize_cluster_value(cluster: str, raw: str) -> str:
    """
    Enforce Stage1 contract for downstream Stage2:
    - value is a short string with "Key: Value" lines only
    - drop per-item "not stated"/"unknown" lines (sparse)
    - dedupe + cap lines to avoid truncation loops
    """
    s = (raw or "").strip()
    if s.lower() in _UNKNOWN_VALUE_VARIANTS:
        return "not stated"

    seen: set[str] = set()
    out_lines: list[str] = []
    for line in s.splitlines():
        ln = " ".join(line.strip().split())
        if not ln:
            continue
        # Strip bullets commonly emitted by LLMs.
        ln2 = re.sub(r"^[-*•]\s*", "", ln).strip()

        m = _LINE_KV_RE.match(ln2)
        if m:
            k = " ".join(m.group(1).strip().split())
            v = " ".join(m.group(2).strip().split())
        else:
            # Recovery path: accept bare CRF item names as positives.
            low = ln2.lower().strip().rstrip(".;")
            if low in _CRF_ITEM_ALLOWLIST_LOWER and cluster in {"PROBLEMS", "SYMPTOMS", "MEDICATIONS", "PROCEDURES", "DEMOGRAPHICS", "DISPOSITION"}:
                k = ln2.strip().rstrip(".;")
                v = "present"
            else:
                # Recovery path: "Key Value" for objective clusters.
                m2 = re.match(r"^([A-Za-z][A-Za-z0-9+/%()._-]{0,40})\s+(.{1,80})$", ln2)
                if m2 and cluster in {"VITALS", "LABS"} and re.search(r"[0-9]|%|/", m2.group(2)):
                    k = m2.group(1).strip()
                    v = m2.group(2).strip()
                else:
                    # Recovery path: "item present/absent/yes/no"
                    m3 = re.match(r"^(.{1,120})\s+(present|absent|yes|no|pos|neg|unknown)$", ln2, flags=re.IGNORECASE)
                    if m3 and m3.group(1).strip():
                        k = m3.group(1).strip()
                        v = m3.group(2).strip()
                    else:
                        continue

        if not k or not v:
            continue
        if v.lower() in _UNKNOWN_VALUE_VARIANTS:
            continue
        if "not stated" in v.lower() or v.lower() == "unknown":
            continue
        norm = f"{k}: {v}"
        if norm.lower() in seen:
            continue
        seen.add(norm.lower())
        out_lines.append(norm)

    cap = int(_CLUSTER_LINE_CAP.get(cluster, 18))
    out_lines = out_lines[:cap]
    return "\n".join(out_lines).strip() if out_lines else "not stated"


def _sanitize_stage1_summary(summary: dict[str, str], keys: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k in keys:
        out[k] = _sanitize_cluster_value(k, str(summary.get(k, "") or ""))
    return out


def _is_stage1_output_acceptable(summary: dict[str, str], keys: list[str]) -> bool:
    if any(k not in summary for k in keys):
        return False
    if any(not isinstance(summary.get(k), str) for k in keys):
        return False
    joined = "\n".join(summary.get(k, "") for k in keys)
    if joined.count(": not stated") >= 3:
        return False
    if len(joined) > 15000:
        return False
    if all((summary.get(k, "").strip().lower() == "not stated") for k in keys):
        return False
    return True


def _hf_rows(dataset: str, *, split: str, offset: int, length: int) -> dict[str, Any]:
    """
    Fetch rows from HF datasets-server.

    Note: /rows endpoint has a max `length` (commonly ~100). For larger `length`, we page.
    """
    length_i = int(length)
    offset_i = int(offset)
    if length_i <= 0:
        return {"rows": []}

    page_size = 100
    out_rows: list[dict[str, Any]] = []
    remaining = length_i
    while remaining > 0:
        take = min(page_size, remaining)
        qs = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": "default",
                "split": split,
                "offset": str(offset_i),
                "length": str(take),
            }
        )
        url = f"https://datasets-server.huggingface.co/rows?{qs}"
        req = urllib.request.Request(url, headers={"User-Agent": "medgemma-crf-stage1-audit/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode("utf-8"))
        batch = data.get("rows") or []
        out_rows.extend(batch)
        if len(batch) < take:
            break
        offset_i += take
        remaining -= take

    return {"rows": out_rows}


def _post_json(url: str, payload: dict[str, Any], *, timeout_s: int) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "medgemma-crf-stage1-audit/1.0"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from llama-server: {raw[:2000]}") from e


def _chat_url(base_url: str) -> str:
    u = base_url.rstrip("/")
    if u.endswith("/v1"):
        return f"{u}/chat/completions"
    if u.endswith("/v1/chat/completions"):
        return u
    return f"{u}/v1/chat/completions"


def _get_gemini_api_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("GEMINI_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass
    return None


def _get_nvidia_api_key() -> str | None:
    key = os.environ.get("NVIDIA_API_KEY")
    if key:
        return key
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("NVIDIA_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass
    return None


def _get_openai_api_key() -> str | None:
    # Preferred: explicit repo env var
    key = os.environ.get("GPT_API_KEY_OPEN_AI")
    if key:
        return key.strip()

    # Fallback: standard env var (if user already set it)
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key.strip()

    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("GPT_API_KEY_OPEN_AI="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
                if line.startswith("OPENAI_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass
    return None


def _get_anthropic_api_key() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key.strip()

    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass
    return None


def _get_gcp_service_account_b64() -> str | None:
    sa = os.environ.get("GCP_SERVICE_ACCOUNT_B64")
    if sa:
        return sa.strip().strip('"').strip("'")
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("GCP_SERVICE_ACCOUNT_B64="):
                    val = line.split("=", 1)[1].strip()
                    return val.strip().strip('"').strip("'")
        except Exception:
            pass
    return None


def _openai_generate(
    *,
    api_key: str,
    model: str,
    system_instruction: str,
    user_text: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_s: float,
    json_mode: bool,
    verbosity: str,
    reasoning_effort: str,
    api_retries: int,
    retry_sleep_s: float,
) -> str:
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("openai package not available; install openai>=1.0.0")
    client = OpenAI(api_key=api_key)  # type: ignore

    response_format: dict[str, Any] = {"type": "json_object"} if bool(json_mode) else {"type": "text"}

    last_err: Exception | None = None
    for attempt in range(max(1, int(api_retries))):
        try:
            kwargs: dict[str, Any] = {
                "model": str(model),
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_text},
                ],
                "response_format": response_format,
                "verbosity": str(verbosity),
                "store": False,
            }
            # Avoid passing unsupported knobs if user disabled them.
            if str(reasoning_effort).strip().lower() not in {"", "none", "off", "false", "0"}:
                kwargs["reasoning_effort"] = str(reasoning_effort)

            kwargs["temperature"] = float(temperature)
            kwargs["top_p"] = float(top_p)
            # Some newer models (e.g., gpt-5.x) require max_completion_tokens.
            kwargs["max_completion_tokens"] = int(max_tokens)

            resp = client.chat.completions.create(timeout=float(timeout_s), **kwargs)
            return str(resp.choices[0].message.content or "")
        except Exception as e:
            last_err = e
            time.sleep(max(float(retry_sleep_s), (2**attempt) * 1.5))
            continue

    raise RuntimeError(f"OpenAI API failed after retries: {last_err}") from last_err


def _anthropic_generate(
    *,
    api_key: str,
    model: str,
    system_instruction: str,
    user_text: str,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
    effort: str,
    api_retries: int,
    retry_sleep_s: float,
) -> str:
    if not _ANTHROPIC_AVAILABLE:
        raise RuntimeError("anthropic package not available; install anthropic")
    # Anthropic SDK configures request timeout at the client level.
    client = anthropic.Anthropic(api_key=api_key, timeout=float(timeout_s))  # type: ignore

    last_err: Exception | None = None
    for attempt in range(max(1, int(api_retries))):
        try:
            kwargs: dict[str, Any] = {
                "model": str(model),
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
                "system": str(system_instruction),
                "messages": [{"role": "user", "content": str(user_text)}],
            }
            eff = str(effort or "").strip().lower()
            if eff and eff not in {"none", "off", "false", "0"}:
                kwargs["output_config"] = {"effort": str(eff)}
            try:
                msg = client.messages.create(**kwargs)
            except TypeError as e:
                # Some anthropic SDK versions don't support output_config; retry without it.
                if "output_config" in kwargs and "output_config" in str(e):
                    kwargs.pop("output_config", None)
                    msg = client.messages.create(**kwargs)
                else:
                    raise
            # msg.content is a list of content blocks; join text blocks.
            blocks = getattr(msg, "content", None) or []
            parts: list[str] = []
            for b in blocks:
                t = getattr(b, "text", None)
                if isinstance(t, str) and t:
                    parts.append(t)
            return "\n".join(parts).strip()
        except Exception as e:
            last_err = e
            time.sleep(max(float(retry_sleep_s), (2**attempt) * 2.0))
            continue

    raise RuntimeError(f"Anthropic API failed after retries: {last_err}") from last_err


def _vertex_gemini_generate(
    *,
    service_account_b64: str,
    model: str,
    location: str,
    system_instruction: str,
    user_text: str,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    api_retries: int,
    retry_sleep_s: float,
) -> str:
    import base64

    import vertexai
    from google.oauth2 import service_account
    from vertexai.generative_models import GenerativeModel

    sa_json = base64.b64decode(service_account_b64).decode("utf-8")
    sa_info = json.loads(sa_json)
    project_id = str(sa_info.get("project_id") or "").strip()
    if not project_id:
        raise ValueError("Service account JSON has no project_id.")

    creds = service_account.Credentials.from_service_account_info(
        sa_info,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    vertexai.init(project=project_id, location=str(location), credentials=creds)

    models = [m.strip() for m in str(model).split(",") if m.strip()]
    if not models:
        raise ValueError("gemini_model resolved to empty list.")

    last_err: Exception | None = None
    for attempt in range(max(1, int(api_retries))):
        model_id = models[min(attempt, len(models) - 1)]
        try:
            gm = GenerativeModel(model_id, system_instruction=system_instruction)
            resp = gm.generate_content(
                user_text,
                generation_config={
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "max_output_tokens": int(max_output_tokens),
                    "response_mime_type": "application/json",
                },
            )
            return str(getattr(resp, "text", "") or "")
        except Exception as e:
            last_err = e
            msg = str(e)
            if any(code in msg for code in ("429", "503")):
                time.sleep(max(float(retry_sleep_s), (2**attempt) * 2.0))
                continue
            time.sleep(max(float(retry_sleep_s), (2**attempt) * 1.0))
            continue

    raise RuntimeError(f"Vertex Gemini API failed after retries: {last_err}") from last_err


def _gemini_generate(
    *,
    api_key: str,
    model: str,
    system_instruction: str,
    user_text: str,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    thinking_budget: int,
    api_retries: int,
    retry_sleep_s: float,
) -> str:
    # Optional dependency (teacher mode only)
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore
    from google.genai import errors  # type: ignore

    client = genai.Client(api_key=api_key)
    models = [m.strip() for m in str(model).split(",") if m.strip()]
    if not models:
        raise ValueError("gemini_model resolved to empty list.")

    last_err: Exception | None = None
    for i in range(max(1, int(api_retries))):
        model_id = models[min(i, len(models) - 1)]
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=user_text)],
                    )
                ],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    max_output_tokens=int(max_output_tokens),
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(thinking_budget=int(thinking_budget)),
                ),
            )
            return str(response.text or "")
        except (errors.ServerError, errors.ClientError) as e:
            # Common transient conditions: 503 (high demand), 429 (rate limit).
            last_err = e
            code = getattr(e, "status_code", None) or getattr(e, "code", None)
            if code in (429, 503):
                # Try to honor suggested retry delay if present in the message.
                msg = str(e)
                m = re.search(r"retry in\\s+([0-9]+(?:\\.[0-9]+)?)s", msg, flags=re.IGNORECASE)
                delay = float(m.group(1)) if m else 0.0
                time.sleep(max(float(retry_sleep_s), delay))
                continue
            raise

    raise RuntimeError(f"Gemini API failed after retries: {last_err}") from last_err


def _nvidia_generate(
    *,
    api_key: str,
    invoke_url: str,
    model: str,
    system_instruction: str,
    user_text: str,
    temperature: float,
    top_p: float,
    top_k: int | None,
    presence_penalty: float | None,
    repetition_penalty: float | None,
    enable_thinking: bool,
    max_tokens: int,
    timeout_s: float,
    api_retries: int,
    retry_sleep_s: float,
) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system_instruction}, {"role": "user", "content": user_text}],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "stream": False,
    }
    if top_k is not None:
        payload["top_k"] = int(top_k)
    if presence_penalty is not None:
        payload["presence_penalty"] = float(presence_penalty)
    if repetition_penalty is not None:
        payload["repetition_penalty"] = float(repetition_penalty)
    if enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": True}

    # NVIDIA structured output (response_format=json_object) is backed by "guidance" for some models.
    # Mistral-family models currently fail with a 400 error when using that backend.
    # We therefore only request response_format when it is known to work, and we also auto-recover
    # by retrying once without response_format on a matching 400 error.
    model_l = str(model or "").lower()
    mistral_family = any(
        token in model_l
        for token in (
            "mistralai/",
            "nv-mistralai/",
            "/mistral",
            "/mixtral",
            "/codestral",
            "/mathstral",
            "/ministral",
            "/devstral",
            "/magistral",
        )
    )
    if not mistral_family:
        payload["response_format"] = {"type": "json_object"}

    last_err: Exception | None = None
    for attempt in range(max(1, int(api_retries))):
        try:
            r = requests.post(str(invoke_url), headers=headers, json=payload, timeout=float(timeout_s))
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After", "").strip()
                delay = 0.0
                try:
                    delay = float(retry_after) if retry_after else 0.0
                except ValueError:
                    delay = 0.0
                time.sleep(max(float(retry_sleep_s), delay, (2**attempt) * 5.0))
                continue
            if r.status_code == 400 and "response_format" in payload:
                # Auto-recover from known tokenizer/guidance incompatibility by retrying
                # once without response_format.
                msg = (r.text or "").lower()
                if "tokenizer" in msg and "structured output" in msg and "guidance" in msg:
                    payload.pop("response_format", None)
                    time.sleep(max(float(retry_sleep_s), 1.0))
                    continue
            r.raise_for_status()
            data = r.json()
            return str(data["choices"][0]["message"]["content"] or "")
        except Exception as e:
            last_err = e
            time.sleep(max(float(retry_sleep_s), (2**attempt) * 2.0))
            continue

    raise RuntimeError(f"NVIDIA API failed after retries: {last_err}") from last_err


def _chat(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout_s: int,
    json_mode: bool,
    repeat_penalty: float,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        "stream": False,
        # Encourage clean termination right after the JSON object.
        # The stop sequence is removed from the output, leaving valid JSON.
        "stop": ["\n\n", "\r\n\r\n"],
        "repeat_penalty": float(repeat_penalty),
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    url = _chat_url(base_url)
    try:
        data = _post_json(url, payload, timeout_s=timeout_s)
    except RuntimeError as e:
        # best-effort retry without response_format if server rejects it
        if json_mode and any(tok in str(e).lower() for tok in ("response_format", "unknown field", "unexpected field")):
            payload.pop("response_format", None)
            data = _post_json(url, payload, timeout_s=timeout_s)
        else:
            raise

    try:
        return str(data["choices"][0]["message"]["content"])
    except Exception as e:
        raise RuntimeError(f"Unexpected llama-server response schema: {json.dumps(data)[:2000]}") from e


def _build_objective_appendix(note: str) -> list[str]:
    """
    Deterministic hint extraction from the raw note to reduce Stage1 false negatives.
    Only emits high-precision facts (avoid guessing).
    """
    t = (note or "")
    # Normalize common unicode subscripts used in clinical notes (e.g., SpO₂, pO₂).
    t = t.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
    low = t.lower()
    out: list[str] = []

    def add(line: str) -> None:
        s = " ".join(str(line).strip().split())
        if not s:
            return
        if s not in out:
            out.append(s)

    # ── Level of consciousness (AVPU) via GCS
    m_gcs = re.search(r"(?i)\b(?:gcs|glasgow)\s*[:=]?\s*(\d{1,2})\b", t)
    if m_gcs:
        try:
            gcs = int(m_gcs.group(1))
            if gcs >= 15:
                add("level of consciousness: A")
            elif gcs >= 9:
                add("level of consciousness: V")
            else:
                add("level of consciousness: P")
        except Exception:
            pass
    else:
        # Text-only cues (conservative)
        if any(x in low for x in ("unresponsive", "coma")):
            add("level of consciousness: P")
        elif any(x in low for x in ("drowsy", "confused", "disoriented")):
            add("level of consciousness: V")
        elif any(x in low for x in ("alert", "awake", "oriented", "a&o")):
            add("level of consciousness: A")

    # ── Respiratory rate (numeric)
    m_rr = re.search(r"(?i)\b(?:rr|rf|fr|respiratory rate)\s*[:=]?\s*(\d{1,3})\b", t)
    if m_rr:
        add(f"respiratory rate: {m_rr.group(1)}")
    else:
        # Text-only cues
        if "tachyp" in low or "tachypnoe" in low or "tachypne" in low:
            add("respiratory rate: tachypneic")
        elif "bradyp" in low or "bradypnoe" in low or "bradypne" in low:
            add("respiratory rate: bradypneic")
        elif "eupno" in low:
            add("respiratory rate: eupneic")

    # ── Allergy (explicit negations)
    if "nkda" in low or "no known allergies" in low or re.search(r"(?i)\ballerg(?:y|ies)\s*:\s*(none|no)\b", t):
        add("history of allergy: no")
    elif "allerg" in low:
        add("history of allergy: yes")

    # ── Active neoplasia
    # Avoid deterministic injection here: a naive keyword match creates false positives and can
    # be misinterpreted downstream as "evidence". We rely on Stage1 model + Stage2 evidence gates.

    # ── Procedures abnormality (high-precision keyword heuristics)
    if "ecg" in low or "ekg" in low:
        if any(x in low for x in ("st elevation", "st depression", "t negative", "t-wave inversion", "t wave inversion", "poor r-wave", "abnormal ecg")):
            add("ecg, any abnormality: present")
        elif "normal ecg" in low:
            add("ecg, any abnormality: absent")

    if "chest x-ray" in low or "chest xray" in low or "chest rx" in low or "cxr" in low:
        if any(x in low for x in ("congestion", "infiltrat", "opacity", "edema", "pulmonary congestion", "vascular thickening", "pneumonia")):
            add("chest rx, any abnormalities: present")
        elif "normal chest x-ray" in low or "normal cxr" in low:
            add("chest rx, any abnormalities: absent")

    if "echocardiogram" in low or re.search(r"(?i)\becho\b", t):
        if any(x in low for x in ("hypokines", "regurg", "stenos", "ef ", "ejection fraction", "mr", "tr", "valv", "dilat")):
            add("cardiac ultrasound, any abnormality: present")

    # ── High-precision objective vitals/labs (label -> number)
    def affirmed_term(term: str) -> bool:
        tlow = term.lower().strip()
        if not tlow:
            return False
        for m in re.finditer(re.escape(tlow), low):
            s = max(0, m.start() - 40)
            ctx = low[s : m.start()]
            if any(neg in ctx for neg in _NEGATION_LEFT_CONTEXT_CUES):
                continue
            return True
        return False

    # BP
    for m_bp in re.finditer(r"(?i)\b(?:bp|blood pressure)\s*[:=]?\s*(\d{2,3})\s*/\s*(\d{2,3})\b", t):
        add(f"blood pressure: {m_bp.group(1)}/{m_bp.group(2)} mmHg")
        break
    # HR
    for m_hr in re.finditer(r"(?i)\b(?:hr|heart rate)\s*[:=]?\s*(\d{2,3})\b", t):
        add(f"heart rate: {m_hr.group(1)}")
        break
    # SpO2
    for m_sp in re.finditer(r"(?i)\b(?:spo2|sp02|so2|s02|sat02|sato2|o2 sat(?:uration)?)\s*[:=]?\s*(\d{2,3})\s*%?", t):
        add(f"spo2: {m_sp.group(1)}%")
        break
    # Temperature
    for m_temp in re.finditer(r"(?i)\b(?:temp|temperature|t)\s*[:=]?\s*(\d{2}(?:[.,]\d)?)\s*(?:°?c|c\b)", t):
        add(f"body temperature: {m_temp.group(1).replace(',', '.')}")
        break
    if (
        "afebrile" in low
        or "apyretic" in low
        or "no fever" in low
        or "without fever" in low
        or "senza febbre" in low  # IT
        or "afebbrile" in low  # IT common misspelling
    ):
        add("body temperature: afebrile")

    def add_lab(label: str, canon_key: str, pat: str) -> None:
        for m in re.finditer(pat, t):
            val = m.group(1).strip()
            if not val:
                continue
            add(f"{canon_key}: {val}")
            break

    add_lab("ph", "ph", r"(?i)\bph\s*[:=]?\s*([-+]?\d+(?:[.,]\d+)?)\b")
    add_lab("pco2", "pac02", r"(?i)\b(?:pco2|paco2)\s*[:=]?\s*([-+]?\d+(?:[.,]\d+)?)\b")
    add_lab("po2", "pa02", r"(?i)\b(?:po2|pao2)\s*[:=]?\s*([-+]?\d+(?:[.,]\d+)?)\b")
    add_lab("hco3", "hc03-", r"(?i)\b(?:hco3-?|bicarb(?:onate)?)\s*[:=]?\s*([-+]?\d+(?:[.,]\d+)?)\b")
    add_lab("lactate", "lactates", r"(?i)\b(?:lac|lactate(?:s)?)\s*[:=]?\s*([-+]?\d+(?:[.,]\d+)?)\b")
    add_lab("na", "blood sodium", r"(?i)\b(?:na\+?|sodium)\s*[:=]?\s*([-+]?\d{2,3}(?:[.,]\d+)?)\b")
    add_lab("k", "blood potassium", r"(?i)\b(?:k\+?|potassium)\s*[:=]?\s*([-+]?\d+(?:[.,]\d+)?)\b")
    add_lab("glucose", "blood glucose", r"(?i)\b(?:glucose|glycemia)\s*[:=]?\s*([-+]?\d+(?:[.,]\d+)?)\b")
    add_lab("creatinine", "creatinine", r"(?i)\b(?:creatinine|creat|cr)\s*[:=]?\s*([-+]?\d+(?:[.,]\d+)?)\b")
    add_lab("wbc", "leukocytes", r"(?i)\b(?:wbc|leukocytes)\s*[:=]?\s*([-+]?\d+(?:[.,]\d+)?)\b")
    add_lab("hb", "hemoglobin", r"(?i)\b(?:hb|hgb|hemoglobin)\s*[:=]?\s*([-+]?\d+(?:[.,]\d+)?)\b")
    add_lab("plt", "platelets", r"(?i)\b(?:plt|platelets)\s*[:=]?\s*([-+]?\d+(?:[.,]\d+)?)\b")
    add_lab("crp", "c-reactive protein", r"(?i)\b(?:crp|pcr)\s*[:=]?\s*([-+]?\d+(?:[.,]\d+)?)\b")
    add_lab("inr", "inr", r"(?i)\binr\s*[:=]?\s*([-+]?\d+(?:[.,]\d+)?)\b")
    add_lab("troponin", "troponin", r"(?i)\b(?:troponin|trop|tnt)\s*[:=]?\s*([-+]?\d+(?:[.,]\d+)?)\b")

    # ── Demographics / context (high-precision section line capture)
    # Capture inline section fragments like "Past Medical History: ... Therapy: ..."
    low_t = low
    pmh_idx = low_t.find("past medical history")
    if pmh_idx != -1:
        frag = t[pmh_idx : pmh_idx + 600]
        m = re.search(r"(?is)past medical history\s*:\s*(.{1,600})", frag)
        if m:
            pmh = m.group(1)
            pmh = re.split(r"(?i)\b(?:therapy|home meds|home medications|medications|allerg(?:y|ies))\s*:", pmh, maxsplit=1)[0]
            pmh = re.sub(r"\s+", " ", pmh).strip(" -;,.")
            if pmh:
                add(f"PMH: {pmh}")

    therapy_idx = low_t.find("therapy")
    if therapy_idx != -1:
        frag = t[therapy_idx : therapy_idx + 800]
        m = re.search(r"(?is)therapy\s*:\s*(.{1,800})", frag)
        if m:
            meds_raw = m.group(1)
            meds_raw = re.split(r"(?i)\b(?:allerg(?:y|ies)|physical exam|vitals|examination)\b\s*:", meds_raw, maxsplit=1)[0]
            meds_raw = re.sub(r"\s+", " ", meds_raw).strip()
            if meds_raw:
                # Keep an aggregate line for readability
                add(f"Home meds: {meds_raw[:220]}")
                # Also emit per-med lines for robust downstream heuristics.
                parts = [p.strip() for p in re.split(r"[,;]", meds_raw) if p.strip()]
                meds: list[str] = []
                for p2 in parts:
                    m2 = re.match(r"^([A-Za-z][A-Za-z0-9-]{1,24})\b", p2)
                    if m2:
                        name = m2.group(1).strip()
                        if name and name.lower() not in {x.lower() for x in meds}:
                            meds.append(name)
                for name in meds[:12]:
                    add(f"medication: {name}")

    # Alternative meds section marker used in some notes: "Tx:"
    m_tx = re.search(r"(?is)\btx\b\s*:\s*(.{1,600})", t)
    if m_tx:
        tx_raw = m_tx.group(1)
        tx_raw = re.split(r"(?i)\b(?:allerg(?:y|ies)|physical exam|vitals|examination|at the time)\b\s*:", tx_raw, maxsplit=1)[0]
        tx_raw = re.sub(r"\s+", " ", tx_raw).strip()
        if tx_raw:
            add(f"Home meds: {tx_raw[:220]}")
            parts = [p.strip() for p in re.split(r"[,;]", tx_raw) if p.strip()]
            meds: list[str] = []
            for p2 in parts:
                m2 = re.match(r"^([A-Za-z][A-Za-z0-9-]{1,24})\b", p2)
                if m2:
                    name = m2.group(1).strip()
                    if name and name.lower() not in {x.lower() for x in meds}:
                        meds.append(name)
            for name in meds[:12]:
                add(f"medication: {name}")

    for m in re.finditer(r"(?i)\bsocial\s*:\s*([^\n]{1,220})", t):
        add(f"Social: {m.group(1).strip()}")
        break

    # ── Key symptom (high FN): dyspnea / dispnea / SOB
    # Conservative: only add when not negated in a small left-context window.
    dyspnea_terms = [
        "dyspnea",
        "dyspnoea",
        "shortness of breath",
        "sob",
        "short of breath",
        "difficulty breathing",
        "labored breathing",
        "breathless",
        "dyspneic",
        "dispnea",  # IT
        "mancanza di respiro",  # IT
        "affanno",  # IT
        "fiato corto",  # IT
    ]
    if any(affirmed_term(term) for term in dyspnea_terms):
        add("presence of dyspnea: present")

    def affirmed_regex(pat: str) -> bool:
        if not pat:
            return False
        for m in re.finditer(pat, low, flags=re.IGNORECASE):
            s = max(0, m.start() - 40)
            ctx = low[s : m.start()]
            if any(neg in ctx for neg in _NEGATION_LEFT_CONTEXT_CUES):
                continue
            return True
        return False

    # ── High-FN symptom flags (conservative)
    if affirmed_regex(r"\b(?:agitat(?:ed|ion)?|agitazione|agitato|agitata|agitati|agitate)\b"):
        add("agitation: present")

    # Foreign body in airways: require strong mention to avoid FP from generic aspiration/pneumonia.
    foreign_body_airway = any(
        affirmed_regex(p)
        for p in (
            r"\bforeign body\b.{0,80}\b(?:airway|trache|bronch|aspirat|inhal)\b",
            r"\b(?:airway|trache|bronch)\b.{0,80}\bforeign body\b",
            r"\bcorpo estraneo\b.{0,80}\b(?:vie aeree|trache|bronch|aspirat|inal)\b",  # IT
            r"\b(?:vie aeree|trache|bronch)\b.{0,80}\bcorpo estraneo\b",  # IT
        )
    )
    if foreign_body_airway:
        add("foreign body in the airways: present")

    return out


def _process_one_doc(
    *,
    rec: dict[str, Any],
    doc_id: str,
    row_i: int,
    args: argparse.Namespace,
    stage1_keys: list[str],
    stage1_sys: str,
    stage1_sys_min: str,
    run_dir: Path,
) -> dict[str, int]:
    """
    Process a single document, write per-doc artifacts, and return stats deltas.
    Safe for ThreadPoolExecutor: outputs are per-doc unique files.
    """
    doc_stats: dict[str, int] = {
        "n": 1,
        "thinking_leak": 0,
        "json_parse_ok": 0,
        "json_parse_fail": 0,
        "missing_keys_total": 0,
        "extra_keys_total": 0,
        "empty_values": 0,
        "retries_used_total": 0,
        "retries_exhausted": 0,
        "reject_nonstring": 0,
        "reject_missing_keys": 0,
        "reject_unacceptable": 0,
    }

    note_full = str(rec.get("clinical_note") or "")
    appendix_lines = _build_objective_appendix(note_full)
    appendix = ""
    if appendix_lines:
        appendix = (
            "\\n\\nCRF Objective Appendix (deterministic):\\n"
            + "\\n".join(appendix_lines)
            + "\\n"
        )

    extra_strats = [s.strip() for s in str(getattr(args, "multi_trim_strategies", "")).split(",") if s.strip()]
    trim_strategies = [str(getattr(args, "trim_strategy", "middle"))] + [
        s for s in extra_strats if s not in {str(getattr(args, "trim_strategy", "middle"))}
    ]

    reminder_lines = [
        "Extraction reminder (high recall, still sparse):",
        "- DEMOGRAPHICS: include PMH/comorbidities if mentioned anywhere (use a line like \"PMH: ...\").",
        "- MEDICATIONS: include home meds / chronic therapy if mentioned (use a line like \"Home meds: ...\").",
        "- Dyspnea/dispnea/SOB: extract ONLY if explicitly mentioned (do not infer).",
        "- Avoid speculative language (possible/likely); omit if uncertain.",
        "- Do NOT write per-item \"not stated\" lines; omit missing facts.",
    ]
    if str(getattr(args, "language", "en")) == "it":
        reminder_lines.append("- The note may be Italian; still output keys in English/canonical terms where possible.")

    per_slice_summaries: list[dict[str, str]] = []
    per_slice_meta: list[dict[str, Any]] = []
    any_thinking_leak = False
    parse_err_any: str | None = None
    missing_keys_any: list[str] = []
    extra_keys_any: list[str] = []

    for strat in trim_strategies:
        if row_i > 0:
            if getattr(args, "engine", "") in ("gemini", "gemini_vertex") and float(getattr(args, "gemini_delay_s", 0.0)) > 0:
                time.sleep(float(getattr(args, "gemini_delay_s", 0.0)))
            if getattr(args, "engine", "") == "nvidia" and float(getattr(args, "nvidia_delay_s", 0.0)) > 0:
                time.sleep(float(getattr(args, "nvidia_delay_s", 0.0)))

        note = _trim_text(note_full, max_chars=int(getattr(args, "max_note_chars", 0)), strategy=strat)
        user_msg = f"Clinical Note:\n{note}{appendix}\n\n" + "\n".join(reminder_lines) + "\n\nJSON Summary:"

        text = ""
        last_err: str | None = None
        retries_used = 0
        sanitized0: dict[str, str] | None = None

        for attempt in range(max(1, int(getattr(args, "retries", 3)))):
            sys_prompt = stage1_sys_min if bool(getattr(args, "min_system_prompt", False)) or attempt > 0 else stage1_sys
            retries_used = attempt + 1
            max_tokens = int(getattr(args, "max_new_tokens", 650))
            if attempt >= 2:
                max_tokens = min(max_tokens, 420)

            if getattr(args, "engine", "") == "gemini":
                api_key = _get_gemini_api_key()
                if not api_key:
                    raise SystemExit("GEMINI_API_KEY not found in environment or .env file.")
                text = _gemini_generate(
                    api_key=api_key,
                    model=str(getattr(args, "gemini_model", "")),
                    system_instruction=sys_prompt,
                    user_text=user_msg,
                    temperature=float(getattr(args, "temperature", 0.0)),
                    top_p=float(getattr(args, "top_p", 1.0)),
                    max_output_tokens=max_tokens,
                    thinking_budget=int(getattr(args, "gemini_thinking_budget", 0)),
                    api_retries=int(getattr(args, "gemini_api_retries", 6)),
                    retry_sleep_s=float(getattr(args, "gemini_retry_sleep_s", 5.0)),
                )
            elif getattr(args, "engine", "") == "gemini_vertex":
                sa_b64 = _get_gcp_service_account_b64()
                if not sa_b64:
                    raise SystemExit("GCP_SERVICE_ACCOUNT_B64 not found in environment or .env file.")
                text = _vertex_gemini_generate(
                    service_account_b64=sa_b64,
                    model=str(getattr(args, "gemini_model", "")),
                    location=str(getattr(args, "gemini_vertex_location", "us-central1")),
                    system_instruction=sys_prompt,
                    user_text=user_msg,
                    temperature=float(getattr(args, "temperature", 0.0)),
                    top_p=float(getattr(args, "top_p", 1.0)),
                    max_output_tokens=max_tokens,
                    api_retries=int(getattr(args, "gemini_api_retries", 6)),
                    retry_sleep_s=float(getattr(args, "gemini_retry_sleep_s", 5.0)),
                )
            elif getattr(args, "engine", "") == "nvidia":
                api_key = _get_nvidia_api_key()
                if not api_key:
                    raise SystemExit("NVIDIA_API_KEY not found in environment or .env file.")
                text = _nvidia_generate(
                    api_key=api_key,
                    invoke_url=str(getattr(args, "nvidia_url", "")),
                    model=str(getattr(args, "nvidia_model", "")),
                    system_instruction=sys_prompt,
                    user_text=user_msg,
                    temperature=float(getattr(args, "temperature", 0.0)),
                    top_p=float(getattr(args, "top_p", 1.0)),
                    top_k=(int(getattr(args, "nvidia_top_k", 0)) if int(getattr(args, "nvidia_top_k", 0)) > 0 else None),
                    presence_penalty=(
                        float(getattr(args, "nvidia_presence_penalty", 0.0))
                        if float(getattr(args, "nvidia_presence_penalty", 0.0)) != 0.0
                        else None
                    ),
                    repetition_penalty=(
                        float(getattr(args, "nvidia_repetition_penalty", 0.0))
                        if float(getattr(args, "nvidia_repetition_penalty", 0.0)) != 0.0
                        else None
                    ),
                    enable_thinking=bool(getattr(args, "nvidia_enable_thinking", False)),
                    max_tokens=max_tokens,
                    timeout_s=float(getattr(args, "timeout_s", 600)),
                    api_retries=int(getattr(args, "nvidia_api_retries", 6)),
                    retry_sleep_s=float(getattr(args, "nvidia_retry_sleep_s", 5.0)),
                )
            elif getattr(args, "engine", "") == "openai":
                api_key = _get_openai_api_key()
                if not api_key:
                    raise SystemExit("GPT_API_KEY_OPEN_AI (or OPENAI_API_KEY) not found in environment or .env file.")
                text = _openai_generate(
                    api_key=api_key,
                    model=str(getattr(args, "openai_model", "gpt-5.2")),
                    system_instruction=sys_prompt,
                    user_text=user_msg,
                    temperature=float(getattr(args, "temperature", 0.0)),
                    top_p=float(getattr(args, "top_p", 1.0)),
                    max_tokens=max_tokens,
                    timeout_s=float(getattr(args, "timeout_s", 600)),
                    json_mode=bool(getattr(args, "json_mode", True)),
                    verbosity=str(getattr(args, "openai_verbosity", "medium")),
                    reasoning_effort=str(getattr(args, "openai_reasoning_effort", "none")),
                    api_retries=int(getattr(args, "openai_api_retries", 6)),
                    retry_sleep_s=float(getattr(args, "openai_retry_sleep_s", 2.0)),
                )
            elif getattr(args, "engine", "") == "anthropic":
                api_key = _get_anthropic_api_key()
                if not api_key:
                    raise SystemExit("ANTHROPIC_API_KEY not found in environment or .env file.")
                text = _anthropic_generate(
                    api_key=api_key,
                    model=str(getattr(args, "anthropic_model", "claude-sonnet-4-6")),
                    system_instruction=sys_prompt,
                    user_text=user_msg,
                    temperature=float(getattr(args, "temperature", 0.0)),
                    max_tokens=max_tokens,
                    timeout_s=float(getattr(args, "timeout_s", 600)),
                    effort=str(getattr(args, "anthropic_effort", "high")),
                    api_retries=int(getattr(args, "anthropic_api_retries", 6)),
                    retry_sleep_s=float(getattr(args, "anthropic_retry_sleep_s", 2.0)),
                )
            else:
                text = _chat(
                    base_url=str(getattr(args, "stage1_url", "http://127.0.0.1:1245")),
                    model=str(getattr(args, "stage1_model", "medgemma-base")),
                    messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_msg}],
                    max_tokens=max_tokens,
                    temperature=float(getattr(args, "temperature", 0.0)),
                    top_p=float(getattr(args, "top_p", 1.0)),
                    timeout_s=int(getattr(args, "timeout_s", 600)),
                    json_mode=bool(getattr(args, "json_mode", True)),
                    repeat_penalty=float(getattr(args, "repeat_penalty", 1.30)),
                )

            leak = has_thinking_leak(text)
            any_thinking_leak = any_thinking_leak or leak
            if leak and not bool(getattr(args, "allow_thinking_leak", False)):
                last_err = "thinking_leak"
                continue
            try:
                obj0 = extract_json_object(text)
                coerced0 = coerce_stage1_summary(obj0, stage1_keys)
                sanitized0 = _sanitize_stage1_summary(coerced0, stage1_keys)
                sanitized0 = _inject_appendix_lines(sanitized0, appendix_lines, stage1_keys)
                if not _is_stage1_output_acceptable(sanitized0, stage1_keys):
                    doc_stats["reject_unacceptable"] += 1
                    raise ValueError("Stage1 JSON failed acceptability checks (repetition/length).")
                last_err = None
                break
            except Exception as e:
                last_err = str(e)
                continue

        doc_stats["retries_used_total"] += int(retries_used)
        if last_err is not None:
            doc_stats["retries_exhausted"] += 1
        if last_err == "thinking_leak":
            doc_stats["thinking_leak"] += 1

        raw_path = run_dir / f"{doc_id}.{strat}.stage1_raw.txt"
        raw_path.write_text(text, encoding="utf-8")

        parse_err: str | None = None
        obj_for_keys: dict[str, Any] | None = None
        try:
            obj_for_keys = extract_json_object(text)
            doc_stats["json_parse_ok"] += 1
        except Exception as e:
            doc_stats["json_parse_fail"] += 1
            parse_err = str(e)
            if sanitized0 is None:
                sanitized0 = _sanitize_stage1_summary(coerce_stage1_summary({}, stage1_keys), stage1_keys)
                sanitized0 = _inject_appendix_lines(sanitized0, appendix_lines, stage1_keys)

        if parse_err and not parse_err_any:
            parse_err_any = parse_err

        missing_keys = [
            k for k in stage1_keys if k not in (obj_for_keys.keys() if isinstance(obj_for_keys, dict) else {})
        ]
        extra_keys = []
        if isinstance(obj_for_keys, dict):
            extra_keys = [k for k in obj_for_keys.keys() if k not in stage1_keys]

        if missing_keys and not missing_keys_any:
            missing_keys_any = missing_keys
        if extra_keys and not extra_keys_any:
            extra_keys_any = extra_keys

        doc_stats["missing_keys_total"] += len(missing_keys)
        doc_stats["extra_keys_total"] += len(extra_keys)

        if sanitized0 is not None:
            per_slice_summaries.append(sanitized0)

        per_slice_meta.append(
            {
                "trim_strategy": strat,
                "parse_error": parse_err,
                "missing_keys": missing_keys,
                "extra_keys": extra_keys,
                "raw_path": raw_path.name,
                "retries_used": retries_used,
            }
        )

    merged = (
        _merge_stage1_summaries(per_slice_summaries, stage1_keys)
        if per_slice_summaries
        else _sanitize_stage1_summary(coerce_stage1_summary({}, stage1_keys), stage1_keys)
    )
    merged = _inject_appendix_lines(merged, appendix_lines, stage1_keys)

    empties = sum(1 for k in stage1_keys if not str(merged.get(k, "")).strip())
    doc_stats["empty_values"] += int(empties)

    out_json = {
        "document_id": doc_id,
        "thinking_leak": bool(any_thinking_leak),
        "parse_error": parse_err_any,
        "missing_keys": missing_keys_any,
        "extra_keys": extra_keys_any,
        "summary": merged,
        "trim_strategies": trim_strategies,
        "slices": per_slice_meta,
    }
    (run_dir / f"{doc_id}.stage1_summary.json").write_text(
        json.dumps(out_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Count thinking leaks per-document (not only hard rejects).
    doc_stats["thinking_leak"] = int(bool(any_thinking_leak))
    return doc_stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--language", default="en", choices=["en", "it"])
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--dataset-id", default="NLP-FBK/dyspnea-crf-train")
    p.add_argument(
        "--engine",
        default="llama",
        choices=["llama", "gemini", "gemini_vertex", "nvidia", "openai", "anthropic"],
    )
    p.add_argument("--stage1-url", default="http://127.0.0.1:1245")
    p.add_argument("--stage1-model", default="medgemma-base")
    p.add_argument("--nvidia-url", default="https://integrate.api.nvidia.com/v1/chat/completions")
    p.add_argument("--nvidia-model", default="meta/llama-3.1-405b-instruct")
    p.add_argument("--nvidia-api-retries", type=int, default=6)
    p.add_argument("--nvidia-retry-sleep-s", type=float, default=5.0)
    p.add_argument("--nvidia-delay-s", type=float, default=0.0)
    p.add_argument("--nvidia-top-k", type=int, default=0, help="Optional NVIDIA top_k (0 disables).")
    p.add_argument("--nvidia-presence-penalty", type=float, default=0.0, help="Optional NVIDIA presence_penalty.")
    p.add_argument(
        "--nvidia-repetition-penalty",
        type=float,
        default=0.0,
        help="Optional NVIDIA repetition_penalty (0 disables, provider default otherwise).",
    )
    p.add_argument(
        "--nvidia-enable-thinking",
        action="store_true",
        default=False,
        help="Enable NVIDIA chat_template_kwargs.enable_thinking (may increase latency and risk of drift).",
    )
    p.add_argument(
        "--allow-thinking-leak",
        action="store_true",
        default=False,
        help=(
            "Accept outputs that contain 'thinking' text before the JSON object (still parses JSON). "
            "Default is strict rejection for Stage1 stability."
        ),
    )
    p.add_argument("--gemini-model", default="gemini-2.0-flash")
    p.add_argument("--gemini-vertex-location", default="us-central1")
    p.add_argument("--gemini-thinking-budget", type=int, default=0)
    p.add_argument("--gemini-api-retries", type=int, default=6)
    p.add_argument("--gemini-retry-sleep-s", type=float, default=5.0)
    p.add_argument("--gemini-delay-s", type=float, default=0.0)
    p.add_argument("--openai-model", default="gpt-5.2")
    p.add_argument("--openai-verbosity", default="medium", choices=["low", "medium", "high"])
    p.add_argument(
        "--openai-reasoning-effort",
        default="none",
        choices=["none", "low", "medium", "high"],
        help="Set to 'none' to omit reasoning_effort parameter.",
    )
    p.add_argument("--openai-api-retries", type=int, default=6)
    p.add_argument("--openai-retry-sleep-s", type=float, default=2.0)
    p.add_argument("--anthropic-model", default="claude-sonnet-4-6")
    p.add_argument("--anthropic-effort", default="high", choices=["low", "medium", "high", "none"])
    p.add_argument("--anthropic-api-retries", type=int, default=6)
    p.add_argument("--anthropic-retry-sleep-s", type=float, default=2.0)
    p.add_argument("--profile", default="9", choices=["9", "7"])
    p.add_argument(
        "--min-system-prompt",
        action="store_true",
        default=False,
        help="Use the minimal Stage1 system prompt even on the first attempt (reduces verbose reasoning in some teacher models).",
    )
    p.add_argument("--max-new-tokens", type=int, default=650)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--repeat-penalty", type=float, default=1.30)
    p.add_argument("--timeout-s", type=int, default=600)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Parallel workers across documents (teacher engines only recommended). Each doc still runs slices sequentially.",
    )
    p.add_argument(
        "--max-note-chars",
        type=int,
        default=0,
        help="Trim the clinical note before Stage1 (0 disables). Use this for dev80 speed/stability.",
    )
    p.add_argument("--trim-strategy", default="middle", choices=["middle", "head", "tail", "head_tail"])
    p.add_argument(
        "--multi-trim-strategies",
        default="",
        help="Optional extra trim strategies (comma-separated) to run Stage1 on and merge per-doc (e.g., 'head_tail,head').",
    )
    p.add_argument("--json-mode", action="store_true", default=True)
    p.add_argument("--no-json-mode", dest="json_mode", action="store_false")
    p.add_argument("--out-dir", default="submissions/stage1_train10")
    p.add_argument(
        "--run-dir",
        default="",
        help="Optional explicit run directory (overrides timestamp-based run_{ts}_{lang}_p{profile}).",
    )
    p.add_argument("--resume", action="store_true", help="Skip docs that already have *.stage1_summary.json in --run-dir.")
    args = p.parse_args()

    stage1_keys = STAGE1_KEYS_9 if args.profile == "9" else STAGE1_KEYS_7
    stage1_sys = build_stage1_system_prompt(stage1_keys)
    keys_str = ", ".join([f'"{k}"' for k in stage1_keys])
    stage1_sys_min = f"""You are an expert clinical summarization engine.

Return ONE valid JSON object with EXACTLY these keys:
{keys_str}

STRICT RULES:
- JSON only, no markdown.
- Each value MUST be a string (NOT a list/dict).
- If a cluster has no supported facts: value must be exactly "not stated".
- Otherwise, put 1-8 lines in the string, each line in the form "Key: Value" using \\n inside the JSON string.
- NEVER output per-item "not stated" lines. Omit missing items.
- NEVER repeat lines.
"""

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    if str(args.run_dir).strip():
        run_dir = Path(str(args.run_dir))
    else:
        run_dir = out_dir / f"run_{run_id}_{args.language}_p{args.profile}"
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = _hf_rows(args.dataset_id, split=args.language, offset=0, length=int(args.limit))
    hf_rows = rows.get("rows") or []
    if not hf_rows:
        raise SystemExit("No rows returned from HF datasets-server.")

    stats = {
        "n": 0,
        "thinking_leak": 0,
        "json_parse_ok": 0,
        "json_parse_fail": 0,
        "missing_keys_total": 0,
        "extra_keys_total": 0,
        "empty_values": 0,
        "retries_used_total": 0,
        "retries_exhausted": 0,
        "reject_nonstring": 0,
        "reject_missing_keys": 0,
        "reject_unacceptable": 0,
    }

    def _accum(dst: dict[str, int], src: dict[str, int]) -> None:
        for k, v in src.items():
            if k in dst:
                dst[k] += int(v)

    def _write_empty(doc_id: str) -> None:
        empty = _sanitize_stage1_summary(coerce_stage1_summary({}, stage1_keys), stage1_keys)
        out_json = {
            "document_id": doc_id,
            "thinking_leak": False,
            "parse_error": "exception",
            "missing_keys": [],
            "extra_keys": [],
            "summary": empty,
            "trim_strategies": [str(args.trim_strategy)],
            "slices": [],
        }
        (run_dir / f"{doc_id}.stage1_summary.json").write_text(
            json.dumps(out_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    todo: list[tuple[int, dict[str, Any], str]] = []
    for row_i, row in enumerate(hf_rows):
        rec = row["row"]
        doc_id = str(rec.get("document_id") or row.get("row_idx"))
        if bool(args.resume) and (run_dir / f"{doc_id}.stage1_summary.json").exists():
            continue
        todo.append((row_i, rec, doc_id))

    max_workers = max(1, int(getattr(args, "max_workers", 1)))
    if max_workers == 1:
        for row_i, rec, doc_id in todo:
            try:
                ds = _process_one_doc(
                    rec=rec,
                    doc_id=doc_id,
                    row_i=row_i,
                    args=args,
                    stage1_keys=stage1_keys,
                    stage1_sys=stage1_sys,
                    stage1_sys_min=stage1_sys_min,
                    run_dir=run_dir,
                )
                _accum(stats, ds)
            except Exception:
                _write_empty(doc_id)
                _accum(stats, {"n": 1, "retries_exhausted": 1})
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {
                ex.submit(
                    _process_one_doc,
                    rec=rec,
                    doc_id=doc_id,
                    row_i=row_i,
                    args=args,
                    stage1_keys=stage1_keys,
                    stage1_sys=stage1_sys,
                    stage1_sys_min=stage1_sys_min,
                    run_dir=run_dir,
                ): doc_id
                for (row_i, rec, doc_id) in todo
            }
            for fut in as_completed(futs):
                doc_id = futs[fut]
                try:
                    ds = fut.result()
                    _accum(stats, ds)
                except Exception:
                    _write_empty(doc_id)
                    _accum(stats, {"n": 1, "retries_exhausted": 1})

    (run_dir / "stage1_audit_summary.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "dataset_id": args.dataset_id,
                "split": args.language,
                "profile": args.profile,
                "engine": str(args.engine),
                "stage1_url": args.stage1_url,
                "stage1_model": args.stage1_model,
                "gemini_model": str(args.gemini_model),
                "gemini_thinking_budget": int(args.gemini_thinking_budget),
                "gemini_vertex_location": str(args.gemini_vertex_location),
                "json_mode": bool(args.json_mode),
                "stats": stats,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote: {run_dir}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
