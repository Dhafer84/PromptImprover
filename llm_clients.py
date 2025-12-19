from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    provider: str  # "openai" or "groq"
    model: str
    temperature: float = 0.2
    max_tokens: int = 700


def _get_env(key: str) -> str:
    v = os.getenv(key, "").strip()
    return v


def chat_completion(cfg: LLMConfig, system: str, user: str) -> str:
    provider = cfg.provider.lower().strip()

    if provider == "openai":
        from openai import OpenAI
        api_key = _get_env("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to Streamlit secrets or env vars.")
        client = OpenAI(api_key=api_key)

        resp = client.chat.completions.create(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""

    if provider == "groq":
        from groq import Groq
        api_key = _get_env("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to Streamlit secrets or env vars.")
        client = Groq(api_key=api_key)

        resp = client.chat.completions.create(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""

    raise ValueError("Unsupported provider. Use 'openai' or 'groq'.")
