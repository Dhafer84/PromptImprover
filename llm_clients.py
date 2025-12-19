from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class LLMConfig:
    provider: str  # "openai" or "groq"
    model: str
    temperature: float = 0.2
    max_tokens: int = 700


# Modèles de secours (tu peux ajuster)
GROQ_FALLBACK_MODELS = [
    "llama-3.1-8b-instant",     # très souvent dispo + rapide
    "llama-3.2-70b-versatile",  # si dispo
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]


def _get_env(key: str) -> str:
    return os.getenv(key, "").strip()


def _looks_like_model_decommissioned(err: Exception) -> bool:
    """
    Détection robuste de l'erreur Groq 'model_decommissioned' via message texte.
    """
    msg = str(err).lower()
    return ("model_decommissioned" in msg) or ("decommissioned" in msg) or ("no longer supported" in msg)


def chat_completion(cfg: LLMConfig, system: str, user: str) -> tuple[str, str]:
    """
    Retourne: (texte, modele_utilise)
    - OpenAI: pas de fallback par défaut (facile à ajouter si tu veux)
    - Groq: fallback automatique si modèle décommissionné
    """
    provider = (cfg.provider or "").lower().strip()

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
        content = resp.choices[0].message.content or ""
        return content, cfg.model

    if provider == "groq":
        from groq import Groq

        api_key = _get_env("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to Streamlit secrets or env vars.")

        client = Groq(api_key=api_key)

        # Liste des modèles à tester: modèle choisi + fallbacks
        models_to_try = [cfg.model] + [m for m in GROQ_FALLBACK_MODELS if m != cfg.model]

        last_err: Exception | None = None

        for model_name in models_to_try:
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                content = resp.choices[0].message.content or ""
                return content, model_name

            except Exception as e:
                last_err = e
                # Si modèle décommissionné → on continue vers fallback
                if _looks_like_model_decommissioned(e):
                    continue
                # Sinon c'est une vraie autre erreur → on remonte
                raise

        raise RuntimeError(
            "All Groq models failed. "
            f"First model was '{cfg.model}'. Last error: {last_err}"
        )

    raise ValueError("Unsupported provider. Use 'openai' or 'groq'.")
