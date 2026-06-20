"""Translate text through LM Studio's local OpenAI-compatible API."""

from __future__ import annotations

import requests

from config import LM_STUDIO_MODEL, LM_STUDIO_URL


class TranslationError(RuntimeError):
    pass


def translate(text: str, source_language: str, target_language: str) -> str:
    text = text.strip()
    if not text:
        return ""

    system_prompt = (
        "You are a professional translator. Translate accurately and naturally. "
        "Return only the translated text with no quotes, labels, or commentary."
    )
    user_prompt = (
        f"Translate the following from {source_language} to {target_language}:\n\n{text}"
    )

    payload = {
        "model": LM_STUDIO_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
        "stream": False,
    }

    try:
        response = requests.post(
            f"{LM_STUDIO_URL.rstrip('/')}/v1/chat/completions",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer lm-studio",
            },
            timeout=120,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise TranslationError(
            f"Could not reach LM Studio at {LM_STUDIO_URL}. "
            "Start LM Studio's local server and load a model."
        ) from exc

    body = response.json()
    try:
        return body["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise TranslationError(f"Unexpected LM Studio response: {body}") from exc
