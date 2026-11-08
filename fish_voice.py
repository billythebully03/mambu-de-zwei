"""Голос Мембу через Fish Audio. Ключ и голос берутся из переменных окружения:
FISH_API_KEY и FISH_VOICE_ID. Без них голос выключен, и инструмент не объявляется модели."""

import logging
import os

import httpx

FISH_URL = "https://api.fish.audio/v1/tts"
FISH_MODEL = os.getenv("FISH_MODEL", "s2.1-pro-free")
MAX_VOICE_CHARS = 240


def voice_enabled() -> bool:
    return bool(
        os.getenv("FISH_API_KEY", "").strip()
        and os.getenv("FISH_VOICE_ID", "").strip()
    )


async def synthesize(text: str) -> bytes | None:
    """Возвращает mp3 или None, если синтез не удался."""
    key = os.getenv("FISH_API_KEY", "").strip()
    voice = os.getenv("FISH_VOICE_ID", "").strip()

    if not key or not voice:
        return None

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "model": FISH_MODEL,
    }
    payload = {
        "text": text[:MAX_VOICE_CHARS],
        "reference_id": voice,
        "format": "mp3",
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(FISH_URL, headers=headers, json=payload)
    except httpx.HTTPError as exc:
        logging.warning("FISH | сеть: %s", exc)
        return None

    if resp.status_code != 200 or not resp.content:
        logging.warning("FISH | статус %s: %s", resp.status_code, resp.text[:200])
        return None

    return resp.content
