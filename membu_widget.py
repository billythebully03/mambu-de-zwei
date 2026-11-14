"""Картинки Мембу: виджет-заглушка и генерация через Cloudflare Workers AI (FLUX).

Нужные переменные окружения: TOKEN, CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN.
Необязательные: CF_IMAGE_MODEL (по умолчанию FLUX.1 Schnell).
Без CLOUDFLARE_ACCOUNT_ID и CLOUDFLARE_API_TOKEN инструмент не объявляется модели.
"""

import base64
import logging
import os
import zlib
from pathlib import Path

import httpx

BASE_DIR = Path(__file__).parent
TEMPLATE_PATH = BASE_DIR / "widget_template.html"
THUMB_PATH = BASE_DIR / "public" / "membu_thumb.webp"

CF_MODEL = os.getenv("CF_IMAGE_MODEL", "@cf/black-forest-labs/flux-1-schnell").strip()
CF_URL = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{model}"
TG_URL = "https://api.telegram.org/bot{token}/{method}"
WIDGET_SIZE = 240
WIDGET_HTML_ZLIB_LIMIT = 4096


def widget_enabled() -> bool:
    return bool(
        os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
        and os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
        and THUMB_PATH.exists()
        and TEMPLATE_PATH.exists()
    )


def build_widget_html() -> str:
    """Подставляет миниатюру в шаблон. Telegram принимает HTML не больше 4 КБ после zlib."""
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    uri = "data:image/webp;base64," + base64.b64encode(THUMB_PATH.read_bytes()).decode()
    html = template.replace("__IMG__", uri)
    packed = len(zlib.compress(html.encode("utf-8"), 9))
    if packed > WIDGET_HTML_ZLIB_LIMIT:
        logging.warning("WIDGET | HTML %s байт после zlib, лимит %s", packed, WIDGET_HTML_ZLIB_LIMIT)
    return html


_WIDGET_HTML: str | None = None


def _html() -> str:
    global _WIDGET_HTML
    if _WIDGET_HTML is None:
        _WIDGET_HTML = build_widget_html()
    return _WIDGET_HTML


async def _tg(method: str, payload: dict) -> dict | None:
    token = os.getenv("TOKEN", "").strip()
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(TG_URL.format(token=token, method=method), json=payload)
        data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        logging.warning("WIDGET | %s сеть: %s", method, exc)
        return None
    if not data.get("ok"):
        desc = str(data.get("description", ""))
        if "not modified" in desc:
            return {"ok": True}
        logging.warning("WIDGET | %s: %s", method, desc[:300])
        return None
    return data


async def send_widget(chat_id: int, caption: str) -> int | None:
    """Отправляет виджет-заглушку. Возвращает message_id или None."""
    data = await _tg(
        "sendWidget",
        {
            "chat_id": chat_id,
            "html": _html(),
            "caption": caption[:200],
            "width": WIDGET_SIZE,
            "height": WIDGET_SIZE,
        },
    )
    if not data:
        return None
    return data["result"]["message_id"]


async def finish_widget(chat_id: int, message_id: int) -> None:
    """Переводит виджет в финальное состояние: он размывается и уходит."""
    await _tg(
        "editMessageWidgetData",
        {"chat_id": chat_id, "message_id": message_id, "chat_data": '{"s":"done"}'},
    )


async def generate_image(prompt: str) -> bytes | None:
    """Рисует картинку через Cloudflare Workers AI. Возвращает байты изображения или None."""
    account = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
    key = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
    if not account or not key:
        return None
    url = CF_URL.format(account=account, model=CF_MODEL)
    try:
        async with httpx.AsyncClient(timeout=90) as client:
            resp = await client.post(
                url,
                headers={"Authorization": f"Bearer {key}"},
                json={"prompt": prompt[:1500], "steps": 4},
            )
    except httpx.HTTPError as exc:
        logging.warning("CF_IMAGE | сеть: %s", exc)
        return None
    if resp.status_code != 200:
        logging.warning("CF_IMAGE | статус %s: %s", resp.status_code, resp.text[:300])
        return None
    try:
        image_b64 = resp.json()["result"]["image"]
        return base64.b64decode(image_b64)
    except (ValueError, KeyError, TypeError) as exc:
        logging.warning("CF_IMAGE | неожиданный ответ: %s", exc)
        return None
