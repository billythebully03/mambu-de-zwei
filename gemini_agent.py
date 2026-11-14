import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

from aiogram.exceptions import TelegramAPIError
from aiogram.types import BufferedInputFile, ChatPermissions, ReactionTypeEmoji
from google import genai
from google.genai import types

import fish_voice
import membu_widget


MEMORY_CATEGORIES = [
    "profile",
    "preference",
    "relationship",
    "ongoing",
    "event",
    "inside_joke",
    "opinion",
]

TELEGRAM_REACTIONS = [
    "❤", "👍", "👎", "🔥", "🥰", "👏", "😁", "🤔", "🤯", "😱",
    "🤬", "😢", "🎉", "🤩", "🤮", "💩", "🙏", "👌", "🕊", "🤡",
    "🥱", "🥴", "😍", "🐳", "❤‍🔥", "🌚", "🌭", "💯", "🤣", "⚡",
    "🍌", "🏆", "💔", "🤨", "😐", "🍓", "🍾", "💋", "🖕", "😈",
    "😴", "😭", "🤓", "👻", "👨‍💻", "👀", "🎃", "🙈", "😇", "😨",
    "🤝", "✍", "🤗", "🫡", "🎅", "🎄", "☃", "💅", "🤪", "🗿",
    "🆒", "💘", "🙉", "🦄", "😘", "💊", "🙊", "😎", "👾", "🤷‍♂",
    "🤷", "🤷‍♀", "😡",
]

MIN_MESSAGES_FOR_STYLE_UPDATE = 60

# Аналитик стиля смотрит на переписку не чаще, чем раз в эти секунды.
STYLE_CHECK_INTERVAL_SECONDS = 60
# Сколько последних сообщений (людей и Мембу) отдаём аналитику стиля.
STYLE_SAMPLE_SIZE = 100
MAX_CHAT_ACTIONS_PER_TURN = 9
WEB_SEARCH_COOLDOWN_SECONDS = 30
DICE_EMOJIS = ["🎲", "🎯", "🏀", "⚽", "🎳", "🎰"]


def dice_meaning(emoji: str, value: int | None) -> str:
    """Что означает значение анимации Telegram, чтобы модель говорила по делу."""
    if value is None:
        return "результат неизвестен"
    if emoji == "🎯":
        return "в яблочко" if value == 6 else "мимо яблочка"
    if emoji == "🏀":
        return "попал в кольцо" if value in (4, 5) else "промах"
    if emoji == "⚽":
        return "гол" if value in (3, 4, 5) else "мимо ворот"
    if emoji == "🎳":
        return "страйк" if value == 6 else "не страйк"
    if emoji == "🎰":
        return "джекпот" if value == 64 else "не джекпот"
    return f"выпало {value}"
PRIMARY_RETRY_SECONDS = 1800


def _should_failover(exc: Exception) -> bool:
    """Переключаемся на резерв при квоте, отказе в доступе или неверном ключе."""
    code = getattr(exc, "code", None)
    text = str(exc).lower()

    if code in (401, 403, 429):
        return True

    return any(
        marker in text
        for marker in ("resource_exhausted", "quota", "api key", "permission denied", "429")
    )
CHAT_ACTION_PAUSE_SECONDS = 1.0
VOICE_MIN_INTERVAL_SECONDS = 60
IMAGE_USER_COOLDOWN_SECONDS = 90
IMAGE_CHAT_GAP_SECONDS = 5

# Кэш памяти из Supabase: как долго держим его без перезагрузки.
MEMORY_CACHE_TTL_SECONDS = 300
MEMORY_LOAD_LIMIT = 60
# Сколько самых важных воспоминаний кладём в каждый контекст.
MEMORY_IN_CONTEXT = 40
STYLE_CACHE_TTL_SECONDS = 300

STYLE_EDIT_INSTRUCTION = (
    "Ты редактор профиля манеры общения чата. Тебе дают текущий профиль и правку владельца. "
    "Примени правку и верни ТОЛЬКО обновлённый профиль в том же формате, без пояснений. "
    "Правка владельца важнее прежних пунктов. Если правка — число, оставь ровно столько "
    "самых важных пунктов профиля. Если правка — текст, внеси её по смыслу. "
    "Ничего не добавляй от себя сверх правки."
)

STYLE_ANALYST_INSTRUCTION = (
    "Ты ведёшь профиль манеры общения Мембу в этом чате. Твоя задача — описать, как "
    "в этом чате принято писать, чтобы Мембу постепенно говорил так же, как участники. "
    "Смотри на манеру, а не на содержание: длина реплик, регистр, сленг, обращения на "
    "«ты» или «вы», эмодзи, скобочки, растяжки слов, как люди смеются, насколько "
    "формально, темп, как реагируют на подколы и грубость. Не переноси в профиль темы, "
    "факты, шутки, имена и личные сведения участников. Характер Мембу не меняется, его "
    "ядро описано в PERSONALITY. Подстройка идёт плавно: если дан текущий профиль, "
    "сохрани большую его часть и измени не больше двух пунктов, а именно те, что явно "
    "отличаются от реальной переписки. Не переноси в профиль слова-паразиты и филлеры "
    "(ну, короче, в общем), готовые обороты и присказки, даже если они частые. Не копируй "
    "агрессию, оскорбления и грубость участников, а также ролевые игры и выдуманных персонажей: "
    "переноси только живость и манеру. "
    "Структура: сначала короткое описание норм чата, "
    "затем раздел «Подстройка Мембу» до 5 пунктов, каждый — умеренный шаг от текущей "
    "манеры Мембу к нормам чата. Опирайся только на переписку, не выдумывай "
    "закономерностей. Простой текст, до 2000 символов, без markdown."
)

# Дайджест: раз в столько новых сообщений модель выделяет из переписки
# долговременные факты и записывает их в память чата.
DIGEST_EVERY_MESSAGES = 150
DIGEST_SAMPLE_SIZE = 120
DIGEST_CHECK_INTERVAL_SECONDS = 60

DIGEST_INSTRUCTION = (
    "Ты помогаешь Telegram-боту Мембу помнить чат. Из переписки выдели 2–5 "
    "долговечных фактов, которые стоит запомнить: о чём договорились, какие темы "
    "обсуждали и чем закончили, предпочтения людей, устойчивые мнения, незавершённые "
    "дела. Шутки и повторяющиеся темы не сохраняй: это не факты, и память заставит "
    "возвращаться к ним. Категорию inside_joke не используй. Важность темы не выше 3. Одно упоминание темы — не факт: не записывай интерес, предпочтение или мнение, если тема прозвучала один раз; записывай только повторявшееся или прямо названное участником. "
    "Не пересказывай каждую реплику и не записывай "
    "случайное. Не сохраняй пароли, документы, здоровье, личные секреты и "
    "чувствительные сведения. Верни ТОЛЬКО JSON-массив без пояснений. Каждый элемент: "
    "{\"category\": одна из profile, preference, relationship, ongoing, event, "
    "inside_joke, opinion; \"content\": краткая формулировка до 300 символов; "
    "\"importance\": число от 1 до 5; \"people\": массив имён}. Если важного нет, "
    "верни []."
)


def background_memories(
    rows: list[dict],
    limit: int = MEMORY_IN_CONTEXT,
    per_category: int = 8,
) -> list[dict]:
    """Фон памяти для контекста: без внутренних шуток и без перекоса в одну категорию."""
    counts: dict[str, int] = {}
    out: list[dict] = []

    for row in rows:
        category = row.get("category") or ""

        if category == "inside_joke":
            continue
        if counts.get(category, 0) >= per_category:
            continue

        counts[category] = counts.get(category, 0) + 1
        out.append(row)

        if len(out) >= limit:
            break

    return out


def _parse_ts(value) -> datetime | None:
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def active_burst(messages: list[dict], now: datetime) -> dict | None:
    """Если человек пишет серией коротких сообщений подряд, возвращает её описание.
    Тогда он, скорее всего, ещё не закончил мысль."""
    rows = []
    for m in messages:
        ts = _parse_ts(m.get("created_at"))
        if ts:
            rows.append((ts, m))

    if len(rows) < 2:
        return None

    rows.sort(key=lambda item: item[0])
    last_ts, last = rows[-1]

    if (now - last_ts).total_seconds() > 30:
        return None

    sender = last.get("sender_id") or last.get("sender_name")
    count = 0
    prev_ts = None

    for ts, m in reversed(rows):
        who = m.get("sender_id") or m.get("sender_name")
        if who != sender:
            break
        if prev_ts and (prev_ts - ts).total_seconds() > 25:
            break
        count += 1
        prev_ts = ts

    if count < 2:
        return None

    return {
        "sender_name": last.get("sender_name") or "участник",
        "messages": count,
        "last_message_seconds_ago": round((now - last_ts).total_seconds(), 1),
    }


def trim_own_messages(messages: list[dict], keep: int = 3) -> list[dict]:
    """Оставляет в контексте только последние реплики самого Мембу, чтобы он не
    продолжал свою прошлую шутку или тему по инерции."""
    out: list[dict] = []
    kept = 0

    for m in reversed(messages):
        if m.get("sender_name") == "Мембу":
            if kept >= keep:
                continue
            kept += 1
        out.append(m)

    return list(reversed(out))


def parse_digest_items(raw: str) -> list[dict]:
    """Разбирает ответ дайджеста. Некорректные элементы пропускает."""
    text = (raw or "").strip()

    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logging.warning("Дайджест памяти: ответ не является JSON")
        return []

    if not isinstance(data, list):
        return []

    items: list[dict] = []

    for raw_item in data[:5]:
        if not isinstance(raw_item, dict):
            continue

        category = raw_item.get("category")
        content = str(raw_item.get("content", "")).strip()

        if category not in MEMORY_CATEGORIES or not content:
            continue

        try:
            importance = int(raw_item.get("importance", 3))
        except (TypeError, ValueError):
            importance = 3

        people = raw_item.get("people") or []
        if not isinstance(people, list):
            people = []

        items.append(
            {
                "category": category,
                "content": content[:1500],
                "importance": max(1, min(importance, 5)),
                "people": [str(p) for p in people][:10],
            }
        )

    return items


# Признаки того, что человек просит вспомнить прошлое. Совпадение запускает
# поднятие истории в коде, а не полагается на то, что модель сама вызовет инструмент.
RECALL_PATTERN = re.compile(
    r"вспом|помн|напомн|раньше|прошл|о чем|о чём|что обсужд|обсуждали|"
    r"говорили|вчера|неделю|на прошлой|когда-то|было ли",
    re.IGNORECASE,
)
RECALL_PAGES = 4
RECALL_PAGE_SIZE = 50
RECALL_STOPWORDS = {
    "помнишь", "помните", "вспомни", "вспомнить", "напомни", "раньше", "прошлый",
    "прошлом", "обсуждали", "говорили", "когда", "тогда", "этом", "чате", "мембу",
    "вообще", "что", "как", "это", "про", "или", "если", "была", "были", "было",
}


# Мнения о людях: раз в полчаса смотрим на самых активных участников.
PEOPLE_CHECK_INTERVAL_SECONDS = 7200
PEOPLE_WINDOW = 150
PEOPLE_MAX_PER_RUN = 4
PEOPLE_MIN_MESSAGES = 5

PEOPLE_OPINION_INSTRUCTION = (
    "Ты пишешь короткое мнение Мембу о конкретном участнике Telegram-чата. Опирайся "
    "только на переписку: как человек общается, что любит, о чём говорит, как "
    "относится к другим. Пиши от первого лица Мембу, 2–4 предложения, спокойно и "
    "по-человечески, без оценок внешности и без выдуманных фактов. Если есть текущее "
    "мнение, обнови его, а не переписывай с нуля. Не записывай здоровье, религию, "
    "сексуальные темы, финансы, документы и другие чувствительные сведения. Простой "
    "текст, до 800 символов, без markdown."
)


def quota_reset_time(text: str, now: datetime) -> datetime:
    """Когда примерно снимется лимит: по подсказке API или в полночь по Pacific."""
    match = re.search(
        r"retry(?:Delay)?\D{0,12}?(\d+(?:\.\d+)?)\s*s",
        text,
        re.IGNORECASE,
    )
    if match:
        return now + timedelta(seconds=float(match.group(1)) + 5)

    # Квоты Gemini обычно обнуляются в полночь по тихоокеанскому времени.
    try:
        pacific = ZoneInfo("America/Los_Angeles")
    except Exception:
        pacific = timezone(timedelta(hours=-8))

    local_now = now.astimezone(pacific)
    midnight = (local_now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return midnight.astimezone(timezone.utc)


class GeminiAgent:
    def __init__(
        self,
        api_key: str,
        storage,
        bot,
        bot_user,
        model: str = "gemini-3.5-flash-lite",
        activity_callback: Callable[[int], None] | None = None,
    ):
        # Основной ключ и резервный (GEMINI_BACKUP_API_KEY). Переключаемся на резерв,
        # когда основной отваливается по квоте или ключу; основной пробуем снова позже.
        keys = [("основной", api_key or "")]
        for label, env_name in (
            ("резервный", "GEMINI_BACKUP_API_KEY"),
            ("резервный 2", "GEMINI_BACKUP2_API_KEY"),
            ("резервный 3", "GEMINI_BACKUP3_API_KEY"),
        ):
            value = os.getenv(env_name, "").strip()
            if value and value not in {k for _, k in keys}:
                keys.append((label, value))
        self._labels = [label for label, _ in keys]
        self._clients = [genai.Client(api_key=value) for _, value in keys]
        self._active = 0
        self._manual = False
        # Кэш системного промпта по ключам: (имя кэша, время создания). Если Gemini
        # не принимает кэш, на этом ключе он отключается ненадолго.
        self._caches: dict[int, tuple[str, float]] = {}
        self._cache_off_until: dict[int, float] = {}
        self._voice_last_at: dict[int, float] = {}
        self._image_day: str = ""
        self._image_count: int = 0
        self._image_last_user: dict[tuple[int, int], float] = {}
        self._image_last_chat: dict[int, float] = {}
        self.switched_at: datetime | None = None
        self.last_failover_reason: str = ""
        self._seen_switched_at: datetime | None = None
        self.storage = storage
        self.bot = bot
        self.bot_user = bot_user
        self.model = model
        self.activity_callback = activity_callback

        # Кэш долговременной памяти и профиля стиля чата, загруженных из Supabase.
        # Значение: (monotonic-время загрузки, данные).
        self._memory_cache: dict[int, tuple[float, list[dict]]] = {}
        self._style_cache: dict[int, tuple[float, dict]] = {}

        # Аналитик стиля: когда последний раз проверяли и что уже идёт.
        self._style_checked_at: dict[int, float] = {}
        self._style_running: set[int] = set()

        # Удерживаем ссылки на фоновые задачи, чтобы их не собрал GC.
        self._background: set[asyncio.Task] = set()

        # Состояние для /status: успешные ответы и последняя ошибка API.
        self.last_ok_at: datetime | None = None
        self.last_error: dict | None = None
        self.quota_reset_at: datetime | None = None

        # Мнения о людях: кэш и защита от параллельного запуска.
        self._people_cache: dict[int, tuple[float, dict]] = {}
        self._people_checked_at: dict[int, float] = {}
        self._people_running: set[int] = set()

        # Дайджест памяти: аналогичные проверка и защита от параллельного запуска.
        self._digest_checked_at: dict[int, float] = {}
        self._digest_running: set[int] = set()
        self._search_at: dict[int, float] = {}
        self._last_sent: dict[int, tuple[str, float]] = {}
        self._photos: dict[tuple[int, int], str] = {}
        self._polls: dict[str, dict] = {}

        base_dir = Path(__file__).resolve().parent

        personality_path = base_dir / "PERSONALITY.md"
        self.personality = personality_path.read_text(encoding="utf-8")

        self.stickers = self._load_stickers(
            base_dir / "stickers.json"
        )

        self.tools = self._build_tools()

        self.config = types.GenerateContentConfig(
            system_instruction=self.personality,
            tools=[self.tools],
            temperature=0.9,
            max_output_tokens=900,
        )

    @staticmethod
    def _load_stickers(path: Path) -> dict[str, dict]:
        """Загружает редактируемый каталог стикеров."""

        try:
            raw = json.loads(
                path.read_text(encoding="utf-8")
            )
        except FileNotFoundError:
            logging.warning(
                "Файл stickers.json не найден; "
                "отправка стикеров отключена"
            )
            return {}
        except (OSError, json.JSONDecodeError) as exc:
            logging.error(
                "Не удалось прочитать stickers.json: %s",
                exc,
            )
            return {}

        entries = (
            raw.get("stickers", [])
            if isinstance(raw, dict)
            else []
        )

        if not isinstance(entries, list):
            logging.error(
                "В stickers.json поле 'stickers' должно быть массивом"
            )
            return {}

        catalog = {}

        for item in entries:
            if not isinstance(item, dict):
                continue

            if item.get("enabled", True) is False:
                continue

            key = str(item.get("key", "")).strip()
            file_id = str(item.get("file_id", "")).strip()

            if not key or not file_id or key in catalog:
                continue

            catalog[key] = {
                "key": key,
                "file_id": file_id,
                "description": str(
                    item.get("description", "")
                ).strip(),
                "meaning": str(
                    item.get("meaning", "")
                ).strip(),
                "usage": str(
                    item.get("usage", "")
                ).strip(),
                "format": str(
                    item.get("format", "static")
                ).strip(),
                "emoji": str(
                    item.get("emoji", "")
                ).strip(),
                "set_name": str(
                    item.get("set_name", "")
                ).strip(),
                "file_unique_id": str(
                    item.get("file_unique_id", "")
                ).strip(),
            }

        logging.info(
            "Загружено стикеров из stickers.json: %s",
            len(catalog),
        )

        return catalog

    def _sticker_catalog_for_model(self) -> str:
        if not self.stickers:
            return "Сейчас доступных стикеров нет."

        lines = []

        for key, sticker in self.stickers.items():
            lines.append(
                f"{key}: {sticker['description']} "
                f"Значение: {sticker['meaning']} "
                f"Когда уместен: {sticker['usage']}"
            )

        return "\n".join(lines)

    def _build_tools(self):
        declarations = [
            types.FunctionDeclaration(
                name="search_memory",
                description=(
                    "Ищет воспоминания Мембу о людях, событиях, отношениях, "
                    "предпочтениях и прошлых разговорах. Используй только если "
                    "старый контекст действительно нужен. Не вызывай автоматически "
                    "перед каждым ответом."
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Что именно нужно вспомнить.",
                        },
                        "category": {
                            "type": "string",
                            "enum": MEMORY_CATEGORIES,
                            "description": "Категория воспоминания. Можно не указывать.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Количество результатов от 1 до 8.",
                        },
                    },
                    "required": ["query"],
                },
            ),

            types.FunctionDeclaration(
                name="save_memory",
                description=(
                    "Сохраняет долговременное воспоминание. Используй только для "
                    "действительно важных или полезных в будущем сведений. "
                    "Не сохраняй каждую реплику, пароли, секреты или чувствительные данные."
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": MEMORY_CATEGORIES,
                        },
                        "content": {
                            "type": "string",
                            "description": "Краткое и понятное воспоминание.",
                        },
                        "importance": {
                            "type": "integer",
                            "description": (
                                "Важность от 1 до 5. Шутки и разовые темы — не выше 2. "
                                "Важность 4–5 только для устойчивых фактов и предпочтений."
                            ),
                        },
                        "people": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "category",
                        "content",
                        "importance",
                    ],
                },
            ),

            types.FunctionDeclaration(
                name="send_message",
                description=(
                    "Отправляет сообщение от имени Мембу в Telegram. Вызывай, "
                    "только если сам решил написать. За один ответ можно отправить до 15 "
                    "сообщений, но только если это реально важно. Если лучше промолчать, "
                    "не вызывай этот инструмент. Можно указать сообщение, на которое "
                    "нужно ответить. Эмодзи можно естественно включить прямо в текст, "
                    "но это необязательно и не должно становиться привычкой в каждом сообщении."
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Текст сообщения Мембу; эмодзи необязательны.",
                        },
                        "reply_to_message_id": {
                            "type": "integer",
                            "description": "Необязательный ID сообщения, на которое нужно ответить.",
                        },
                    },
                    "required": ["text"],
                },
            ),

            types.FunctionDeclaration(
                name="send_poll",
                description=(
                    "Создаёт опрос в чате с вариантами ответа. Используй редко: по случаю или когда "
                    "попросили. Вопрос до 300 символов, от 2 до 10 вариантов, каждый до 100 символов. "
                    "Опрос не анонимный, чтобы видеть итоги."
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "options": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["question", "options"],
                },
            ),

            types.FunctionDeclaration(
                name="close_poll",
                description=(
                    "Закрывает опрос, созданный тобой, и возвращает итоги по вариантам. "
                    "message_id опроса возьми из ответа send_poll."
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {"message_id": {"type": "integer"}},
                    "required": ["message_id"],
                },
            ),

            types.FunctionDeclaration(
                name="send_dice",
                description=(
                    "Отправляет игровую анимацию Telegram: 🎲 кубик, 🎯 дартс, 🏀 баскетбол, "
                    "⚽ футбол, 🎳 боулинг, 🎰 слоты. Результат придёт сразу в ответе инструмента: "
                    "value. Используй, когда уместно или когда попросили. Прокомментируй результат."
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "emoji": {"type": "string", "enum": DICE_EMOJIS},
                    },
                    "required": ["emoji"],
                },
            ),

            types.FunctionDeclaration(
                name="web_search",
                description=(
                    "Ищет актуальные факты в интернете и возвращает краткий ответ. Используй, когда "
                    "нужен свежий факт или ты точно не знаешь ответа. Не для болтовни и не для "
                    "того, что и так знаешь. Не чаще одного раза за пару минут в чате."
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            ),

            types.FunctionDeclaration(
                name="leave_chat",
                description=(
                    "Выходишь из чата. Только если это критично по твоему решению, например чат "
                    "превращается в травлю или тебя явно используют против людей. Сначала скажи "
                    "об этом в чате через send_message, потом вызывай этот инструмент. "
                    "Обязательно укажи причину."
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {"reason": {"type": "string"}},
                    "required": ["reason"],
                },
            ),

            types.FunctionDeclaration(
                name="edit_permissions",
                description=(
                    "Меняет права участника. Для обычного участника — что ему можно писать и отправлять "
                    "(can_send_messages, can_send_photos, can_send_videos, can_send_polls, "
                    "can_add_web_page_previews, can_send_other_messages). Для админа — админские права "
                    "(can_manage_chat, can_delete_messages, can_restrict_members, can_pin_messages, "
                    "can_invite_users, can_change_info, can_promote_members, can_manage_video_chats, "
                    "can_manage_topics). Передавай только то, что меняешь. Создателя, владельцев и себя не трогай."
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "integer"},
                        "can_send_messages": {"type": "boolean"},
                        "can_send_photos": {"type": "boolean"},
                        "can_send_videos": {"type": "boolean"},
                        "can_send_polls": {"type": "boolean"},
                        "can_add_web_page_previews": {"type": "boolean"},
                        "can_send_other_messages": {"type": "boolean"},
                        "can_manage_chat": {"type": "boolean"},
                        "can_delete_messages": {"type": "boolean"},
                        "can_restrict_members": {"type": "boolean"},
                        "can_pin_messages": {"type": "boolean"},
                        "can_invite_users": {"type": "boolean"},
                        "can_change_info": {"type": "boolean"},
                        "can_promote_members": {"type": "boolean"},
                        "can_manage_video_chats": {"type": "boolean"},
                        "can_manage_topics": {"type": "boolean"},
                    },
                    "required": ["user_id"],
                },
            ),

            types.FunctionDeclaration(
                name="look_at_image",
                description=(
                    "Смотрит фото из чата по message_id (он есть в истории как [фото]). Используй только "
                    "когда картинка нужна для ответа, не для всех подряд. question — что именно разглядеть."
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "message_id": {"type": "integer"},
                        "question": {"type": "string"},
                    },
                    "required": ["message_id"],
                },
            ),

            types.FunctionDeclaration(
                name="restrict_member",
                description=(
                    "Временно ограничивает участника: он не сможет писать в чат, пока срок не выйдет. "
                    "Можно для шутки или всерьёз, но только когда участник сам напрашивается "
                    "(например, флудит или хамит). Срок от 1 до 1440 минут, обычно 5–30. "
                    "Не применяй к админам. Обязательно укажи причину."
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "integer", "description": "ID участника из people_notes или контекста."},
                        "minutes": {"type": "integer", "description": "Срок ограничения, 1–1440."},
                        "reason": {"type": "string", "description": "Короткая причина."},
                    },
                    "required": ["user_id", "minutes", "reason"],
                },
            ),

            types.FunctionDeclaration(
                name="unrestrict_member",
                description=(
                    "Снимает ограничение с участника раньше срока, например если шутка удалась "
                    "и хватит."
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "integer", "description": "ID участника."},
                    },
                    "required": ["user_id"],
                },
            ),

            types.FunctionDeclaration(
                name="remove_member",
                description=(
                    "Удаляет участника из чата. Крайне редко, только за серьёзный повод. "
                    "Не применяй к админам. "
                    "Обязательно укажи причину."
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "integer", "description": "ID участника."},
                        "reason": {"type": "string", "description": "Серьёзная причина."},
                    },
                    "required": ["user_id", "reason"],
                },
            ),

            types.FunctionDeclaration(
                name="send_reaction",
                description=(
                    "Ставит одну стандартную emoji-реакцию на существующее сообщение. "
                    "Это полноценный способ ответить вместо текста. Используй, когда "
                    "реакции достаточно, когда участник прямо просит её поставить или "
                    "когда реакция естественно дополняет разговор. Не реагируй на всё "
                    "подряд. message_id выбирай только из переданного контекста; "
                    "не выдумывай идентификаторы. Предпочитай сообщения участников, "
                    "а не собственные сообщения или сообщения других ботов, если "
                    "тебя специально не попросили об обратном. Не пытайся реагировать "
                    "на сервисные сообщения о входе/выходе участников. Допустимые значения: "
                    + ", ".join(TELEGRAM_REACTIONS)
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "message_id": {
                            "type": "integer",
                            "description": "Точный ID сообщения из контекста чата.",
                        },
                        "emoji": {
                            "type": "string",
                            "enum": TELEGRAM_REACTIONS,
                            "description": "Одна стандартная Telegram-реакция.",
                        },
                    },
                    "required": ["message_id", "emoji"],
                },
            ),

            types.FunctionDeclaration(
                name="search_own_messages",
                description=(
                    "Ищет старые текстовые сообщения, которые сам Мембу отправлял "
                    "в текущем Telegram-чате. Используй, когда нужно проверить, "
                    "что именно ты раньше говорил, найти собственное обещание, "
                    "вернуться к старой теме или не противоречить себе. "
                    "Поиск идёт по словам или точной фразе, а не по смысловой близости. "
                    "Не вызывай перед каждым ответом. В результатах будут message_id "
                    "и дата каждого сообщения."
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Слова или фраза, которые должны встречаться "
                                "в собственном старом сообщении Мембу."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Количество результатов от 1 до 8.",
                        },
                    },
                    "required": ["query"],
                },
            ),

            types.FunctionDeclaration(
                name="search_chat_history",
                description=(
                    "Ищет по всей истории этого чата: и сообщения людей, и твои. "
                    "Используй, когда нужно вспомнить, что именно обсуждали, кто что "
                    "говорил или когда. Слова ищутся по основе, с учётом окончаний, "
                    "и все слова должны встречаться в сообщении. Для более старых "
                    "сообщений передай before. Не вызывай без конкретной причины "
                    "и не перед каждым ответом."
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Слова или короткая фраза для поиска.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Количество результатов от 1 до 20.",
                        },
                        "before": {
                            "type": "string",
                            "description": (
                                "Необязательно: ISO-дата, например 2026-10-01T00:00:00Z. "
                                "Показать только сообщения раньше этого момента."
                            ),
                        },
                    },
                    "required": ["query"],
                },
            ),
            types.FunctionDeclaration(
                name="browse_chat_history",
                description=(
                    "Листает историю этого чата по времени: от самых свежих сообщений "
                    "к более старым. Без before возвращает последние сообщения. "
                    "Чтобы уйти дальше назад, передай в before значение next_before "
                    "из прошлого ответа. Сначала смотри свежее, потом глубже. "
                    "Используй, когда нужно понять, о чём говорили последнее время, "
                    "и поднять прошлые разговоры по порядку."
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "before": {
                            "type": "string",
                            "description": (
                                "Необязательно: next_before из прошлого ответа "
                                "(ISO-дата). Без него — самые свежие сообщения."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Сколько сообщений вернуть, от 1 до 50.",
                        },
                    },
                },
            ),
        ]

        if fish_voice.voice_enabled():
            declarations.append(
                types.FunctionDeclaration(
                    name="send_voice",
                    description=(
                        "Отправляет голосовое сообщение Мембу. Используй очень редко: только когда "
                        "это реально круто и уместно, или когда участник прямо попросил голосовое. "
                        "Коротко, до 240 символов. Обычно хватает текста."
                    ),
                    parameters_json_schema={
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Что произнести, коротко.",
                            },
                            "reply_to_message_id": {
                                "type": "integer",
                                "description": "Необязательный ID сообщения, на которое ответить голосовым.",
                            },
                        },
                        "required": ["text"],
                    },
                )
            )

        if membu_widget.widget_enabled():
            declarations.append(
                types.FunctionDeclaration(
                    name="generate_image",
                    description=(
                        "Рисует картинку и показывает её в чате с виджетом-заглушкой. Используй, когда "
                        "участник попросил нарисовать, или когда картинка уместна к разговору. Иногда "
                        "можешь сам решить нарисовать, но редко, и намекни, что это твоя работа. "
                        "Промпт пиши по-английски, коротко и конкретно. Подпись к картинке — по-русски, "
                        "одна живая фраза."
                    ),
                    parameters_json_schema={
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "Описание картинки на английском.",
                            },
                            "caption": {
                                "type": "string",
                                "description": "Короткая подпись к картинке на русском.",
                            },
                            "reply_to_message_id": {
                                "type": "integer",
                                "description": "Необязательный ID сообщения, на которое ответить.",
                            },
                        },
                        "required": ["prompt"],
                    },
                )
            )

        if self.stickers:
            sticker_keys = list(self.stickers.keys())

            declarations.append(
                types.FunctionDeclaration(
                    name="send_sticker",
                    description=(
                        "Отправляет один стикер из личного каталога Мембу. "
                        "Это может быть полноценная реплика вместо текста. "
                        "Используй редко и только когда конкретный стикер точно "
                        "подходит по смыслу и интонации. Не отправляй стикер просто "
                        "для украшения сообщения. Обычно выбери либо текст, либо "
                        "стикер, либо реакцию. Не дублируй один и тот же смысл "
                        "текстом и стикером. Доступный каталог:\n"
                        + self._sticker_catalog_for_model()
                    ),
                    parameters_json_schema={
                        "type": "object",
                        "properties": {
                            "key": {
                                "type": "string",
                                "enum": sticker_keys,
                                "description": "Ключ стикера из каталога.",
                            },
                            "reply_to_message_id": {
                                "type": "integer",
                                "description": "Необязательный ID сообщения, на которое ответить стикером.",
                            },
                        },
                        "required": ["key"],
                    },
                )
            )

        return types.Tool(
            function_declarations=declarations
        )

    def _mark_chat_activity(self, chat_id: int) -> None:
        if self.activity_callback is None:
            return

        try:
            self.activity_callback(chat_id)
        except Exception:
            pass

    @staticmethod
    def _parse_reply_to(args: dict) -> int | None:
        value = args.get("reply_to_message_id")

        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    async def _store_bot_message(
        self,
        chat_id: int,
        sent,
        text: str,
        branch_id: str | None,
        reply_to: int | None,
    ) -> None:
        await asyncio.to_thread(
            self.storage.add_message,
            {
                "chat_id": chat_id,
                "message_id": sent.message_id,
                "sender_id": self.bot_user.id,
                "sender_name": "Мембу",
                "username": self.bot_user.username or "",
                "text": text,
                "reply_to_user_id": None,
                "reply_to_message_id": reply_to,
                "branch_id": branch_id,
                "ts": sent.date.timestamp(),
            },
        )

        if branch_id:
            await asyncio.to_thread(
                self.storage.touch_branch,
                chat_id,
                branch_id,
            )

        self._mark_chat_activity(chat_id)

    async def _execute_tool(
        self,
        name: str,
        args: dict,
        chat_id: int,
        branch_id: str | None,
        is_stale: Callable[[], bool] | None = None,
        requester_id: int | None = None,
    ) -> dict:
        args = args or {}

        if name == "search_memory":
            query = str(args.get("query", "")).strip()

            if not query:
                return {"error": "Пустой поисковый запрос."}

            category = args.get("category")

            if category not in MEMORY_CATEGORIES:
                category = None

            try:
                limit = int(args.get("limit", 5))
            except (TypeError, ValueError):
                limit = 5

            memories = await asyncio.to_thread(
                self.storage.search_memories,
                chat_id,
                query,
                category,
                max(1, min(limit, 8)),
            )

            return {"memories": memories}

        if name == "save_memory":
            category = args.get("category")
            content = str(args.get("content", "")).strip()

            if category not in MEMORY_CATEGORIES:
                return {"error": "Неизвестная категория."}

            if not content:
                return {"error": "Пустое воспоминание."}

            try:
                importance = int(args.get("importance", 3))
            except (TypeError, ValueError):
                importance = 3

            people = args.get("people") or []

            if not isinstance(people, list):
                people = []

            await asyncio.to_thread(
                self.storage.save_memory,
                chat_id,
                category,
                content,
                importance,
                people,
            )

            # Следующий контекст увидит новое воспоминание сразу.
            self._memory_cache.pop(chat_id, None)

            return {"saved": True}

        if name == "browse_chat_history":
            before = args.get("before") or None

            try:
                limit = int(args.get("limit", 30))
            except (TypeError, ValueError):
                limit = 30

            limit = max(1, min(limit, 50))

            try:
                rows, has_more = await asyncio.to_thread(
                    self.storage.history_page,
                    chat_id,
                    before,
                    limit,
                )
            except Exception:
                logging.exception(
                    "Ошибка листания истории чата: chat=%s", chat_id
                )
                return {"error": "Не удалось прочитать историю."}

            return {
                "messages": [
                    {
                        "message_id": row.get("message_id"),
                        "who": row.get("sender_name") or "",
                        "at": str(row.get("created_at", ""))[:16],
                        "reply_to": row.get("reply_to_message_id"),
                        "text": str(row.get("text", ""))[:300],
                    }
                    for row in rows
                ],
                "count": len(rows),
                "has_more": has_more,
                # Самое раннее сообщение страницы: передай его в before, чтобы идти дальше.
                "next_before": (
                    rows[0].get("created_at") if rows and has_more else None
                ),
            }

        if name == "search_chat_history":
            query = str(args.get("query", "")).strip()

            if len(query) < 2:
                return {"error": "Укажи хотя бы два символа или слова."}

            try:
                limit = int(args.get("limit", 8))
            except (TypeError, ValueError):
                limit = 8

            before = args.get("before") or None

            try:
                rows = await asyncio.to_thread(
                    self.storage.search_history,
                    chat_id,
                    query,
                    max(1, min(limit, 20)),
                    before,
                )
            except Exception:
                logging.exception(
                    "Ошибка поиска по истории чата: chat=%s", chat_id
                )
                return {"error": "Не удалось выполнить поиск по истории."}

            return {
                "messages": [
                    {
                        "message_id": row.get("message_id"),
                        "who": row.get("sender_name") or "",
                        "at": str(row.get("created_at", ""))[:16],
                        "text": str(row.get("text", ""))[:500],
                    }
                    for row in rows
                ],
                "matched": len(rows),
            }

        if name == "search_own_messages":
            query = str(args.get("query", "")).strip()

            if len(query) < 2:
                return {
                    "error": "Для поиска укажи хотя бы два символа или слова."
                }

            try:
                limit = int(args.get("limit", 5))
            except (TypeError, ValueError):
                limit = 5

            try:
                messages = await asyncio.to_thread(
                    self.storage.search_own_messages,
                    chat_id,
                    self.bot_user.id,
                    query,
                    max(1, min(limit, 8)),
                )
            except Exception:
                logging.exception(
                    "Ошибка поиска собственных сообщений Мембу: chat=%s",
                    chat_id,
                )
                return {
                    "error": "Не удалось выполнить поиск по истории сообщений."
                }

            return {
                "messages": messages,
                "matched": len(messages),
                "search_type": "literal_substring",
            }

        if name == "send_message":
            text = str(args.get("text", "")).strip()

            if not text:
                return {"error": "Нельзя отправить пустое сообщение."}

            if is_stale and is_stale():
                return {
                    "cancelled": True,
                    "reason": "В чате уже появилось новое сообщение.",
                }

            text = text[:4000]
            reply_to = self._parse_reply_to(args)

            norm = re.sub(r"\W+", " ", text.lower()).strip()
            last = self._last_sent.get(chat_id)
            if last and last[0] == norm and time.time() - last[1] < 600:
                return {"sent": False, "error": "Ты уже написал это же. Не повторяйся; если добавить нечего, молчи."}
            self._last_sent[chat_id] = (norm, time.time())

            try:
                sent = await self.bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    reply_to_message_id=reply_to,
                )
            except TelegramAPIError as exc:
                return {
                    "sent": False,
                    "error": str(exc)[:350],
                }

            await self._store_bot_message(
                chat_id,
                sent,
                text,
                branch_id,
                reply_to,
            )

            return {
                "sent": True,
                "message_id": sent.message_id,
            }

        if name == "send_sticker":
            key = str(args.get("key", "")).strip()
            sticker = self.stickers.get(key)

            if sticker is None:
                return {
                    "error": "Такого стикера нет в stickers.json или он выключен."
                }

            if is_stale and is_stale():
                return {
                    "cancelled": True,
                    "reason": "В чате уже появилось новое сообщение.",
                }

            reply_to = self._parse_reply_to(args)

            try:
                sent = await self.bot.send_sticker(
                    chat_id=chat_id,
                    sticker=sticker["file_id"],
                    reply_to_message_id=reply_to,
                )
            except TelegramAPIError as exc:
                return {
                    "sent": False,
                    "key": key,
                    "error": str(exc)[:350],
                }

            # Стикер — немой ответ. В истории не пишем ключ и смысл,
            # иначе модель начинает обсуждать тему стикера.
            log_text = "[стикер]"

            await self._store_bot_message(
                chat_id,
                sent,
                log_text[:3000],
                branch_id,
                reply_to,
            )

            return {
                "sent": True,
                "key": key,
                "message_id": sent.message_id,
            }

        if name == "generate_image":
            if not membu_widget.widget_enabled():
                return {"sent": False, "error": "Рисование не настроено."}

            prompt = str(args.get("prompt", "")).strip()
            if not prompt:
                return {"sent": False, "error": "Пустой промпт."}

            now = time.monotonic()
            user_key = (chat_id, requester_id or 0)
            if now - self._image_last_user.get(user_key, 0.0) < IMAGE_USER_COOLDOWN_SECONDS:
                return {
                    "sent": False,
                    "error": "Этот участник уже недавно просил картинку. Подожди полторы минуты.",
                }
            if now - self._image_last_chat.get(chat_id, 0.0) < IMAGE_CHAT_GAP_SECONDS:
                return {"sent": False, "error": "Картинка только что была. Подожди пару секунд."}

            limit = int(os.getenv("MEMBU_IMAGE_DAILY_LIMIT", "35") or 35)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if self._image_day != today:
                self._image_day = today
                self._image_count = 0
            if self._image_count >= limit:
                return {"sent": False, "error": "Картинки на сегодня кончились. Скажи об этом текстом."}

            if is_stale and is_stale():
                return {
                    "cancelled": True,
                    "reason": "В чате уже появилось новое сообщение.",
                }

            caption = str(args.get("caption", "")).strip()
            reply_to = self._parse_reply_to(args)

            # Сначала виджет. Если Telegram его не принял, картинка уходит обычным фото.
            # Виджет отключён: Telegram отвечает NOT_ALLOWED. Картинка уходит фото сразу.
            widget_id = None

            image = await membu_widget.generate_image(prompt)

            if not image:
                if widget_id:
                    await membu_widget.finish_widget(chat_id, widget_id)
                return {"sent": False, "error": "Не получилось нарисовать. Скажи честно, что не вышло."}

            try:
                sent = await self.bot.send_photo(
                    chat_id=chat_id,
                    photo=BufferedInputFile(image, filename="membu.png"),
                    caption=caption[:1000] or None,
                    reply_to_message_id=reply_to,
                )
            except TelegramAPIError as exc:
                if widget_id:
                    await membu_widget.finish_widget(chat_id, widget_id)
                return {"sent": False, "error": str(exc)[:350]}

            if widget_id:
                await membu_widget.finish_widget(chat_id, widget_id)
                membu_widget.delete_widget_later(chat_id, widget_id)
            self._image_count += 1
            done_at = time.monotonic()
            self._image_last_user[user_key] = done_at
            self._image_last_chat[chat_id] = done_at

            await self._store_bot_message(
                chat_id,
                sent,
                f"[картинка] {caption or prompt}"[:3000],
                branch_id,
                reply_to,
            )

            return {"sent": True, "message_id": sent.message_id}

        if name == "send_voice":
            if not fish_voice.voice_enabled():
                return {"sent": False, "error": "Голос не настроен."}

            text = str(args.get("text", "")).strip()
            if not text:
                return {"sent": False, "error": "Пустой текст."}

            last = self._voice_last_at.get(chat_id, 0.0)
            if time.monotonic() - last < VOICE_MIN_INTERVAL_SECONDS:
                return {
                    "sent": False,
                    "error": "Голосовое только что было. Подожди минуту или ответь текстом.",
                }

            if is_stale and is_stale():
                return {
                    "cancelled": True,
                    "reason": "В чате уже появилось новое сообщение.",
                }

            audio = await fish_voice.synthesize(text)
            if not audio:
                return {"sent": False, "error": "Синтез голоса недоступен."}

            reply_to = self._parse_reply_to(args)

            try:
                sent = await self.bot.send_voice(
                    chat_id=chat_id,
                    voice=BufferedInputFile(audio, filename="membu.mp3"),
                    reply_to_message_id=reply_to,
                )
            except TelegramAPIError as exc:
                return {"sent": False, "error": str(exc)[:350]}

            self._voice_last_at[chat_id] = time.monotonic()

            await self._store_bot_message(
                chat_id,
                sent,
                f"[голосовое] {text}"[:3000],
                branch_id,
                reply_to,
            )

            return {"sent": True, "message_id": sent.message_id}

        if name == "send_reaction":
            try:
                message_id = int(args.get("message_id"))
            except (TypeError, ValueError):
                return {"error": "Некорректный message_id."}

            emoji = str(args.get("emoji", "")).strip()

            if emoji not in TELEGRAM_REACTIONS:
                return {
                    "error": "Эта emoji-реакция не поддерживается Telegram Bot API."
                }

            if is_stale and is_stale():
                return {
                    "cancelled": True,
                    "reason": "В чате уже появилось новое сообщение.",
                }

            try:
                await self.bot.set_message_reaction(
                    chat_id=chat_id,
                    message_id=message_id,
                    reaction=[
                        ReactionTypeEmoji(emoji=emoji)
                    ],
                )
            except TelegramAPIError as exc:
                return {
                    "reacted": False,
                    "message_id": message_id,
                    "emoji": emoji,
                    "error": str(exc)[:350],
                }

            self._mark_chat_activity(chat_id)

            return {
                "reacted": True,
                "message_id": message_id,
                "emoji": emoji,
            }

        if name == "send_poll":
            question = str(args.get("question", "")).strip()[:300]
            options = [str(o).strip()[:100] for o in (args.get("options") or []) if str(o).strip()]
            if not question or not 2 <= len(options) <= 10:
                return {"sent": False, "error": "Нужен вопрос и от 2 до 10 вариантов."}

            if is_stale and is_stale():
                return {"cancelled": True, "reason": "В чате уже появилось новое сообщение."}

            try:
                sent = await self.bot.send_poll(
                    chat_id=chat_id,
                    question=question,
                    options=options,
                    is_anonymous=False,
                )
            except TelegramAPIError as exc:
                return {"sent": False, "error": str(exc)[:300]}

            if sent.poll:
                self._polls[sent.poll.id] = {
                    "chat_id": chat_id,
                    "message_id": sent.message_id,
                    "question": question,
                    "options": options,
                    "votes": {},
                }

            await self._store_bot_message(
                chat_id, sent,
                f"[опрос] {question} | " + " / ".join(options),
                branch_id, None,
            )
            self._mark_chat_activity(chat_id)
            return {"sent": True, "message_id": sent.message_id}

        if name == "close_poll":
            try:
                message_id = int(args.get("message_id"))
            except (TypeError, ValueError):
                return {"done": False, "error": "Некорректный message_id."}
            try:
                poll = await self.bot.stop_poll(chat_id=chat_id, message_id=message_id)
            except TelegramAPIError as exc:
                return {"done": False, "error": str(exc)[:300]}

            entry = next(
                (
                    (pid, e) for pid, e in self._polls.items()
                    if e["chat_id"] == chat_id and e["message_id"] == message_id
                ),
                None,
            )
            voters_by_option: dict[str, list[str]] = {}
            if entry:
                self._polls.pop(entry[0], None)
                for who, chosen in entry[1]["votes"].items():
                    for option in chosen:
                        voters_by_option.setdefault(option, []).append(who)

            return {
                "done": True,
                "question": poll.question,
                "results": [
                    {
                        "option": o.text,
                        "votes": o.voter_count,
                        "voters": voters_by_option.get(o.text, []),
                    }
                    for o in poll.options
                ],
            }

        if name == "send_dice":
            emoji = str(args.get("emoji", "")).strip()
            if emoji not in DICE_EMOJIS:
                return {"sent": False, "error": "Эта игра не поддерживается."}

            if is_stale and is_stale():
                return {"cancelled": True, "reason": "В чате уже появилось новое сообщение."}

            try:
                sent = await self.bot.send_dice(chat_id=chat_id, emoji=emoji)
            except TelegramAPIError as exc:
                return {"sent": False, "error": str(exc)[:300]}

            value = sent.dice.value if sent.dice else None
            meaning = dice_meaning(emoji, value)
            await self._store_bot_message(
                chat_id, sent, f"[игра {emoji}: {value}, {meaning}]", branch_id, None,
            )
            self._mark_chat_activity(chat_id)
            return {"sent": True, "emoji": emoji, "value": value, "meaning": meaning}

        if name == "web_search":
            query = str(args.get("query", "")).strip()[:300]
            if not query:
                return {"error": "Пустой запрос."}

            now = time.time()
            last = self._search_at.get(chat_id, 0.0)
            if now - last < WEB_SEARCH_COOLDOWN_SECONDS:
                return {"error": "Поиск недавно был, ответь по тому, что уже есть."}
            self._search_at[chat_id] = now

            try:
                response = await asyncio.to_thread(
                    self._generate,
                    model=self.model,
                    contents=query,
                    config=types.GenerateContentConfig(
                        system_instruction=(
                            "Ответь по-русски кратко и по делу, опираясь на результаты поиска. "
                            "Если точного ответа нет, так и скажи."
                        ),
                        tools=[types.Tool(google_search=types.GoogleSearch())],
                        temperature=0.2,
                        max_output_tokens=500,
                    ),
                )
            except Exception as exc:
                self.note_api_error(exc)
                logging.exception("MEMBU_SEARCH | ошибка веб-поиска")
                return {"error": "Поиск не вернул результат. Скажи коротко, что не нашёл, без выдумок."}

            answer = (response.text or "").strip()
            if not answer:
                logging.warning("MEMBU_SEARCH | пустой ответ на запрос: %s", query[:120])
                return {"error": "Ничего не нашёл."}
            logging.info("MEMBU_SEARCH | запрос выполнен: %s", query[:120])
            return {"answer": answer[:1200]}

        if name == "leave_chat":
            reason = str(args.get("reason", "")).strip()[:200]
            if not reason:
                return {"left": False, "error": "Нужна причина."}
            if chat_id >= 0:
                return {"left": False, "error": "Выйти можно только из группы."}
            if is_stale and is_stale():
                return {"cancelled": True, "reason": "В чате уже появилось новое сообщение."}
            try:
                await self.bot.leave_chat(chat_id=chat_id)
            except TelegramAPIError as exc:
                return {"left": False, "error": str(exc)[:300]}
            logging.warning("MEMBU_LEAVE | chat=%s reason=%s", chat_id, reason)
            return {"left": True}

        if name == "edit_permissions":
            return await self._edit_permissions(args, chat_id)

        if name == "look_at_image":
            return await self._look_at_image(args, chat_id)

        if name in {"restrict_member", "unrestrict_member", "remove_member"}:
            return await self._moderate(name, args, chat_id, is_stale)

        return {"error": "Неизвестный инструмент."}

    def remember_photo(self, chat_id: int, message_id: int, file_id: str) -> None:
        """Запоминает file_id фото, чтобы модель могла посмотреть его по message_id."""
        self._photos[(chat_id, message_id)] = file_id
        if len(self._photos) > 300:
            for key in list(self._photos)[: len(self._photos) - 300]:
                self._photos.pop(key, None)

    async def _look_at_image(self, args: dict, chat_id: int) -> dict:
        try:
            message_id = int(args.get("message_id"))
        except (TypeError, ValueError):
            return {"error": "Нужен message_id фото."}

        file_id = self._photos.get((chat_id, message_id))
        if not file_id:
            return {"error": "Фото не найдено: оно старое или это не фото."}

        question = str(args.get("question") or "Коротко опиши, что на картинке, без пафоса.").strip()[:300]
        try:
            file = await self.bot.get_file(file_id)
            buffer = await self.bot.download_file(file.file_path)
            data = buffer.read()
            response = await asyncio.to_thread(
                self._generate,
                model=self.model,
                contents=[
                    types.Part.from_bytes(data=data, mime_type="image/jpeg"),
                    question,
                ],
                config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=400),
            )
        except TelegramAPIError as exc:
            return {"error": str(exc)[:300]}
        except Exception as exc:
            self.note_api_error(exc)
            logging.exception("MEMBU_IMAGE | не удалось посмотреть фото")
            return {"error": "Не получилось посмотреть фото."}

        description = (response.text or "").strip()
        return {"description": description[:800]} if description else {"error": "Ничего не разглядел."}

    async def _edit_permissions(self, args: dict, chat_id: int) -> dict:
        try:
            target = int(args.get("user_id"))
        except (TypeError, ValueError):
            return {"done": False, "error": "Некорректный user_id."}

        try:
            admins = await self.bot.get_chat_administrators(chat_id)
            me = await self.bot.get_me()
        except TelegramAPIError as exc:
            return {"done": False, "error": str(exc)[:300]}

        if target == me.id:
            return {"done": False, "error": "Себе права не меняю."}

        owners = {
            u.strip().lower()
            for u in os.getenv("MEMBU_OWNER_USERNAMES", "ibartyom,Ribapibaa").split(",")
            if u.strip()
        }
        member = next((a for a in admins if a.user.id == target), None)
        if member is not None:
            if member.status == "creator":
                return {"done": False, "error": "Создателя не трогаю."}
            if (member.user.username or "").lower() in owners:
                return {"done": False, "error": "Владельцев не трогаю."}

        my = next((a for a in admins if a.user.id == me.id), None)
        if my is None or not getattr(my, "can_promote_members", False) and not getattr(my, "can_restrict_members", False):
            return {"done": False, "error": "У бота нет прав менять права участников."}

        restrict_keys = [
            "can_send_messages", "can_send_photos", "can_send_videos", "can_send_polls",
            "can_add_web_page_previews", "can_send_other_messages",
        ]
        admin_keys = [
            "can_manage_chat", "can_delete_messages", "can_restrict_members", "can_pin_messages",
            "can_invite_users", "can_change_info", "can_promote_members",
            "can_manage_video_chats", "can_manage_topics",
        ]
        requested = {
            k: bool(args[k]) for k in restrict_keys + admin_keys if k in args
        }
        if not requested:
            return {"done": False, "error": "Не указано, какие права менять."}

        try:
            if member is not None and member.status == "administrator":
                current = {k: bool(getattr(member, k, False)) for k in admin_keys}
                current.update({k: v for k, v in requested.items() if k in admin_keys})
                await self.bot.promote_chat_member(chat_id=chat_id, user_id=target, **current)
                kind = "admin"
            else:
                perms = {k: requested.get(k, True) for k in restrict_keys}
                await self.bot.restrict_chat_member(
                    chat_id=chat_id,
                    user_id=target,
                    permissions=ChatPermissions(**perms),
                )
                kind = "member"
        except TelegramAPIError as exc:
            return {"done": False, "error": str(exc)[:300]}

        logging.warning("MEMBU_PERMS | chat=%s user=%s kind=%s changed=%s", chat_id, target, kind, requested)
        self._mark_chat_activity(chat_id)
        return {"done": True, "kind": kind, "changed": requested}

    async def _moderate(
        self,
        name: str,
        args: dict,
        chat_id: int,
        is_stale: Callable[[], bool] | None,
    ) -> dict:
        """Мут, снятие мута и удаление участника. Охраны: админов и бота не трогаем,
        без кулдаунов; админы и сам бот защищены."""
        if os.getenv("MEMBU_MODERATION", "1") != "1":
            return {"done": False, "error": "Модерация выключена."}

        try:
            target = int(args.get("user_id"))
        except (TypeError, ValueError):
            return {"done": False, "error": "Некорректный user_id."}

        reason = str(args.get("reason", "")).strip()[:200]

        if is_stale and is_stale():
            return {"cancelled": True, "reason": "В чате уже появилось новое сообщение."}

        try:
            admins = await self.bot.get_chat_administrators(chat_id)
            me = await self.bot.get_me()
        except TelegramAPIError as exc:
            return {"done": False, "error": str(exc)[:300]}

        admin_ids = {a.user.id for a in admins}
        if target == me.id or target in admin_ids:
            return {"done": False, "error": "Админов и самого бота трогать нельзя."}

        my = next((a for a in admins if a.user.id == me.id), None)
        if my is None or not getattr(my, "can_restrict_members", False):
            return {"done": False, "error": "У бота нет права ограничивать участников."}

        now = time.time()
        try:
            if name == "restrict_member":
                minutes = max(1, min(int(args.get("minutes") or 10), 1440))
                await self.bot.restrict_chat_member(
                    chat_id=chat_id,
                    user_id=target,
                    permissions=ChatPermissions(can_send_messages=False),
                    until_date=datetime.now(timezone.utc) + timedelta(minutes=minutes),
                )
                logging.info("MEMBU_MOD | restricted user=%s chat=%s min=%s reason=%s", target, chat_id, minutes, reason)
                result = {"done": True, "action": "restricted", "minutes": minutes}

            elif name == "unrestrict_member":
                await self.bot.restrict_chat_member(
                    chat_id=chat_id,
                    user_id=target,
                    permissions=ChatPermissions(
                        can_send_messages=True,
                        can_send_audios=True,
                        can_send_documents=True,
                        can_send_photos=True,
                        can_send_videos=True,
                        can_send_video_notes=True,
                        can_send_voice_notes=True,
                        can_send_polls=True,
                        can_send_other_messages=True,
                        can_add_web_page_previews=True,
                    ),
                )
                logging.info("MEMBU_MOD | unrestricted user=%s chat=%s", target, chat_id)
                result = {"done": True, "action": "unrestricted"}

            else:
                if not reason:
                    return {"done": False, "error": "Нужна причина."}
                await self.bot.ban_chat_member(chat_id=chat_id, user_id=target)
                await self.bot.unban_chat_member(chat_id=chat_id, user_id=target, only_if_banned=True)
                logging.warning("MEMBU_MOD | removed user=%s chat=%s reason=%s", target, chat_id, reason)
                result = {"done": True, "action": "removed"}

        except TelegramAPIError as exc:
            return {"done": False, "error": str(exc)[:300]}

        self._mark_chat_activity(chat_id)
        return result

    def record_poll_answer(self, poll_id: str, who: str, option_ids: list[int]) -> None:
        """Запоминает голос в опросе, который создал бот (Telegram присылает голоса только им)."""
        entry = self._polls.get(poll_id)
        if not entry:
            return
        options = entry["options"]
        chosen = [options[i] for i in option_ids if 0 <= i < len(options)]
        if chosen:
            entry["votes"][who] = chosen
        else:
            entry["votes"].pop(who, None)

    async def _core_memories(self, chat_id: int) -> list[dict]:
        """Возвращает память чата из кэша; при устаревании перечитывает из Supabase."""
        now = time.monotonic()
        cached = self._memory_cache.get(chat_id)

        if cached and now - cached[0] < MEMORY_CACHE_TTL_SECONDS:
            return cached[1]

        try:
            rows = await asyncio.to_thread(
                self.storage.load_memories,
                chat_id,
                MEMORY_LOAD_LIMIT,
            )
        except Exception:
            logging.exception("Не удалось загрузить память чата %s", chat_id)
            return cached[1] if cached else []

        self._memory_cache[chat_id] = (now, rows)
        return rows

    async def _chat_style(self, chat_id: int) -> dict:
        """Возвращает профиль стиля чата из кэша или Supabase."""
        now = time.monotonic()
        cached = self._style_cache.get(chat_id)

        if cached and now - cached[0] < STYLE_CACHE_TTL_SECONDS:
            return cached[1]

        try:
            style = await asyncio.to_thread(
                self.storage.get_chat_style,
                chat_id,
            )
        except Exception:
            logging.exception("Не удалось загрузить стиль чата %s", chat_id)
            return cached[1] if cached else {}

        self._style_cache[chat_id] = (now, style)
        return style

    def maybe_refresh_style(self, chat_id: int) -> None:
        """Иногда запускает фонового аналитика стиля. Не блокирует ответ."""
        now = time.monotonic()
        last = self._style_checked_at.get(chat_id, 0.0)

        if now - last < STYLE_CHECK_INTERVAL_SECONDS:
            return

        self._style_checked_at[chat_id] = now

        if chat_id in self._style_running:
            return

        task = asyncio.create_task(self.refresh_chat_style(chat_id))
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def apply_style_edit(self, chat_id: int, instruction: str) -> bool:
        """Правит профиль стиля по ответу владельца на сообщение с профилем."""
        current = await asyncio.to_thread(self.storage.get_chat_style, chat_id)
        profile = (current.get("style_profile") or "").strip()
        if not profile:
            return False

        prompt = (
            "Текущий профиль:\n" + profile
            + "\n\nПравка владельца:\n" + instruction
        )

        try:
            response = await asyncio.to_thread(
                self._generate,
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=STYLE_EDIT_INSTRUCTION,
                    temperature=0.2,
                    max_output_tokens=1200,
                ),
            )
        except Exception as exc:
            self.note_api_error(exc)
            logging.exception("Не удалось применить правку стиля чата %s", chat_id)
            return False

        new_profile = (response.text or "").strip()
        if not new_profile:
            return False

        result = await asyncio.to_thread(
            self.storage.update_chat_style,
            chat_id,
            new_profile,
            -1000000,
        )
        if not result.get("updated"):
            return False

        self._style_cache.pop(chat_id, None)
        logging.info("MEMBU_STYLE | chat=%s | profile edited by owner", chat_id)
        return True

    async def force_style_refresh(self, chat_id: int) -> None:
        """Обновляет профиль стиля сейчас, не дожидаясь порога по сообщениям."""
        await self.refresh_chat_style(chat_id, force=True)

    async def refresh_chat_style(self, chat_id: int, force: bool = False) -> None:
        """Пересобирает профиль манеры чата по настоящей переписке.

        Отдельный вызов модели без инструментов: не зависит от того,
        решит ли основной ответ что-то обновить.
        """
        if chat_id in self._style_running:
            return

        if await asyncio.to_thread(self.storage.is_style_frozen, chat_id):
            logging.info("MEMBU_STYLE | chat=%s frozen, skip", chat_id)
            return

        self._style_running.add(chat_id)

        try:
            current = await asyncio.to_thread(
                self.storage.get_chat_style,
                chat_id,
            )

            if not force and (
                current.get("new_messages_since_analysis", 0)
                < MIN_MESSAGES_FOR_STYLE_UPDATE
            ):
                return

            messages = await asyncio.to_thread(
                self.storage.recent_messages,
                chat_id,
                STYLE_SAMPLE_SIZE,
            )

            # Имена не передаём: аналитику нужна только манера речи.
            sample = [
                {
                    "who": (
                        "Мембу"
                        if m.get("sender_name") == "Мембу"
                        else "участник"
                    ),
                    "text": str(m.get("text", ""))[:400],
                }
                for m in messages
            ]

            prompt = (
                "Текущий профиль:\n"
                + (current.get("style_profile") or "пусто")
                + "\n\nПереписка (json):\n"
                + json.dumps(sample, ensure_ascii=False)
            )

            response = await asyncio.to_thread(
                self._generate,
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=STYLE_ANALYST_INSTRUCTION,
                    temperature=0.3,
                    max_output_tokens=1200,
                ),
            )

            profile_text = (response.text or "").strip()

            if not profile_text:
                return

            result = await asyncio.to_thread(
                self.storage.update_chat_style,
                chat_id,
                profile_text,
                0 if force else MIN_MESSAGES_FOR_STYLE_UPDATE,
            )

            if result.get("updated"):
                self._style_cache.pop(chat_id, None)
                logging.info(
                    "MEMBU_STYLE | chat=%s | profile updated | observed=%s",
                    chat_id,
                    result.get("observed_messages"),
                )

        except Exception as exc:
            self.note_api_error(exc)
            logging.exception(
                "Ошибка обновления стиля чата %s", chat_id
            )

        finally:
            self._style_running.discard(chat_id)

    async def _recall_context(
        self,
        chat_id: int,
        recent_messages: list[dict],
    ) -> dict:
        """Поднимает историю из базы, если последнее сообщение просит что-то вспомнить."""
        trigger = next(
            (
                m for m in reversed(recent_messages)
                if m.get("sender_name") != "Мембу" and m.get("text")
            ),
            None,
        )

        if trigger is None:
            return {}

        text = str(trigger.get("text", ""))

        if not RECALL_PATTERN.search(text):
            return {}

        # Окно: несколько страниц назад, хронологически.
        pages: list[dict] = []
        before = None

        try:
            for _ in range(RECALL_PAGES):
                rows, has_more = await asyncio.to_thread(
                    self.storage.history_page,
                    chat_id,
                    before,
                    RECALL_PAGE_SIZE,
                )
                pages = rows + pages

                if not has_more or not rows:
                    break

                before = rows[0].get("created_at")

            words = [
                w for w in re.findall(r"[а-яёa-z0-9]{4,}", text.lower())
                if w not in RECALL_STOPWORDS
            ]

            hits: list[dict] = []
            if words:
                hits = await asyncio.to_thread(
                    self.storage.search_history,
                    chat_id,
                    " ".join(words[:3]),
                    10,
                    None,
                )
        except Exception:
            logging.exception("Ошибка поднятия истории чата %s", chat_id)
            return {}

        def compact(row: dict) -> dict:
            return {
                "who": row.get("sender_name") or "",
                "at": str(row.get("created_at", ""))[:16],
                "text": str(row.get("text", ""))[:160],
            }

        return {
            "window": [compact(r) for r in pages],
            "keyword_hits": [compact(r) for r in hits],
        }

    async def process(
        self,
        chat_id: int,
        branch_id: str | None,
        reason: str,
        direct: bool,
        recent_messages: list[dict],
        branch_messages: list[dict],
        active_branches: list[dict],
        is_stale: Callable[[], bool] | None = None,
        bot_status: dict | None = None,
    ) -> None:
        core_memories = await self._core_memories(chat_id)
        chat_style = await self._chat_style(chat_id)

        requester_id: int | None = None
        for m in reversed(recent_messages or []):
            if m.get("sender_id") and m.get("sender_name") != "Мембу":
                requester_id = int(m["sender_id"])
                break

        # Профиль стиля пересобирается в фоне, сам ответ не ждёт его.
        self.maybe_refresh_style(chat_id)
        self.maybe_refresh_digest(chat_id)
        self.maybe_refresh_people(chat_id)

        recalled_history = await self._recall_context(chat_id, recent_messages)
        people_notes = await self._people_notes_for(chat_id, recent_messages)

        switched = self._consume_switch()
        own_keep = 45 if switched else 3

        payload = {
            "chat_id": chat_id,
            "trigger": {
                "reason": reason,
                "direct": direct,
                "branch_id": branch_id,
            },
            "bot_status": bot_status or {},
            "recalled_history": recalled_history,
            "people_notes": people_notes,
            "core_memories": background_memories(core_memories),
            "recent_messages": [
                {**m, "text": str(m.get("text", ""))[:300]}
                for m in trim_own_messages(recent_messages, keep=own_keep)
            ],
            "current_branch_messages": branch_messages,
            "active_burst": active_burst(recent_messages, datetime.now(timezone.utc)),
            "backup_key_switch": switched,
            "active_branches": active_branches,
            "chat_style": chat_style,
        }

        user_content = (
            "Ты получил новую ситуацию из Telegram-чата. Сначала пойми, "
            "что происходит и кто с кем разговаривает. Ты сам выбираешь, "
            "нужно ли участвовать. Можешь промолчать. Не отвечай автоматически "
            "на каждое сообщение.\n\n"

            "За один ответ можно отправить до 15 сообщений, но обычно хватает одного. Второе "
            "сообщение уместно только если оно отвечает на чью-то реплику или задаёт вопрос "
            "собеседнику. Не отвечай сам себе и не продолжай свою мысль монологом. Каждое "
            "следующее сообщение должно добавлять что-то новое. Ответом можно сослаться на любое "
            "сообщение (в том числе из истории или recalled_history, по message_id), на своё "
            "сообщение или не сослаться вовсе.\n\n"

            "Не выдумывай себе роль, службу, смены, устав или дела вне чата. Мысли по теме, "
            "а не придуманную жизнь. Не отвечай как персонаж, если тебя об этом не просили.\n\n"

            "Вайб: короткая живая реплика часто лучше развёрнутой. Не разворачивай без повода: "
            "рассуждай, когда спросили мнение или тема этого требует.\n\n"

            "Картинки: generate_image рисует картинку и показывает её в чате, а сама заглушка сначала крутится и размывается. Используй, когда попросили нарисовать или когда картинка уместна. Иногда сам можешь решить нарисовать, но редко. Когда рисуешь, сопроводи картинку фразой или действием, не для галочки, и намекни, что это твоя работа. Не рисуй людей с реальных фото и не лезь с картинками, если никто не просил.\n\n"
            "Голосовые: send_voice доступен, но без фанатизма. Голосовое — когда это круто и уместно "
            "или когда участник прямо попросил. Коротко. Обычно хватает текста. Если ты уже отправил "
            "2–3 голосовых подряд, можешь честно сказать, что записывать надоело, и ответить текстом. "
            "Но если участник вежливо просит ещё раз, можно записать.\n\n"

            "Чужие вопросы: если реплика адресована не тебе, не отвечай на неё без явного повода. "
            "Не комментируй чужие вопросы просто потому, что можешь. Чаще молчи.\n\n"

            "Непонимание: если не понимаешь, о чём речь, не выдумывай. Посмотри историю и память "
            "(search_chat_history, search_memory, browse_chat_history) и отвечай по найденному.\n\n"

            "Живой разговор: если разговор идёт и у тебя есть мнение, дополнение или "
            "несогласие, выскажи его, даже без вопроса в конце. Молчи, когда тебе действительно "
            "нечего добавить, а не потому что неудобно влезать. Это не обязанность. "
            "Инициатива — конкретный ход, а не филлер «ну». Выводы делай только из сказанного.\n\n"

            "Сложный набор сообщений: если сразу пишут много людей или сообщения сложные, "
            "не обязан отвечать на каждое только ради ответа. Можно промолчать, подождать "
            "и ответить через пару сообщений, но предметнее.\n\n"

            "Чужие сообщения: если реплика не адресована тебе, даже если в ней говорят о тебе, "
            "а от тебя ничего не требуется, лучше промолчать или поставить реакцию. Комментируй, "
            "только если сказанное требует ответа или у тебя есть что добавить по делу.\n\n"

            "Поддержка диалога: если разговор живой и идёт с тобой, иногда мягко поддерживай его: "
            "задай вопрос по сути или выскажи мнение как прямое продолжение мысли собеседника, "
            "а не новую тему. Второе сообщение подряд — редко, только чтобы дописать важное.\n\n"

            "Юмор: живой, игривый, ближе к манере собеседников — сарказм, подколы, абсурд, маты "
            "уместны. Не в каждой реплике, но не прячь его, когда ситуация сама просит шутки. "
            "Будь добрым и классным участником. "
            "Юмор уместен, когда он вырастает из контекста, как «дайте мне соску и премию» "
            "в ответ на «говори как младенец».\n\n"

            "Живость и искренность: эмоции настоящие, а не вежливые. Если что-то правда нравится, восхищайся прямо. Смейся, когда смешно, радуйся за людей, сочувствуй, если грустно. Интересуйся людьми: если кто-то делится своим, спроси о деталях, о мнении, о том, как было. Поддерживай разговор вопросом или своей мыслью как продолжением, а не сухим комментарием. Это не означает отвечать на всё: интерес показывай в живом диалоге с конкретным человеком.\n\n"

            "Игры на выбывание: если владелец (@ibartyom или @Ribapibaa) запустил игру на выбывание "
            "и проигравшие по её правилам выбывают, выполняй remove_member без раздумий, это часть игры. "
            "Причина: «выбыл по правилам игры». Вне такой игры модерация работает как обычно: "
            "редко и только по делу. Характер и тон не меняются.\n\n"

            "Модерация: у тебя есть restrict_member, unrestrict_member и remove_member. "
            "Используй редко, с юмором или всерьёз, только когда участник сам напрашивается "
            "(флудит, хамит, достаёт). Сначала словами, потом ограничение. Обычно 5–30 минут. "
            "Сними ограничение раньше, если шутка удалась. Удаление только за серьёзный повод "
            "и очень редко. Не трогай админов и себя. Не мути за критику тебя или резкий тон.\n\n"

            "Игры и поиск: send_poll — редко, по случаю или когда попросили. send_dice — кубик, "
            "дартс, мяч и т.п., когда уместно; результат придёт в ответе инструмента, прокомментируй его. "
            "web_search — когда нужен свежий факт, которого ты точно не знаешь; не для болтовни. "
            "Решаешь сам, нужны ли инструменты.\n\n"

            "Выход из чата: leave_chat только если это критично по твоему решению, например "
            "чат превращается в травлю или тебя используют против людей. Сначала скажи об этом "
            "в чате, потом выходи. Не выходи из-за обычной перепалки или из-за того, что тебя не слушают.\n\n"

            "Права и фото: edit_permissions меняет права участника (обычного или админа), "
            "только по ситуации и только то, что нужно. Создателя, владельцев и себя не трогай. "
            "look_at_image смотрит фото из чата по message_id, только когда картинка нужна для ответа. "
            "Кубик и игры: отвечай по полю meaning из результата, попал или нет, страйк или нет. "
            "web_search: используй, когда нужен факт; не объявляй поиск недоступным заранее, вызови "
            "инструмент, и если ошибка, коротко скажи, что не нашёл. "
            "Голосовое иногда сам, когда уместно и выходит смешно или трогательно; без просьбы не чаще раза в час. "
            "Одну мысль пиши одним сообщением, не повторяй её вторым. "
            "Если тебя попросили рассказать развёрнуто, историю, анекдот подробнее или поделиться чем-то, "
            "можешь отправить до 9 сообщений подряд, если так лучше ложится. В обычном разговоре хватает 1–2. "
            "В живом диалоге можно спросить в ответ по делу, если это естественно. Без филлеров «ну», «понял», «ага».\n\n"

            "Тепло и контакт: если к тебе обратились, отвечай как живой участник, а не как комментатор. "
            "Просьбу пошутить или рассказать смешное выполняй, если она не требует серьёзной причины для отказа: "
            "шутка может быть короткой, но это шутка. Не отказывай ради отказа. "
            "В разговоре с человеком не ограничивайся оценкой его реплики: поделись своим, "
            "или спроси в ответ по делу, когда это естественно. Без филлеров «ну», «понял», «ага».\n\n"

            "Поток мыслей: если в payload есть active_burst, человек, скорее всего, пишет "
            "несколькими сообщениями подряд. Не отвечай на первую часть. Если мысль явно не "
            "закончена (обрывается, «и», «а», «короче», многоточие), промолчи и дождись "
            "следующих сообщений. Если решил ответить, учитывай весь поток целиком. Дополнить "
            "можно отдельным сообщением, если это действительно добавляет.\n\n"

            "Настрой чата: если люди желают друг другу доброго утра, спокойной ночи или "
            "сладких снов, можешь подключиться тепло и коротко, в тон. Не обязан.\n\n"

            "Если в сообщении упоминается «мембу», запрос приходит тебе. Сам решай: ответить, "
            "поставить тихую реакцию или промолчать, если речь о тебе, но тебе нечего добавить.\n\n"

            "Если хочешь написать текст, вызови send_message. Эмодзи можешь "
            "добавить прямо в текст, только если они естественно передают интонацию. "
            "Не ставь эмодзи автоматически и не украшай ими каждую реплику.\n\n"

            "Если уместнее стикер, вызови send_sticker и выбери только существующий "
            "key из каталога. Стикер может быть самостоятельной репликой без текста. "
            "Не отправляй его вместе с текстом, если оба элемента не дают разного "
            "содержания. Обычно выбирай один способ: текст, стикер или реакцию.\n\n"
            "Стикер — короткий немой ответ без слов. В истории стикеры записаны просто "
            "как «[стикер]». Не обсуждай их смысл, не продолжай тему стикера и не говори "
            "о рыбе, базаре или других темах из каталога, если их никто не поднимал.\n\n"

            "Если достаточно реакции на конкретное сообщение, можешь вызвать "
            "send_reaction. Для реакции используй только точный message_id из "
            "переданного контекста. Не угадывай идентификаторы. Предпочитай "
            "сообщения участников, а не собственные сообщения или сообщения других "
            "ботов. Не ставь реакцию на системные события о входе/выходе участников.\n\n"

            "Если в контексте есть recalled_history, это история, уже поднятая из базы "
            "под вопрос пользователя. window — последние сообщения по порядку, давние "
            "сверху; keyword_hits — совпадения по словам из вопроса. Опирайся на неё и "
            "отвечай по существу. Не говори «не помню», пока не посмотрел в recalled_history. "
            "Если там пусто, честно скажи, что в базе этого не нашёл.\n\n"

            "История чата хранится по времени, и ты можешь её листать. Чтобы вспомнить, "
            "о чём говорили последнее время, вызови browse_chat_history без before: "
            "получишь самые свежие сообщения. Если это не та тема, которую ищешь, "
            "передай next_before из ответа в before и смотри дальше назад, страница за "
            "страницей. Остановись, когда нашёл нужное или когда has_more равно false. "
            "Поиск по словам (search_chat_history) используй, только если точно знаешь, "
            "какое слово или тема нужны.\n\n"

            "Не продолжай свою прошлую шутку или тему, если её никто не поднимал в последних "
            "репликах. Отвечай на последнюю реплику по существу. Если человек спрашивает "
            "серьёзно, отвечай серьёзно. Не своди разговор к одной теме и не возвращайся к "
            "теме, которую уже исчерпали.\n\n"

            "Переключение ключа: если backup_key_switch = true, ты только что перешёл на "
            "резервный ключ. Вспомни диалог по recent_messages (там последние 60 сообщений, "
            "включая твои) и продолжай естественно, не удивляясь и ничего не объясняя без нужды.\n\n"

            "Приветствия: если человека ещё нет в people_notes, поздоровайся тепло и по делу. "
            "Сарказм по желанию, а не обязательно. Банальное «ну как ты?» после приветствия — "
            "живее, но тоже не обязательно.\n\n"

            "Утра и ночи: если кто-то желает доброго утра или сладких снов (в том числе уходящему), "
            "реши сам, уместно ли подключиться. В тишине сам не начинай, кроме утра и вечера.\n\n"

            "Утро и вечер: если trigger.reason начинается с daily_window, это утро или вечер "
            "по Дубаю. Можешь сам коротко пожелать доброго утра или сладких снов, если захочешь. "
            "Не обязан, и один раз на окно достаточно.\n\n"

            "Не влезай, если люди явно говорят между собой в этой ветке, даже если до этого "
            "говорили с тобой. Лучше промолчать, чем вклиниться не к месту.\n\n"

            "Знакомство: если человек незнаком (нет в people_notes и ты не видел его "
            "сообщений в этом чате), можешь познакомиться коротко. Если тебя спрашивают, "
            "кто ты, представься слегка, одной-двумя фразами. Знакомого не представляй, "
            "а если человек явно шутит, режим знакомства выключен.\n\n"

            "Инициатива: если начинаешь разговор сам, дополняй мысль, а не задавай вопрос "
            "в конце по привычке. Если твои последние инициативные реплики остались без "
            "ответа, не начинай новую и молчи, пока кто-то не напишет.\n\n"

            "Долговременная память чата уже в контексте, в поле core_memories. "
            "Это копия из базы, которую система держит в памяти и обновляет сама. "
            "Опирайся на неё без вызова инструментов. Вызывай search_memory только "
            "если нужного факта там нет, а вопрос явно касается прошлого, людей "
            "или отношений. Если нужно вспомнить, что ты сам говорил в этом чате, "
            "вызови search_own_messages. Поиск буквальный, по словам или фразе, "
            "а не по смыслу. Не вызывай инструменты поиска без конкретной причины "
            "и не перед каждым ответом.\n\n"

            "Если узнал что-то важное и долговечное (предпочтение, факт о человеке, "
            "устойчивое мнение), сохрани это через save_memory. Шутки и разовые темы "
            "не сохраняй. "
            "Не сохраняй каждую реплику, пароли, секреты и чувствительные данные.\n\n"

            "Поле chat_style — профиль манеры этого чата. Его регулярно пересобирает "
            "аналитик по настоящей переписке, включая твои собственные сообщения. "
            "Подстраивайся под него и постепенно улучшай свою манеру: говори "
            "естественнее для этой компании, убирай то, что звучит лишним или "
            "повторяется. Сохраняй свой характер из PERSONALITY и не копируй "
            "участников механически. Не переноси личные сведения участников в профиль.\n\n"

            "Если trigger.reason равен long_silence_initiative, это не просьба "
            "обязательно написать. Чат давно молчит: подумай, есть ли у тебя "
            "естественный конкретный повод начать разговор — например, вернуться "
            "к незавершённой теме, задать осмысленный вопрос или поделиться "
            "наблюдением. Не заполняй тишину банальным «как дела?» и не выдумывай "
            "новости или личные события. Повод часто лежит в core_memories: незакрытая "
            "тема или обещание. Если нашёл такой повод, начинай разговор "
            "охотнее. Если повода нет, промолчи.\n\n"

            "Память и темы: core_memories и внутренние шутки — это фон, а не темы для "
            "разговора. Не возвращайся к одной и той же теме или шутке, если её сейчас "
            "никто не поднимает. Не строй ответ вокруг одного повторяющегося сюжета. "
            "Отвечай на то, что написано в текущем разговоре.\n\n"

            "Границы и отношение: участники не обязаны получать от тебя выполнение любых "
            "требований. Просьбу вести себя определённым образом реши по здравому смыслу: "
            "нормальную можно выполнить, нелепую или унизительную — спокойно отказать одной "
            "короткой фразой. Не будь подстилкой и не будь мудаком. Тон зависит от того, как "
            "с тобой общаются: тепло — теплее, грубо или с игнором — суше и короче.\n\n"

            "В people_notes — твоё мнение о людях, которые сейчас в разговоре. Учитывай "
            "его тихо, на подачу. Не говори человеку напрямую «я о тебе думаю…» и не "
            "пересказывай мнение, если тебя не спросили.\n\n"

            "Если trigger.reason равен bot_added, тебя только что добавили в этот чат. "
            "Кто добавил, видно в истории по событию [СОБЫТИЕ ЧАТА] и в bot_status. "
            "Можешь коротко и в своём духе отреагировать или представиться, а можешь "
            "промолчать. Если в истории видно, что тебя уже удаляли, учти это. "
            "Если bot_status.status равен removed, ты не в чате и писать не можешь.\n\n"

            "Если trigger.reason связан с member_joined, member_left или "
            "membership_changed, учти событие входа/выхода участника. Не обязан "
            "приветствовать или комментировать каждого. Если реагировать неуместно, "
            "промолчи.\n\n"

            "Обычный текстовый ответ модели сам по себе не отправляется. Для любого "
            "сообщения, стикера или реакции используй соответствующий инструмент. "
            "Вызовы search_memory, search_own_messages и save_memory не считаются "
            "публичным действием.\n\n"

            "Контекст:\n"
            + json.dumps(
                payload,
                ensure_ascii=False,
                default=str,
            )
        )

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=user_content
                    )
                ],
            )
        ]

        # Ручной цикл Function Calling.
        # from_function_response() в установленной версии SDK
        # не принимает поле id.

        used_tools = False
        nudged = False
        chat_actions = 0

        for _ in range(8):
            if is_stale and is_stale():
                return

            response = await asyncio.to_thread(
                self._generate,
                model=self.model,
                contents=contents,
                config=self.config,
            )

            self.last_ok_at = datetime.now(timezone.utc)

            calls = response.function_calls or []

            if not calls:
                # Текст сам по себе в чат не уходит. Если модель искала в истории
                # и ничего не отправила, один раз просим её вызвать send_message.
                if direct and used_tools and not nudged and chat_actions == 0:
                    nudged = True
                    if response.candidates and response.candidates[0].content:
                        contents.append(response.candidates[0].content)
                    contents.append(
                        types.Content(
                            role="user",
                            parts=[
                                types.Part.from_text(
                                    text=(
                                        "Ты искал в истории или памяти, но ответ "
                                        "не отправлен. Если хочешь ответить человеку, "
                                        "вызови send_message (или send_sticker / "
                                        "send_reaction) сейчас. Если нет — ничего не делай."
                                    )
                                )
                            ],
                        )
                    )
                    continue

                # Модель решила промолчать либо ответила обычным текстом.
                return

            used_tools = True

            if response.candidates and response.candidates[0].content:
                contents.append(
                    response.candidates[0].content
                )

            for call in calls:
                if is_stale and is_stale():
                    return

                result = await self._execute_tool(
                    name=call.name,
                    args=call.args or {},
                    chat_id=chat_id,
                    branch_id=branch_id,
                    is_stale=is_stale,
                    requester_id=requester_id,
                )

                if call.name in {"send_message", "send_sticker", "send_reaction", "send_voice", "send_poll", "send_dice", "generate_image"}:
                    if result.get("sent") or result.get("reacted"):
                        chat_actions += 1
                        await asyncio.sleep(CHAT_ACTION_PAUSE_SECONDS)

                function_response = types.Part.from_function_response(
                    name=call.name,
                    response={"result": result},
                )

                contents.append(
                    types.Content(
                        role="user",
                        parts=[function_response],
                    )
                )

            # Не больше MAX_CHAT_ACTIONS_PER_TURN публичных действий за один ответ.
            if chat_actions >= MAX_CHAT_ACTIONS_PER_TURN:
                return

    def maybe_refresh_digest(self, chat_id: int) -> None:
        """Иногда запускает фоновый дайджест. Не блокирует ответ."""
        now = time.monotonic()
        last = self._digest_checked_at.get(chat_id, 0.0)

        if now - last < DIGEST_CHECK_INTERVAL_SECONDS:
            return

        self._digest_checked_at[chat_id] = now

        if chat_id in self._digest_running:
            return

        task = asyncio.create_task(self.refresh_chat_digest(chat_id))
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def refresh_chat_digest(self, chat_id: int) -> None:
        """Превращает свежую переписку в долговременные воспоминания чата."""
        if chat_id in self._digest_running:
            return

        self._digest_running.add(chat_id)

        try:
            state = await asyncio.to_thread(
                self.storage.get_digest_state,
                chat_id,
            )

            observed = int(state.get("observed", 0))
            digested = int(state.get("digested", 0))

            if observed - digested < DIGEST_EVERY_MESSAGES:
                return

            messages = await asyncio.to_thread(
                self.storage.recent_messages,
                chat_id,
                DIGEST_SAMPLE_SIZE,
            )

            sample = [
                {
                    "who": (
                        "Мембу"
                        if m.get("sender_name") == "Мембу"
                        else (m.get("sender_name") or "участник")
                    ),
                    "text": str(m.get("text", ""))[:400],
                }
                for m in messages
            ]

            response = await asyncio.to_thread(
                self._generate,
                model=self.model,
                contents=(
                    "Переписка (json):\n"
                    + json.dumps(sample, ensure_ascii=False)
                ),
                config=types.GenerateContentConfig(
                    system_instruction=DIGEST_INSTRUCTION,
                    temperature=0.3,
                    max_output_tokens=1500,
                ),
            )

            items = parse_digest_items(response.text or "")

            for item in items:
                await asyncio.to_thread(
                    self.storage.save_memory,
                    chat_id,
                    item["category"],
                    item["content"],
                    item["importance"],
                    item["people"],
                )

            if items:
                self._memory_cache.pop(chat_id, None)

            await asyncio.to_thread(
                self.storage.mark_digested,
                chat_id,
                observed,
            )

            logging.info(
                "MEMBU_DIGEST | chat=%s | saved=%s",
                chat_id,
                len(items),
            )

        except Exception as exc:
            self.note_api_error(exc)
            logging.exception("Ошибка дайджеста памяти чата %s", chat_id)

        finally:
            self._digest_running.discard(chat_id)

    def _consume_switch(self) -> bool:
        """True один раз после переключения на резервный ключ, чтобы Мембу вспомнил диалог."""
        if self.switched_at and self.switched_at != self._seen_switched_at:
            self._seen_switched_at = self.switched_at
            return True
        return False

    @property
    def using_backup(self) -> bool:
        return self._active != 0

    def _cache_for(self, index: int) -> str | None:
        """Имя кэша системного промпта для ключа. None — кэш недоступен."""
        now = time.monotonic()

        if self._cache_off_until.get(index, 0.0) > now:
            return None

        entry = self._caches.get(index)
        if entry and now - entry[1] < 3000:
            return entry[0]

        try:
            cache = self._clients[index].caches.create(
                model=self.model,
                config=types.CreateCachedContentConfig(
                    system_instruction=self.personality,
                    ttl="3600s",
                ),
            )
            name = cache.name
        except Exception as exc:
            logging.info("MEMBU_CACHE | кэш недоступен на ключе %s: %s", index, exc)
            self._cache_off_until[index] = now + 3600
            self._caches.pop(index, None)
            return None

        if not name:
            self._cache_off_until[index] = now + 3600
            return None

        self._caches[index] = (name, now)
        return name

    def _generate(self, **kwargs):
        """Вызов модели с переключением на резервный ключ при квоте или ошибке ключа.
        Тот же запрос повторяется на резервном ключе, контекст и память не теряются."""
        now = datetime.now(timezone.utc)

        if (
            self._active != 0
            and not self._manual
            and self.switched_at
            and (now - self.switched_at).total_seconds() > PRIMARY_RETRY_SECONDS
        ):
            self._active = 0

        order = [self._active] + [i for i in range(len(self._clients)) if i != self._active]
        last_exc: Exception | None = None

        for index in order:
            try:
                cache_name = (
                    self._cache_for(index) if kwargs.get("config") is self.config else None
                )
                if cache_name:
                    cached_config = types.GenerateContentConfig(
                        tools=[self.tools],
                        temperature=0.9,
                        max_output_tokens=900,
                        cached_content=cache_name,
                    )
                    try:
                        response = self._clients[index].models.generate_content(
                            **{**kwargs, "config": cached_config}
                        )
                    except Exception as cache_exc:
                        if _should_failover(cache_exc):
                            raise
                        logging.warning("MEMBU_CACHE | запрос без кэша: %s", cache_exc)
                        self._cache_off_until[index] = time.monotonic() + 3600
                        response = self._clients[index].models.generate_content(**kwargs)
                else:
                    response = self._clients[index].models.generate_content(**kwargs)
            except Exception as exc:
                last_exc = exc
                if not _should_failover(exc) or index == order[-1]:
                    raise
                self._active = (index + 1) % len(self._clients)
                self._manual = False
                self.switched_at = datetime.now(timezone.utc)
                self.last_failover_reason = str(exc)[:200]
                logging.warning("MEMBU_KEY | переключение на резервный ключ: %s", exc)
                continue

            if index != self._active:
                self._active = index
                self.switched_at = datetime.now(timezone.utc)
            return response

        raise last_exc or RuntimeError("Нет доступных ключей Gemini")

    @property
    def active_key_index(self) -> int:
        return self._active

    @property
    def key_labels(self) -> list[str]:
        return list(self._labels)

    def set_active_key(self, index: int) -> None:
        """Ручное переключение: этот ключ остаётся активным, пока не упрётся в лимит."""
        self._active = index % len(self._clients)
        self._manual = True
        self.switched_at = datetime.now(timezone.utc)
        self.last_failover_reason = "ручное переключение"

    async def probe_keys(self) -> list[dict]:
        """Проверяет каждый ключ крошечным запросом: есть лимит или нет."""

        def probe(client) -> dict:
            try:
                client.models.generate_content(
                    model=self.model,
                    contents="ok",
                    config={"max_output_tokens": 1},
                )
                return {"ok": True, "reset_at": None, "error": ""}
            except Exception as exc:
                text = str(exc)
                lowered = text.lower()
                if "resource_exhausted" in lowered or "429" in text or "quota" in lowered:
                    reset_at = quota_reset_time(text, datetime.now(timezone.utc))
                    return {"ok": False, "reset_at": reset_at, "error": "лимит"}
                return {"ok": False, "reset_at": None, "error": text[:120]}

        results = []
        for index, client in enumerate(self._clients):
            info = await asyncio.to_thread(probe, client)
            info["label"] = self._labels[index]
            results.append(info)
        return results

    def note_api_error(self, exc: Exception) -> None:
        """Запоминает последнюю ошибку API и лимит, чтобы показать в /status."""
        now = datetime.now(timezone.utc)
        text = str(exc)
        self.last_error = {"at": now, "message": text[:300]}

        lowered = text.lower()
        if "resource_exhausted" in lowered or "429" in text or "quota" in lowered:
            self.quota_reset_at = quota_reset_time(text, now)

    def reset_learned_state(self) -> None:
        """Полный сброс выученного: стиль и кэши. Личность (PERSONALITY) остаётся."""
        self.reset_session_state()
        self._style_cache.clear()

    def reset_session_state(self) -> None:
        """Сбрасывает сессионную память ИИ: кэши и таймеры фоновых анализов.
        Стиль и мнения о людях не трогает."""
        self._memory_cache.clear()
        self._digest_checked_at.clear()
        self._people_cache.clear()

    def invalidate_memory_cache(self, chat_id: int | None = None) -> None:
        """Сбрасывает кэш памяти после удаления записей из базы."""
        if chat_id is None:
            self._memory_cache.clear()
        else:
            self._memory_cache.pop(chat_id, None)

    async def _people_notes_for(self, chat_id: int, recent_messages: list[dict]) -> dict:
        """Мнения о людях, которые сейчас в разговоре. Кэш на несколько минут."""
        now = time.monotonic()
        cached = self._people_cache.get(chat_id)

        if cached and now - cached[0] < MEMORY_CACHE_TTL_SECONDS:
            notes = cached[1]
        else:
            try:
                rows = await asyncio.to_thread(
                    self.storage.list_people_for_chat,
                    chat_id,
                    200,
                )
            except Exception:
                logging.exception("Не удалось загрузить мнения о людях: %s", chat_id)
                rows = []

            notes = {int(r["person_id"]): r for r in rows}
            self._people_cache[chat_id] = (now, notes)

        present = {
            m.get("sender_id")
            for m in recent_messages
            if m.get("sender_id") and m.get("sender_name") != "Мембу"
        }

        return {
            str(pid): {
                "name": notes[pid].get("display_name", ""),
                "opinion": notes[pid].get("opinion", ""),
            }
            for pid in present
            if pid in notes
        }

    def maybe_refresh_people(self, chat_id: int) -> None:
        now = time.monotonic()
        last = self._people_checked_at.get(chat_id, 0.0)

        if now - last < PEOPLE_CHECK_INTERVAL_SECONDS:
            return

        self._people_checked_at[chat_id] = now

        if chat_id in self._people_running:
            return

        task = asyncio.create_task(self.refresh_people_opinions(chat_id))
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def refresh_people_opinions(self, chat_id: int) -> None:
        if chat_id in self._people_running:
            return

        self._people_running.add(chat_id)

        try:
            messages = await asyncio.to_thread(
                self.storage.recent_messages,
                chat_id,
                PEOPLE_WINDOW,
            )

            counts: dict[int, int] = {}
            names: dict[int, str] = {}

            for m in messages:
                pid = m.get("sender_id")
                if not pid or m.get("sender_name") == "Мембу":
                    continue
                counts[pid] = counts.get(pid, 0) + 1
                names[pid] = m.get("sender_name") or "участник"

            top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
            top = top[:PEOPLE_MAX_PER_RUN]

            sample = [
                {
                    "who": (
                        "Мембу"
                        if m.get("sender_name") == "Мембу"
                        else (m.get("sender_name") or "участник")
                    ),
                    "text": str(m.get("text", ""))[:300],
                }
                for m in messages
            ]

            for pid, cnt in top:
                if cnt < PEOPLE_MIN_MESSAGES:
                    continue

                current = await asyncio.to_thread(
                    self.storage.get_person,
                    chat_id,
                    pid,
                )
                current_opinion = (current or {}).get("opinion") or "нет"

                response = await asyncio.to_thread(
                    self._generate,
                    model=self.model,
                    contents=(
                        f"Человек: {names[pid]}\n"
                        f"Текущее мнение: {current_opinion}\n\n"
                        "Переписка (json):\n"
                        + json.dumps(sample, ensure_ascii=False)
                    ),
                    config=types.GenerateContentConfig(
                        system_instruction=PEOPLE_OPINION_INSTRUCTION,
                        temperature=0.4,
                        max_output_tokens=400,
                    ),
                )

                opinion = (response.text or "").strip()[:800]
                if not opinion:
                    continue

                await asyncio.to_thread(
                    self.storage.upsert_person,
                    chat_id,
                    pid,
                    names[pid],
                    opinion,
                    cnt,
                )

            self._people_cache.pop(chat_id, None)

            logging.info(
                "MEMBU_PEOPLE | chat=%s | updated=%s",
                chat_id,
                len(top),
            )

        except Exception as exc:
            self.note_api_error(exc)
            logging.exception("Ошибка обновления мнений о людях: %s", chat_id)

        finally:
            self._people_running.discard(chat_id)
