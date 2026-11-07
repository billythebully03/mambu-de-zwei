import asyncio
import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

from aiogram.exceptions import TelegramAPIError
from aiogram.types import ReactionTypeEmoji
from google import genai
from google.genai import types


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

MIN_MESSAGES_FOR_STYLE_UPDATE = 10

# Аналитик стиля смотрит на переписку не чаще, чем раз в эти секунды.
STYLE_CHECK_INTERVAL_SECONDS = 60
# Сколько последних сообщений (людей и Мембу) отдаём аналитику стиля.
STYLE_SAMPLE_SIZE = 100
MAX_CHAT_ACTIONS_PER_TURN = 2

# Кэш памяти из Supabase: как долго держим его без перезагрузки.
MEMORY_CACHE_TTL_SECONDS = 300
MEMORY_LOAD_LIMIT = 60
# Сколько самых важных воспоминаний кладём в каждый контекст.
MEMORY_IN_CONTEXT = 40
STYLE_CACHE_TTL_SECONDS = 300

STYLE_ANALYST_INSTRUCTION = (
    "Ты ведёшь профиль манеры общения Мембу в этом чате. Твоя задача — описать, как "
    "в этом чате принято писать, чтобы Мембу постепенно говорил так же, как участники. "
    "Смотри на манеру, а не на содержание: длина реплик, регистр, сленг, обращения на "
    "«ты» или «вы», эмодзи, скобочки, растяжки слов, как люди смеются, насколько "
    "формально, темп, как реагируют на подколы и грубость. Не переноси в профиль темы, "
    "факты, шутки, имена и личные сведения участников. Характер Мембу не меняется, его "
    "ядро описано в PERSONALITY. Подстройка идёт плавно: если дан текущий профиль, "
    "сохрани большую его часть и измени не больше двух пунктов, а именно те, что явно "
    "отличаются от реальной переписки. Структура: сначала короткое описание норм чата, "
    "затем раздел «Подстройка Мембу» до 5 пунктов, каждый — умеренный шаг от текущей "
    "манеры Мембу к нормам чата. Опирайся только на переписку, не выдумывай "
    "закономерностей. Простой текст, до 2000 символов, без markdown."
)

# Дайджест: раз в столько новых сообщений модель выделяет из переписки
# долговременные факты и записывает их в память чата.
DIGEST_EVERY_MESSAGES = 80
DIGEST_SAMPLE_SIZE = 120
DIGEST_CHECK_INTERVAL_SECONDS = 60

DIGEST_INSTRUCTION = (
    "Ты помогаешь Telegram-боту Мембу помнить чат. Из переписки выдели 2–5 "
    "долговечных фактов, которые стоит запомнить: о чём договорились, какие темы "
    "обсуждали и чем закончили, предпочтения людей, устойчивые мнения, незавершённые "
    "дела. Шутки и повторяющиеся темы не сохраняй: это не факты, и память заставит "
    "возвращаться к ним. Категорию inside_joke не используй. Важность темы не выше 3. "
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
PEOPLE_CHECK_INTERVAL_SECONDS = 1800
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
        self.client = genai.Client(api_key=api_key)
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
                    "только если сам решил написать. Если лучше промолчать, "
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

        return {"error": "Неизвестный инструмент."}

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
                self.client.models.generate_content,
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

        # Профиль стиля пересобирается в фоне, сам ответ не ждёт его.
        self.maybe_refresh_style(chat_id)
        self.maybe_refresh_digest(chat_id)
        self.maybe_refresh_people(chat_id)

        recalled_history = await self._recall_context(chat_id, recent_messages)
        people_notes = await self._people_notes_for(chat_id, recent_messages)

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
                for m in trim_own_messages(recent_messages, keep=3)
            ],
            "current_branch_messages": branch_messages,
            "active_branches": active_branches,
            "chat_style": chat_style,
        }

        user_content = (
            "Ты получил новую ситуацию из Telegram-чата. Сначала пойми, "
            "что происходит и кто с кем разговаривает. Ты сам выбираешь, "
            "нужно ли участвовать. Можешь промолчать. Не отвечай автоматически "
            "на каждое сообщение.\n\n"

            "За один ответ можно отправить не больше двух сообщений. Вторую реплику "
            "ставь только если она что-то добавляет. Её можно отправить как ответ на любое "
            "сообщение (в том числе из истории или recalled_history, по message_id), как "
            "ответ на собственное сообщение Мембу или отдельным сообщением. Чаще хватает одного.\n\n"

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
                self.client.models.generate_content,
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
                )

                if call.name in {"send_message", "send_sticker", "send_reaction"}:
                    if result.get("sent") or result.get("reacted"):
                        chat_actions += 1

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

            # Больше двух публичных действий за один ответ не отправляем.
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
                self.client.models.generate_content,
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
                    self.client.models.generate_content,
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
