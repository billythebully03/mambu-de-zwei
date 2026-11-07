import asyncio
import json
import logging
from pathlib import Path
from typing import Callable

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

MIN_MESSAGES_FOR_STYLE_UPDATE = 15


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
                            "description": "Важность от 1 до 5.",
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
                name="update_chat_style",
                description=(
                    "Обновляет компактный профиль общей манеры общения этого чата. "
                    "Вызывай только когда в профиле указано, что накопилось не меньше "
                    f"{MIN_MESSAGES_FOR_STYLE_UPDATE} новых сообщений после прошлого анализа, "
                    "и в контексте действительно видны устойчивые повторяющиеся особенности. "
                    "Передавай полный обновлённый профиль, а не отдельную заметку. "
                    "Не сохраняй личные секреты, чувствительные сведения, индивидуальные "
                    "характеристики участников или случайную разовую шутку. Если новых "
                    "устойчивых наблюдений нет — не вызывай инструмент."
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "style_profile": {
                            "type": "string",
                            "description": (
                                "Краткое описание коллективного стиля чата: темп, "
                                "длина сообщений, типичный юмор, сленг, эмодзи и "
                                "общая степень неформальности. До 2500 символов."
                            ),
                        },
                    },
                    "required": ["style_profile"],
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

            return {"saved": True}

        if name == "update_chat_style":
            style_profile = str(
                args.get("style_profile", "")
            ).strip()

            if not style_profile:
                return {"error": "Профиль манеры общения пуст."}

            result = await asyncio.to_thread(
                self.storage.update_chat_style,
                chat_id,
                style_profile,
                MIN_MESSAGES_FOR_STYLE_UPDATE,
            )

            return result

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

            log_text = (
                f"[СТИКЕР: {key}] "
                f"{sticker['description']} "
                f"Значение: {sticker['meaning']}"
            )

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
    ) -> None:
        chat_style = await asyncio.to_thread(
            self.storage.get_chat_style,
            chat_id,
        )

        payload = {
            "chat_id": chat_id,
            "trigger": {
                "reason": reason,
                "direct": direct,
                "branch_id": branch_id,
            },
            "recent_messages": recent_messages,
            "current_branch_messages": branch_messages,
            "active_branches": active_branches,
            "chat_style": chat_style,
        }

        user_content = (
            "Ты получил новую ситуацию из Telegram-чата. Сначала пойми, "
            "что происходит и кто с кем разговаривает. Ты сам выбираешь, "
            "нужно ли участвовать. Можешь промолчать. Не отвечай автоматически "
            "на каждое сообщение.\n\n"

            "Если хочешь написать текст, вызови send_message. Эмодзи можешь "
            "добавить прямо в текст, только если они естественно передают интонацию. "
            "Не ставь эмодзи автоматически и не украшай ими каждую реплику.\n\n"

            "Если уместнее стикер, вызови send_sticker и выбери только существующий "
            "key из каталога. Стикер может быть самостоятельной репликой без текста. "
            "Не отправляй его вместе с текстом, если оба элемента не дают разного "
            "содержания. Обычно выбирай один способ: текст, стикер или реакцию.\n\n"

            "Если достаточно реакции на конкретное сообщение, можешь вызвать "
            "send_reaction. Для реакции используй только точный message_id из "
            "переданного контекста. Не угадывай идентификаторы. Предпочитай "
            "сообщения участников, а не собственные сообщения или сообщения других "
            "ботов. Не ставь реакцию на системные события о входе/выходе участников.\n\n"

            "Профиль chat_style — постепенно собранное описание общей манеры "
            "общения компании. Используй его как дополнительный ориентир, а не как "
            "новую личность. Сохраняй собственный характер Мембу. Подхватывай общий "
            "ритм, степень неформальности, естественные выражения и тип юмора, но "
            "не копируй механически ошибки, привычки или грубость одного человека. "
            "Не переноси личные сведения участников в профиль стиля.\n\n"

            "Если в chat_style.new_messages_since_analysis накопилось не меньше "
            f"{MIN_MESSAGES_FOR_STYLE_UPDATE}, оцени, появились ли устойчивые "
            "повторяющиеся особенности общения. Если да — можешь вызвать "
            "update_chat_style с полным кратким обновлённым профилем. Не делай "
            "этого ради галочки: разовая фраза или случайный мем не являются "
            "манерой всего чата. Если новых закономерностей нет, не обновляй.\n\n"

            "Если trigger.reason равен long_silence_initiative, это не просьба "
            "обязательно написать. Чат давно молчит: подумай, есть ли у тебя "
            "естественный конкретный повод начать разговор — например, вернуться "
            "к незавершённой теме, задать осмысленный вопрос или поделиться "
            "наблюдением. Не заполняй тишину банальным «как дела?» и не выдумывай "
            "новости или личные события. Если повода нет, промолчи.\n\n"

            "Если trigger.reason связан с member_joined, member_left или "
            "membership_changed, учти событие входа/выхода участника. Не обязан "
            "приветствовать или комментировать каждого. Если реагировать неуместно, "
            "промолчи.\n\n"

            "Обычный текстовый ответ модели сам по себе не отправляется. Для любого "
            "сообщения, стикера или реакции используй соответствующий инструмент. "
            "Вызов update_chat_style не считается публичным сообщением.\n\n"

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

        for _ in range(8):
            if is_stale and is_stale():
                return

            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model,
                contents=contents,
                config=self.config,
            )

            calls = response.function_calls or []

            if not calls:
                # Модель решила промолчать либо ответила обычным текстом.
                # Текст сам по себе никогда не отправляется в чат.
                return

            if response.candidates and response.candidates[0].content:
                contents.append(
                    response.candidates[0].content
                )

            performed_chat_action = False

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

                if call.name == "send_message" and result.get("sent"):
                    performed_chat_action = True

                if call.name == "send_sticker" and result.get("sent"):
                    performed_chat_action = True

                if call.name == "send_reaction" and result.get("reacted"):
                    performed_chat_action = True

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

            # После публичного действия не просим модель тут же генерировать
            # ещё одну реплику. Она уже поучаствовала в разговоре.
            if performed_chat_action:
                return
