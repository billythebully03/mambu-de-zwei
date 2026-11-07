import asyncio
import json
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

# Эмодзи, которые Telegram Bot API принимает как стандартные реакции.
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

        personality_path = Path(__file__).with_name("PERSONALITY.md")
        self.personality = personality_path.read_text(encoding="utf-8")

        self.tools = self._build_tools()
        self.config = types.GenerateContentConfig(
            system_instruction=self.personality,
            tools=[self.tools],
            temperature=0.9,
            max_output_tokens=800,
        )

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
                    "required": ["category", "content", "importance"],
                },
            ),
            types.FunctionDeclaration(
                name="send_message",
                description=(
                    "Отправляет сообщение от имени Мембу в Telegram. Вызывай, "
                    "только если сам решил написать. Если лучше промолчать, "
                    "не вызывай этот инструмент. Можно указать сообщение, на которое "
                    "нужно ответить. В обычной инициативе без конкретного адресата "
                    "отправляй сообщение без reply."
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Текст сообщения Мембу.",
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
                    "реакции достаточно, когда участник прямо просит о реакции или "
                    "когда реакция естественно дополняет разговор. Не реагируй на всё "
                    "подряд. message_id выбирай только из переданного контекста; "
                    "не выдумывай идентификаторы. Предпочитай сообщения участников, "
                    "а не собственные сообщения или сообщения других ботов, если "
                    "тебя специально не попросили об обратном. Не пытайся реагировать "
                    "на сервисные сообщения о входе/выходе участников: Telegram может "
                    "запрещать это. Допустимые значения emoji: "
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
        ]

        return types.Tool(function_declarations=declarations)

    def _mark_chat_activity(self, chat_id: int) -> None:
        if self.activity_callback is None:
            return
        try:
            self.activity_callback(chat_id)
        except Exception:
            # Отправленное сообщение уже не должно считаться неуспешным
            # только из-за сбоя локального таймера активности.
            pass

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

        if name == "send_message":
            text = str(args.get("text", "")).strip()
            if not text:
                return {"error": "Нельзя отправить пустое сообщение."}

            if is_stale and is_stale():
                return {"cancelled": True, "reason": "В чате уже появилось новое сообщение."}

            text = text[:4000]
            reply_to = args.get("reply_to_message_id")

            try:
                reply_to = int(reply_to) if reply_to is not None else None
            except (TypeError, ValueError):
                reply_to = None

            sent = await self.bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_to_message_id=reply_to,
            )

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
            return {"sent": True, "message_id": sent.message_id}

        if name == "send_reaction":
            try:
                message_id = int(args.get("message_id"))
            except (TypeError, ValueError):
                return {"error": "Некорректный message_id."}

            emoji = str(args.get("emoji", "")).strip()
            if emoji not in TELEGRAM_REACTIONS:
                return {"error": "Эта emoji-реакция не поддерживается Telegram Bot API."}

            if is_stale and is_stale():
                return {"cancelled": True, "reason": "В чате уже появилось новое сообщение."}

            try:
                await self.bot.set_message_reaction(
                    chat_id=chat_id,
                    message_id=message_id,
                    reaction=[ReactionTypeEmoji(emoji=emoji)],
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
        }

        user_content = (
            "Ты получил новую ситуацию из Telegram-чата. Сначала пойми, "
            "что происходит и кто с кем разговаривает. Ты сам выбираешь, "
            "нужно ли участвовать. Можешь промолчать. Не отвечай автоматически "
            "на каждое сообщение.\n\n"
            "Если хочешь написать текст, вызови send_message. Если достаточно "
            "реакции или участник прямо просит её поставить, можешь вызвать "
            "send_reaction. Обычно выбирай один способ взаимодействия; "
            "используй оба только если каждый действительно добавляет что-то.\n\n"
            "В recent_messages у сообщений есть точные message_id. Для реакции "
            "используй только ID сообщения из контекста. Не угадывай ID. "
            "Предпочитай сообщения участников, а не собственные сообщения "
            "или сообщения других ботов. Не ставь реакцию на системные события "
            "о входе/выходе участников.\n\n"
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
            "Обычный текстовый ответ модели сам по себе не отправляется. "
            "Для любого сообщения или реакции используй соответствующий инструмент.\n\n"
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
                parts=[types.Part.from_text(text=user_content)],
            )
        ]

        # Ручной цикл Function Calling. Ответ инструмента отправляется назад
        # в Gemini без поля id: google-genai Part.from_function_response()
        # в установленной версии SDK не принимает такой аргумент.
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
                # В чат ничего не отправляем автоматически.
                return

            if response.candidates and response.candidates[0].content:
                contents.append(response.candidates[0].content)

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
                if call.name == "send_reaction" and result.get("reacted"):
                    performed_chat_action = True

                # Важно: не передаём call.id. Установленный SDK его не принимает.
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
            # ещё одну реплику. Она уже поговорила или отреагировала.
            if performed_chat_action:
                return
