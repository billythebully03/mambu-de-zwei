import asyncio
import json

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


class GeminiAgent:
    def __init__(
        self,
        api_key: str,
        storage,
        bot,
        bot_user,
        model: str = "gemini-3.5-flash-lite",
    ):
        self.client = genai.Client(
            api_key=api_key
        )

        self.storage = storage
        self.bot = bot
        self.bot_user = bot_user
        self.model = model

        from pathlib import Path

        personality_path = (
            Path(__file__).with_name("PERSONALITY.md")
        )

        self.personality = personality_path.read_text(
            encoding="utf-8"
        )

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
                    "Ищет воспоминания Мембу о людях, "
                    "событиях, отношениях, предпочтениях "
                    "и прошлых разговорах. Используй только "
                    "если старый контекст действительно нужен. "
                    "Не вызывай автоматически перед каждым ответом."
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Что именно нужно вспомнить."
                            ),
                        },
                        "category": {
                            "type": "string",
                            "enum": MEMORY_CATEGORIES,
                            "description": (
                                "Категория воспоминания. "
                                "Можно не указывать."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "description": (
                                "Количество результатов от 1 до 8."
                            ),
                        },
                    },
                    "required": ["query"],
                },
            ),

            types.FunctionDeclaration(
                name="save_memory",
                description=(
                    "Сохраняет долговременное воспоминание. "
                    "Используй только для действительно важных "
                    "или полезных в будущем сведений. "
                    "Не сохраняй каждую реплику."
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
                            "description": (
                                "Краткое и понятное воспоминание."
                            ),
                        },
                        "importance": {
                            "type": "integer",
                            "description": (
                                "Важность от 1 до 5."
                            ),
                        },
                        "people": {
                            "type": "array",
                            "items": {
                                "type": "string",
                            },
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
                    "Отправляет сообщение от имени Мембу "
                    "в Telegram. Вызывай только если сам "
                    "решил вступить в разговор. Если лучше "
                    "промолчать, не вызывай этот инструмент."
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": (
                                "Текст сообщения Мембу."
                            ),
                        },
                        "reply_to_message_id": {
                            "type": "integer",
                            "description": (
                                "Необязательный ID сообщения, "
                                "на которое нужно ответить."
                            ),
                        },
                    },
                    "required": ["text"],
                },
            ),
        ]

        return types.Tool(
            function_declarations=declarations
        )

    async def _execute_tool(
        self,
        name: str,
        args: dict,
        chat_id: int,
        branch_id: str | None,
    ):

        args = args or {}

        if name == "search_memory":

            query = str(
                args.get("query", "")
            ).strip()

            if not query:
                return {
                    "error": "Пустой поисковый запрос."
                }

            category = args.get("category")

            if category not in MEMORY_CATEGORIES:
                category = None

            limit = int(
                args.get("limit", 5)
            )

            memories = await asyncio.to_thread(
                self.storage.search_memories,
                chat_id,
                query,
                category,
                max(1, min(limit, 8)),
            )

            return {
                "memories": memories,
            }

        if name == "save_memory":

            category = args.get("category")
            content = str(
                args.get("content", "")
            ).strip()

            if category not in MEMORY_CATEGORIES:
                return {
                    "error": "Неизвестная категория."
                }

            if not content:
                return {
                    "error": "Пустое воспоминание."
                }

            importance = int(
                args.get("importance", 3)
            )

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

            return {
                "saved": True,
            }

        if name == "send_message":

            text = str(
                args.get("text", "")
            ).strip()

            if not text:
                return {
                    "error": "Нельзя отправить пустое сообщение."
                }

            text = text[:4000]

            reply_to = args.get(
                "reply_to_message_id"
            )

            try:
                reply_to = (
                    int(reply_to)
                    if reply_to is not None
                    else None
                )
            except (TypeError, ValueError):
                reply_to = None

            sent = await self.bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_to_message_id=reply_to,
            )

            # Сохраняем и собственные сообщения Мембу.
            await asyncio.to_thread(
                self.storage.add_message,
                {
                    "chat_id": chat_id,
                    "message_id": sent.message_id,
                    "sender_id": self.bot_user.id,
                    "sender_name": "Мембу",
                    "username": (
                        self.bot_user.username or ""
                    ),
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

            return {
                "sent": True,
                "message_id": sent.message_id,
            }

        return {
            "error": "Неизвестный инструмент."
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
    ):

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
            "Проанализируй текущую ситуацию в групповом чате. "
            "Определи, обращаются ли к тебе, продолжается ли "
            "разговор с твоим участием и есть ли естественный "
            "повод вступить. Ты можешь промолчать. "
            "Если решишь написать, обязательно вызови "
            "send_message. Обычный текстовый ответ не отправляется.\n\n"
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
        # Модель сама выбирает инструменты и порядок их вызова.
        for _ in range(6):

            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model,
                contents=contents,
                config=self.config,
            )

            calls = response.function_calls or []

            # Если модель решила просто промолчать,
            # никаких сообщений в Telegram не отправляем.
            if not calls:
                return

            if (
                response.candidates
                and response.candidates[0].content
            ):
                contents.append(
                    response.candidates[0].content
                )

            sent_message = False

            for call in calls:

                result = await self._execute_tool(
                    name=call.name,
                    args=call.args or {},
                    chat_id=chat_id,
                    branch_id=branch_id,
                )

                if (
                    call.name == "send_message"
                    and result.get("sent")
                ):
                    sent_message = True

                response_kwargs = {
                    "name": call.name,
                    "response": {
                        "result": result,
                    },
                }

                if getattr(call, "id", None):
                    response_kwargs["id"] = call.id

                function_response = (
                    types.Part.from_function_response(
                        **response_kwargs
                    )
                )

                contents.append(
                    types.Content(
                        role="user",
                        parts=[function_response],
                    )
                )

            # После реальной отправки не нужно
            # генерировать ещё одну реплику в этом цикле.
            if sent_message:
                return
