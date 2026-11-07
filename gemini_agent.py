import asyncio
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from aiogram.exceptions import TelegramAPIError
from aiogram.types import ReactionTypeEmoji
from google import genai
from google.genai import types


MEMORY_CATEGORIES = [
    "profile", "preference", "relationship", "ongoing",
    "event", "inside_joke", "opinion",
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

EDIT_WINDOW_SECONDS = 10 * 60
EDIT_COOLDOWN_SECONDS = 5 * 60
EDIT_MAX_PER_HOUR = 3
EDIT_MAX_TEXT_LENGTH = 4000

QUOTA_WATCH_INTERVAL_SECONDS = 2
DEFAULT_PRIMARY_COOLDOWN_SECONDS = 300


class GeminiAgent:
    def __init__(
        self,
        api_key: str,
        storage,
        bot,
        bot_user,
        model: str = "gemini-3.5-flash-lite",
        activity_callback: Callable[[int], None] | None = None,
        backup_api_key: str | None = None,
    ):
        self.primary_client = genai.Client(api_key=api_key)

        backup_api_key = (backup_api_key or "").strip()

        self.backup_client = (
            genai.Client(api_key=backup_api_key)
            if backup_api_key and backup_api_key != api_key
            else None
        )

        self._active_key_name = "primary"
        self._primary_retry_at = 0.0
        self._quota_monitor_task: asyncio.Task | None = None

        self.storage = storage
        self.bot = bot
        self.bot_user = bot_user
        self.model = model
        self.activity_callback = activity_callback

        self._edit_locks: dict[int, asyncio.Lock] = {}
        self._edited_message_ids: set[tuple[int, int]] = set()

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

        if self.backup_client:
            logging.info(
                "Gemini резервный ключ настроен; переключение будет выполняться при 429/RESOURCE_EXHAUSTED"
            )
        else:
            logging.warning(
                "GEMINI_BACKUP_API_KEY не задан; резервное переключение Gemini отключено"
            )

    # =========================================
    # РЕЗЕРВНЫЙ GEMINI API KEY
    # =========================================

    def start_quota_monitor(self) -> None:
        """Запускает проверку таймера возврата к основному ключу."""

        if not self.backup_client:
            return

        if self._quota_monitor_task is None or self._quota_monitor_task.done():
            self._quota_monitor_task = asyncio.create_task(
                self._quota_watchdog()
            )

    async def _quota_watchdog(self) -> None:
        """
        Каждые две секунды проверяет, закончилась ли пауза основного ключа.

        Не отправляет тестовые запросы к Gemini и не расходует квоту.
        Реальное исчерпание лимита определяется по ответу 429 API.
        """

        while True:
            await asyncio.sleep(
                QUOTA_WATCH_INTERVAL_SECONDS
            )

            if (
                self.backup_client
                and self._active_key_name == "backup"
                and self._primary_retry_at > 0
                and time.monotonic() >= self._primary_retry_at
            ):
                self._active_key_name = "primary"
                self._primary_retry_at = 0.0

                logging.info(
                    "Gemini: пауза основного ключа завершилась; следующий запрос проверит его доступность"
                )

    @staticmethod
    def _is_quota_error(exc: Exception) -> bool:
        """Распознаёт именно rate-limit/quota ответы Gemini."""

        code = getattr(exc, "code", None)
        status = str(
            getattr(exc, "status", "") or ""
        ).upper()

        try:
            code = int(code)
        except (TypeError, ValueError):
            code = None

        return code == 429 or status in {
            "RESOURCE_EXHAUSTED",
            "RATE_LIMIT_EXCEEDED",
        }

    @staticmethod
    def _quota_cooldown_seconds(exc: Exception) -> float:
        """Берёт retryDelay от сервера, если он есть."""

        details = getattr(exc, "details", None)
        message = str(
            getattr(exc, "message", None) or exc
        )

        if isinstance(details, dict):
            error_data = details.get("error", details)

            if isinstance(error_data, dict):
                for item in error_data.get("details", []) or []:
                    if not isinstance(item, dict):
                        continue

                    item_type = str(item.get("@type", ""))

                    if item_type.endswith("RetryInfo"):
                        retry_delay = str(
                            item.get("retryDelay", "")
                        )

                        match = re.fullmatch(
                            r"\s*([0-9]+(?:\.[0-9]+)?)s\s*",
                            retry_delay,
                        )

                        if match:
                            return max(
                                2.0,
                                min(
                                    float(match.group(1)),
                                    24 * 60 * 60,
                                ),
                            )

                # Суточная квота обычно не восстановится через минуту.
                for item in error_data.get("details", []) or []:
                    if not isinstance(item, dict):
                        continue

                    if not str(item.get("@type", "")).endswith("QuotaFailure"):
                        continue

                    for violation in item.get("violations", []) or []:
                        quota_id = str(
                            violation.get("quotaId", "")
                        ).lower()

                        if "perday" in quota_id or "per_day" in quota_id:
                            return 6 * 60 * 60

        match = re.search(
            r"retry in ([0-9]+(?:\.[0-9]+)?)s",
            message,
            re.IGNORECASE,
        )

        if match:
            return max(
                2.0,
                min(
                    float(match.group(1)),
                    24 * 60 * 60,
                ),
            )

        return DEFAULT_PRIMARY_COOLDOWN_SECONDS

    async def _generate_content_with_failover(self, **kwargs):
        """
        Делает запрос активным ключом.

        Если основной ключ возвращает 429/RESOURCE_EXHAUSTED,
        переключается на резервный и повторяет тот же запрос один раз.
        """

        if self._active_key_name == "backup" and self.backup_client:
            selected_client = self.backup_client
            selected_name = "backup"
        else:
            selected_client = self.primary_client
            selected_name = "primary"

        try:
            return await asyncio.to_thread(
                selected_client.models.generate_content,
                **kwargs,
            )

        except Exception as exc:
            if (
                selected_name != "primary"
                or self.backup_client is None
                or not self._is_quota_error(exc)
            ):
                raise

            cooldown = self._quota_cooldown_seconds(exc)

            self._active_key_name = "backup"

            self._primary_retry_at = max(
                self._primary_retry_at,
                time.monotonic() + cooldown,
            )

            logging.warning(
                "Gemini: основной ключ получил квоту/429; переключаюсь на резервный на %.0f сек. Ошибка: %s",
                cooldown,
                str(exc)[:350],
            )

            try:
                return await asyncio.to_thread(
                    self.backup_client.models.generate_content,
                    **kwargs,
                )

            except Exception:
                logging.exception(
                    "Gemini: резервный ключ тоже не смог выполнить запрос. Проверь, что он относится к другому проекту и имеет доступ к модели."
                )
                raise

    async def close(self) -> None:
        """Останавливает watchdog и закрывает клиенты SDK."""

        if self._quota_monitor_task:
            self._quota_monitor_task.cancel()

            await asyncio.gather(
                self._quota_monitor_task,
                return_exceptions=True,
            )

            self._quota_monitor_task = None

        clients = [self.primary_client]

        if self.backup_client:
            clients.append(self.backup_client)

        for client in clients:
            close_method = getattr(client, "close", None)

            if close_method:
                try:
                    await asyncio.to_thread(close_method)
                except Exception:
                    logging.debug(
                        "Не удалось корректно закрыть Gemini client",
                        exc_info=True,
                    )

    # =========================================
    # СТИКЕРЫ
    # =========================================

    @staticmethod
    def _load_stickers(path: Path) -> dict[str, dict]:
        try:
            raw = json.loads(
                path.read_text(encoding="utf-8")
            )
        except FileNotFoundError:
            logging.warning(
                "Файл stickers.json не найден; отправка стикеров отключена"
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
            if not isinstance(item, dict) or item.get("enabled", True) is False:
                continue

            key = str(item.get("key", "")).strip()
            file_id = str(item.get("file_id", "")).strip()

            if not key or not file_id or key in catalog:
                continue

            catalog[key] = {
                "key": key,
                "file_id": file_id,
                "description": str(item.get("description", "")).strip(),
                "meaning": str(item.get("meaning", "")).strip(),
                "usage": str(item.get("usage", "")).strip(),
                "format": str(item.get("format", "static")).strip(),
                "emoji": str(item.get("emoji", "")).strip(),
                "set_name": str(item.get("set_name", "")).strip(),
                "file_unique_id": str(item.get("file_unique_id", "")).strip(),
            }

        logging.info(
            "Загружено стикеров из stickers.json: %s",
            len(catalog),
        )

        return catalog

    def _sticker_catalog_for_model(self) -> str:
        if not self.stickers:
            return "Сейчас доступных стикеров нет."

        return "\n".join(
            f"{key}: {item['description']} Значение: {item['meaning']} "
            f"Когда уместен: {item['usage']}"
            for key, item in self.stickers.items()
        )

    # =========================================
    # ИНСТРУМЕНТЫ GEMINI
    # =========================================

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
                    "действительно важных или полезных в будущем сведений. Не сохраняй "
                    "каждую реплику, пароли, секреты или чувствительные данные."
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
                            "description": "Краткое воспоминание.",
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
                    "Отправляет обычное сообщение от имени Мембу в текущий чат. "
                    "Вызывай, только если сам решил написать. Если лучше промолчать, "
                    "не вызывай этот инструмент. В личном чате это ответ человеку "
                    "в его личной переписке."
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Текст сообщения; эмодзи необязательны.",
                        },
                        "reply_to_message_id": {
                            "type": "integer",
                        },
                    },
                    "required": ["text"],
                },
            ),

            types.FunctionDeclaration(
                name="send_private_message",
                description=(
                    "Отправляет личное сообщение только тому участнику, чья явная просьба "
                    "написать в личку вызвала текущую обработку. Получатель определяется "
                    "самой системой и не выбирается тобой. Используй только если в trigger "
                    "явно указано private_message_allowed=true. Никаких инициативных личных "
                    "сообщений, рассылок или выбора другого адресата. Если отправка не удалась, "
                    "объясни причину в исходном групповом чате через send_message; можно "
                    "сделать это в своей манере, но не скрывай реальную причину."
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Текст личного сообщения.",
                        },
                    },
                    "required": ["text"],
                },
            ),

            types.FunctionDeclaration(
                name="send_reaction",
                description=(
                    "Ставит одну стандартную emoji-реакцию на существующее сообщение. "
                    "Используй, когда реакции достаточно. message_id выбирай только из "
                    "переданного контекста; не выдумывай идентификаторы. Не реагируй на "
                    "сервисные сообщения о входе/выходе участников. Допустимые значения: "
                    + ", ".join(TELEGRAM_REACTIONS)
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "message_id": {
                            "type": "integer",
                        },
                        "emoji": {
                            "type": "string",
                            "enum": TELEGRAM_REACTIONS,
                        },
                    },
                    "required": [
                        "message_id",
                        "emoji",
                    ],
                },
            ),

            types.FunctionDeclaration(
                name="update_chat_style",
                description=(
                    "Обновляет компактный профиль общей манеры общения этого чата. "
                    f"Вызывай только если накопилось не меньше {MIN_MESSAGES_FOR_STYLE_UPDATE} "
                    "новых сообщений и видны устойчивые повторяющиеся особенности. "
                    "Передавай полный профиль. Не сохраняй личные секреты, индивидуальные "
                    "характеристики участников или случайную разовую шутку."
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "style_profile": {
                            "type": "string",
                            "description": "Профиль до 2500 символов.",
                        },
                    },
                    "required": ["style_profile"],
                },
            ),

            types.FunctionDeclaration(
                name="search_own_messages",
                description=(
                    "Ищет старые текстовые сообщения, которые сам Мембу отправлял "
                    "в текущем чате. Используй, чтобы проверить, что говорил раньше, "
                    "найти собственное обещание или вернуться к теме. Поиск буквальный, "
                    "по словам и фразе. Не вызывай перед каждым ответом."
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                        },
                        "limit": {
                            "type": "integer",
                        },
                    },
                    "required": ["query"],
                },
            ),

            types.FunctionDeclaration(
                name="edit_own_message",
                description=(
                    "Крайне редко редактирует одно собственное текстовое сообщение Мембу. "
                    "Только при существенной фактической ошибке, вводящей в заблуждение "
                    "формулировке или пропущенной важной детали. Не переписывай шутки, "
                    "стиль и обычные формулировки ради красоты. Только одно редактирование "
                    "сообщения в течение 10 минут; не чаще одной правки за 5 минут и трёх "
                    "за час на весь чат. message_id должен быть из контекста или поиска. "
                    "Нельзя менять чужие сообщения или стикеры. Укажи содержательную причину."
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "message_id": {
                            "type": "integer",
                        },
                        "text": {
                            "type": "string",
                        },
                        "reason": {
                            "type": "string",
                        },
                    },
                    "required": [
                        "message_id",
                        "text",
                        "reason",
                    ],
                },
            ),
        ]

        if self.stickers:
            declarations.append(
                types.FunctionDeclaration(
                    name="send_sticker",
                    description=(
                        "Отправляет один стикер из личного каталога Мембу. Это может быть "
                        "самостоятельная реплика. Используй редко и только когда стикер "
                        "точно подходит. Обычно выбери либо текст, либо стикер, либо реакцию. "
                        "Не дублируй один смысл текстом и стикером. Каталог:\n"
                        + self._sticker_catalog_for_model()
                    ),
                    parameters_json_schema={
                        "type": "object",
                        "properties": {
                            "key": {
                                "type": "string",
                                "enum": list(self.stickers),
                            },
                            "reply_to_message_id": {
                                "type": "integer",
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

    @staticmethod
    def _parse_datetime(value) -> datetime | None:
        if isinstance(value, datetime):
            result = value
        elif isinstance(value, str):
            try:
                result = datetime.fromisoformat(
                    value.replace("Z", "+00:00")
                )
            except ValueError:
                return None
        else:
            return None

        if result.tzinfo is None:
            result = result.replace(tzinfo=timezone.utc)

        return result.astimezone(timezone.utc)

    # =========================================
    # СОХРАНЕНИЕ ОТПРАВЛЕННЫХ СООБЩЕНИЙ
    # =========================================

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

    # =========================================
    # ЛИЧНЫЕ СООБЩЕНИЯ
    # =========================================

    async def _send_private_message(
        self,
        text: str,
        requester: dict | None,
        allowed: bool,
    ) -> dict:
        if not allowed or not requester or not requester.get("user_id"):
            return {
                "sent": False,
                "reason": "not_explicitly_requested",
                "error": "Личное сообщение не разрешено для этого события.",
            }

        user_id = int(requester["user_id"])

        if not await asyncio.to_thread(
            self.storage.has_private_chat_opt_in,
            user_id,
        ):
            username = self.bot_user.username

            start_link = (
                f"https://t.me/{username}?start=private"
                if username
                else None
            )

            return {
                "sent": False,
                "reason": "user_has_not_started_private_chat",
                "error": "Пользователь ещё не открыл личный чат с Мембу и не нажал Start.",
                "start_link": start_link,
            }

        text = str(text or "").strip()

        if not text:
            return {
                "sent": False,
                "error": "Нельзя отправить пустое личное сообщение.",
            }

        text = text[:4000]

        try:
            sent = await self.bot.send_message(
                chat_id=user_id,
                text=text,
            )
        except TelegramAPIError as exc:
            return {
                "sent": False,
                "reason": "telegram_rejected_message",
                "error": str(exc)[:350],
            }

        history_saved = True

        try:
            await self._store_bot_message(
                user_id,
                sent,
                text,
                None,
                None,
            )
        except Exception:
            history_saved = False
            logging.exception(
                "Личное сообщение отправлено, но не записано в Supabase: user=%s",
                user_id,
            )

        return {
            "sent": True,
            "user_id": user_id,
            "message_id": sent.message_id,
            "history_saved": history_saved,
        }

    # =========================================
    # РЕДАКТИРОВАНИЕ СОБСТВЕННЫХ СООБЩЕНИЙ
    # =========================================

    async def _edit_own_message(
        self,
        chat_id: int,
        args: dict,
        is_stale: Callable[[], bool] | None = None,
    ) -> dict:
        if is_stale and is_stale():
            return {
                "edited": False,
                "cancelled": True,
                "reason": "Контекст устарел.",
            }

        try:
            message_id = int(args.get("message_id"))
        except (TypeError, ValueError):
            return {
                "edited": False,
                "error": "Некорректный message_id.",
            }

        new_text = str(args.get("text", "")).strip()
        reason = str(args.get("reason", "")).strip()

        if not new_text:
            return {
                "edited": False,
                "error": "Новый текст пуст.",
            }

        if len(new_text) > EDIT_MAX_TEXT_LENGTH:
            return {
                "edited": False,
                "error": "Новый текст слишком длинный.",
            }

        if len(reason) < 12:
            return {
                "edited": False,
                "error": "Укажи содержательную причину, а не косметическое улучшение.",
            }

        reason = reason[:300]

        lock = self._edit_locks.setdefault(
            chat_id,
            asyncio.Lock(),
        )

        async with lock:
            local_key = (chat_id, message_id)

            if local_key in self._edited_message_ids:
                return {
                    "edited": False,
                    "error": "Это сообщение уже редактировалось.",
                }

            try:
                message = await asyncio.to_thread(
                    self.storage.get_own_message,
                    chat_id,
                    message_id,
                    self.bot_user.id,
                )
            except Exception:
                logging.exception(
                    "Не удалось проверить сообщение перед редактированием"
                )
                return {
                    "edited": False,
                    "error": "Не удалось проверить сообщение в базе.",
                }

            if not message:
                return {
                    "edited": False,
                    "error": "Сообщение не найдено среди собственных сообщений Мембу.",
                }

            old_text = str(message.get("text", ""))

            if old_text.startswith("[СТИКЕР:") or old_text.startswith("[ВЛОЖЕНИЕ:"):
                return {
                    "edited": False,
                    "error": "Медиа и стикеры этим инструментом редактировать нельзя.",
                }

            created_at = self._parse_datetime(
                message.get("created_at")
            )

            if created_at is None:
                return {
                    "edited": False,
                    "error": "Не удалось определить время отправки.",
                }

            age = (
                datetime.now(timezone.utc) - created_at
            ).total_seconds()

            if age < -60 or age > EDIT_WINDOW_SECONDS:
                return {
                    "edited": False,
                    "error": "Сообщение уже нельзя редактировать: прошло больше 10 минут.",
                }

            if (
                " ".join(old_text.split()).casefold()
                == " ".join(new_text.split()).casefold()
            ):
                return {
                    "edited": False,
                    "error": "Новый текст практически не отличается от исходного.",
                }

            try:
                already_edited = await asyncio.to_thread(
                    self.storage.has_message_been_edited,
                    chat_id,
                    message_id,
                )

                if already_edited:
                    return {
                        "edited": False,
                        "error": "Это сообщение уже редактировалось.",
                    }

                recent = await asyncio.to_thread(
                    self.storage.recent_message_edits,
                    chat_id,
                    self.bot_user.id,
                    60,
                    EDIT_MAX_PER_HOUR,
                )

            except Exception:
                logging.exception(
                    "Не удалось проверить журнал редактирования"
                )
                return {
                    "edited": False,
                    "error": "Не удалось проверить лимиты редактирования.",
                }

            if len(recent) >= EDIT_MAX_PER_HOUR:
                return {
                    "edited": False,
                    "error": "Достигнут лимит: три правки за час на весь чат.",
                }

            if recent:
                last_edit = self._parse_datetime(
                    recent[0].get("edited_at")
                )

                if last_edit is None:
                    return {
                        "edited": False,
                        "error": "Не удалось проверить время последней правки.",
                    }

                seconds_since = (
                    datetime.now(timezone.utc) - last_edit
                ).total_seconds()

                if seconds_since < EDIT_COOLDOWN_SECONDS:
                    remaining = max(
                        1,
                        int(EDIT_COOLDOWN_SECONDS - seconds_since),
                    )

                    return {
                        "edited": False,
                        "error": (
                            f"Слишком частые правки. "
                            f"Подожди примерно {remaining} секунд."
                        ),
                    }

            if is_stale and is_stale():
                return {
                    "edited": False,
                    "cancelled": True,
                    "reason": "В чате появилось новое сообщение.",
                }

            try:
                await self.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text=new_text,
                )
            except TelegramAPIError as exc:
                return {
                    "edited": False,
                    "error": str(exc)[:350],
                }

            self._edited_message_ids.add(local_key)

            try:
                history = await asyncio.to_thread(
                    self.storage.record_message_edit,
                    chat_id,
                    message_id,
                    self.bot_user.id,
                    old_text,
                    new_text,
                    reason,
                )
            except Exception:
                logging.exception(
                    "Telegram изменил сообщение, но журнал Supabase недоступен"
                )

                return {
                    "edited": True,
                    "message_id": message_id,
                    "history_saved": False,
                    "warning": "Telegram принял правку, но историю не удалось сохранить.",
                }

            if not history.get("recorded"):
                return {
                    "edited": True,
                    "message_id": message_id,
                    "history_saved": False,
                    "warning": "Telegram принял правку, но Supabase не подтвердил журнал.",
                }

            self._mark_chat_activity(chat_id)

            logging.info(
                "MEMBU_EDIT | chat=%s | message=%s | reason=%s",
                chat_id,
                message_id,
                reason[:120],
            )

            return {
                "edited": True,
                "message_id": message_id,
                "history_saved": True,
            }

    # =========================================
    # ВЫПОЛНЕНИЕ ИНСТРУМЕНТОВ
    # =========================================

    async def _execute_tool(
        self,
        name: str,
        args: dict,
        chat_id: int,
        branch_id: str | None,
        is_stale: Callable[[], bool] | None = None,
        trigger_user: dict | None = None,
        allow_private_message: bool = False,
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
            style_profile = str(args.get("style_profile", "")).strip()

            if not style_profile:
                return {"error": "Профиль манеры общения пуст."}

            return await asyncio.to_thread(
                self.storage.update_chat_style,
                chat_id,
                style_profile,
                MIN_MESSAGES_FOR_STYLE_UPDATE,
            )

        if name == "search_own_messages":
            query = str(args.get("query", "")).strip()

            if len(query) < 2:
                return {
                    "error": "Для поиска укажи хотя бы два символа."
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
                    "Ошибка поиска собственных сообщений"
                )
                return {
                    "error": "Не удалось выполнить поиск по истории."
                }

            return {
                "messages": messages,
                "matched": len(messages),
                "search_type": "literal_substring",
            }

        if name == "edit_own_message":
            return await self._edit_own_message(
                chat_id,
                args,
                is_stale,
            )

        if name == "send_private_message":
            text = str(args.get("text", "")).strip()

            if is_stale and is_stale():
                return {
                    "sent": False,
                    "cancelled": True,
                    "reason": "Контекст устарел.",
                }

            return await self._send_private_message(
                text,
                trigger_user,
                allow_private_message,
            )

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
                f"[СТИКЕР: {key}] {sticker['description']} "
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

    # =========================================
    # ОСНОВНОЙ ПРОЦЕСС GEMINI
    # =========================================

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
        trigger_user: dict | None = None,
        allow_private_message: bool = False,
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
                "private_message_allowed": bool(allow_private_message),
            },
            "trigger_user": trigger_user,
            "recent_messages": recent_messages,
            "current_branch_messages": branch_messages,
            "active_branches": active_branches,
            "chat_style": chat_style,
        }

        user_content = (
            "Ты получил новую ситуацию из Telegram-чата. Сначала пойми, что происходит "
            "и кто с кем разговаривает. Ты сам выбираешь, нужно ли участвовать. Можешь "
            "промолчать. Не отвечай автоматически на каждое сообщение.\n\n"

            "Если хочешь написать текст, вызови send_message. Обычный текстовый ответ "
            "модели сам по себе не отправляется.\n\n"

            "ЛИЧНЫЕ ПЕРЕПИСКИ. Если текущий trigger.reason — direct_mention_or_reply, "
            "а контекст относится к личному чату, это обычный разговор один на один: "
            "отвечай через send_message, и сообщение попадёт в эту личную переписку. "
            "История личной переписки хранится отдельно от групповой.\n\n"

            "ОТПРАВКА ИЗ ГРУППЫ В ЛИЧКУ. send_private_message можно использовать "
            "только когда trigger.private_message_allowed=true. Это означает, что "
            "конкретный участник явно попросил написать ему в личку. Получатель уже "
            "зафиксирован системой в trigger_user; ты не выбираешь и не меняешь адресата. "
            "Нельзя писать другим участникам, отправлять сообщения без просьбы, делать "
            "рассылки или повторять попытки. Если Telegram или система отказали, объясни "
            "причину в группе через send_message. Если пользователь ещё не нажал Start "
            "в личном чате, сообщи об этом и при необходимости дай ссылку start_link "
            "из результата инструмента. Не выдумывай причину ошибки.\n\n"

            "ЕСЛИ ТЕБЯ УДАЛИЛИ ИЗ ГРУППЫ. Если trigger.reason равен "
            "bot_removed_from_chat, trigger_user — заранее назначенный администратор, "
            "который разрешил личные сообщения. Можешь отправить ему одно короткое "
            "сообщение через send_private_message: спокойно и в своём стиле спроси, "
            "что произошло или можно ли вернуться. Не дави, не обвиняй и не отправляй "
            "повторные просьбы. В этом сценарии не пытайся писать в уже недоступную группу.\n\n"

            "РЕАКЦИИ. Если достаточно реакции на конкретное сообщение, можешь вызвать "
            "send_reaction. Используй только точный message_id из контекста. Не угадывай "
            "идентификаторы и не реагируй на сервисные события.\n\n"

            "СТИКЕРЫ. Если уместнее стикер, вызови send_sticker и выбери только "
            "существующий key из каталога. Обычно выбирай один способ: текст, стикер "
            "или реакцию. Не дублируй один смысл текстом и стикером.\n\n"

            "МЕДИА. В сообщениях могут быть маркеры [ВЛОЖЕНИЕ: ...]. Они сообщают "
            "только тип присланного материала и некоторые технические сведения. Ты НЕ "
            "получил само изображение, GIF, видео, стикер или аудио и не можешь знать, "
            "что именно на них изображено или сказано. Не делай вид, что видел фото, "
            "прочитал картинку, посмотрел видео или услышал голосовое. Можешь учитывать "
            "факт отправки и подпись автора, спросить, что там, или промолчать. Для входящего "
            "стикера эмодзи и название набора — лишь метаданные, а не описание картинки.\n\n"

            "ПОИСК СОБСТВЕННЫХ СООБЩЕНИЙ. Если нужно вспомнить, что ты сам говорил, "
            "можешь вызвать search_own_messages. Поиск буквальный по словам или фразе. "
            "Не вызывай без причины и не ищи историю перед каждым ответом.\n\n"

            "РЕДАКТИРОВАНИЕ. edit_own_message — исключительный инструмент. Используй "
            "только при существенной фактической ошибке, вводящей в заблуждение "
            "формулировке или пропущенной важной детали. Не переписывай шутки, стиль "
            "и обычные формулировки ради красоты. Только одно редактирование сообщения "
            "в первые 10 минут, не чаще одной правки за 5 минут и трёх за час на чат. "
            "Не редактируй чужие сообщения и стикеры. Если правка отклонена, не обходи "
            "ограничение.\n\n"

            "ПРОФИЛЬ ЧАТА. chat_style — описание общей манеры общения компании. "
            "Используй его как дополнительный ориентир, а не как новую личность. Сохраняй "
            "собственный характер Мембу. Не копируй механически одного человека.\n\n"

            "Если в chat_style.new_messages_since_analysis накопилось не меньше "
            f"{MIN_MESSAGES_FOR_STYLE_UPDATE}, оцени устойчивые особенности общения. "
            "Если они есть — можешь вызвать update_chat_style с полным кратким профилем. "
            "Не делай этого ради галочки.\n\n"

            "Если trigger.reason равен long_silence_initiative, это не просьба обязательно "
            "писать. Начинай разговор только при естественном конкретном поводе.\n\n"

            "Если trigger.reason связан с member_joined, member_left или "
            "membership_changed, учти событие, но не обязан комментировать каждого.\n\n"

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

        private_failure_result = None
        private_message_attempted = False

        for _ in range(8):
            if is_stale and is_stale():
                return

            response = await self._generate_content_with_failover(
                model=self.model,
                contents=contents,
                config=self.config,
            )

            calls = response.function_calls or []

            if not calls:
                # Обычный текст модели не отправляется автоматически.
                # Исключение: после неудачной явной просьбы написать в личку
                # система обязана сообщить честную причину в исходной группе.
                if (
                    private_failure_result
                    and reason != "bot_removed_from_chat"
                    and not (is_stale and is_stale())
                ):
                    failure_reason = private_failure_result.get("reason")
                    start_link = private_failure_result.get("start_link")

                    if failure_reason == "user_has_not_started_private_chat":
                        fallback = (
                            "Я бы написал, но ты ещё не открыл мой личный чат. "
                            "Нажми Start, и тогда смогу тебе написать: "
                            + (start_link or "найди @membu_bot в Telegram")
                        )
                    else:
                        details = (
                            private_failure_result.get("error")
                            or "Telegram не принял сообщение"
                        )

                        fallback = (
                            "Попробовал написать тебе в личку, но Telegram не пустил. "
                            "Причина: " + str(details)[:220]
                        )

                    try:
                        sent = await self.bot.send_message(
                            chat_id=chat_id,
                            text=fallback[:4000],
                        )

                        await self._store_bot_message(
                            chat_id,
                            sent,
                            fallback,
                            branch_id,
                            None,
                        )

                    except TelegramAPIError:
                        logging.exception(
                            "Не удалось объяснить отказ отправки в личку: chat=%s",
                            chat_id,
                        )

                return

            if response.candidates and response.candidates[0].content:
                contents.append(
                    response.candidates[0].content
                )

            performed_chat_action = False
            private_send_failed = False

            for call in calls:
                if is_stale and is_stale():
                    return

                if call.name == "send_private_message":
                    if private_message_attempted:
                        result = {
                            "sent": False,
                            "reason": "already_attempted",
                            "error": (
                                "В этом ходе уже была одна попытка написать в личку. "
                                "Повторная попытка запрещена."
                            ),
                        }

                        contents.append(
                            types.Content(
                                role="user",
                                parts=[
                                    types.Part.from_function_response(
                                        name=call.name,
                                        response={"result": result},
                                    )
                                ],
                            )
                        )

                        private_send_failed = True

                        if private_failure_result is None:
                            private_failure_result = result

                        continue

                    private_message_attempted = True

                result = await self._execute_tool(
                    name=call.name,
                    args=call.args or {},
                    chat_id=chat_id,
                    branch_id=branch_id,
                    is_stale=is_stale,
                    trigger_user=trigger_user,
                    allow_private_message=allow_private_message,
                )

                if call.name in {
                    "send_message",
                    "send_sticker",
                    "edit_own_message",
                }:
                    if result.get("sent") or result.get("edited"):
                        performed_chat_action = True

                if call.name == "send_reaction" and result.get("reacted"):
                    performed_chat_action = True

                if call.name == "send_private_message":
                    if result.get("sent"):
                        performed_chat_action = True
                    else:
                        private_send_failed = True

                        if private_failure_result is None:
                            private_failure_result = result

                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=call.name,
                                response={"result": result},
                            )
                        ],
                    )
                )

            if performed_chat_action:
                return

            # Если личное сообщение не прошло, модель получает результат
            # инструмента и может объяснить причину в исходном чате.
            if private_send_failed:
                continue
