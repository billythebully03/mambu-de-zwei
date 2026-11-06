import re
import time

from collections import defaultdict, deque
from dataclasses import dataclass, field

from aiogram.types import Message, User


@dataclass
class MessageRecord:
    user_id: int
    username: str
    text: str
    timestamp: float
    message_id: int


@dataclass
class Decision:
    action: str
    score: float
    reason: str
    context: list[str] = field(default_factory=list)


class MembuCore:
    def __init__(self):
        self.history = defaultdict(lambda: deque(maxlen=40))
        self.pending = defaultdict(list)
        self.last_send = defaultdict(float)

        self.send_threshold = 3.0
        self.cooldown = 20
        self.context_lifetime = 120
        self.max_context_messages = 15

    def process(self, message: Message, bot_user: User) -> Decision:
        if not message.from_user:
            return self.make("SKIP", 0, "no_sender")

        if message.from_user.is_bot:
            return self.make("SKIP", 0, "bot_message")

        text = (message.text or message.caption or "").strip()
        chat_id = message.chat.id
        now = time.monotonic()

        if message.chat.type == "private":
            return self.make(
                "SEND",
                10,
                "private_message",
                [text] if text else [],
            )

        if text.startswith("/"):
            return self.make("SKIP", 0, "command")

        if self.is_short_reaction(text):
            return self.make("SKIP", 0, "short_reaction")

        self.clean_context(chat_id, now)

        username = (
            message.from_user.username
            or message.from_user.first_name
            or "unknown"
        )

        content = text or self.get_media_description(message)

        record = MessageRecord(
            user_id=message.from_user.id,
            username=username,
            text=content,
            timestamp=now,
            message_id=message.message_id,
        )

        self.history[chat_id].append(record)
        self.pending[chat_id].append(record)

        if len(self.pending[chat_id]) > self.max_context_messages:
            self.pending[chat_id] = self.pending[chat_id][
                -self.max_context_messages:
            ]

        return self.evaluate(message, bot_user, chat_id, text, now)

    def evaluate(
        self,
        message: Message,
        bot_user: User,
        chat_id: int,
        text: str,
        now: float,
    ) -> Decision:

        score = 0.0
        reasons = []

        mentioned = self.is_mentioned(message, bot_user)
        replied = self.is_reply_to_bot(message, bot_user)

        # Прямое обращение
        if mentioned:
            score += 8
            reasons.append("direct_mention")

        # Ответ на сообщение Мембу
        if replied:
            score += 8
            reasons.append("reply_to_membu")

        # Вопрос
        if self.is_question(text):
            score += 2
            reasons.append("question")

        # Просьба или поручение
        if self.is_request(text):
            score += 2.5
            reasons.append("request")

        # Содержательное сообщение
        if len(text.split()) >= 5:
            score += 0.8
            reasons.append("substantial_message")

        # Обращение к участникам
        if self.has_direct_address(text):
            score += 0.5
            reasons.append("direct_address")

        # Контекст переписки
        recent = list(self.pending[chat_id])

        participants = {
            item.user_id for item in recent
        }

        if len(participants) >= 2:
            score += 0.4
            reasons.append("multiple_participants")

        if len(recent) >= 4:
            score += 0.5
            reasons.append("ongoing_discussion")

        # Недавняя активность
        history = list(self.history[chat_id])

        recent_activity = [
            item for item in history
            if now - item.timestamp < 30
        ]

        if len(recent_activity) >= 7:
            score -= 0.4
            reasons.append("high_message_frequency")

        # Прямое обращение имеет приоритет
        if mentioned or replied:
            score = max(score, 8)

        score = round(score, 2)

        # Проверяем ограничение частоты запросов
        elapsed = now - self.last_send[chat_id]

        if score >= self.send_threshold:
            if elapsed < self.cooldown:
                return self.make(
                    "DEFER",
                    score,
                    "cooldown_active",
                    self.get_context(chat_id),
                )

            self.last_send[chat_id] = now

            context = self.get_context(chat_id)
            self.pending[chat_id].clear()

            return self.make(
                "SEND",
                score,
                self.format_reasons(reasons),
                context,
            )

        return self.make(
            "DEFER",
            score,
            self.format_reasons(reasons),
            self.get_context(chat_id),
        )

    def clean_context(self, chat_id: int, now: float):
        self.pending[chat_id] = [
            item for item in self.pending[chat_id]
            if now - item.timestamp <= self.context_lifetime
        ]

    def get_context(self, chat_id: int) -> list[str]:
        return [
            f"{item.username}: {item.text}"
            for item in self.pending[chat_id]
        ]

    def get_media_description(self, message: Message) -> str:
        if message.photo:
            return "[фотография]"

        if message.video:
            return "[видео]"

        if message.voice:
            return "[голосовое сообщение]"

        if message.video_note:
            return "[видеосообщение]"

        if message.document:
            return "[файл]"

        if message.sticker:
            return "[стикер]"

        if message.animation:
            return "[анимация]"

        return "[медиа]"

    def make(
        self,
        action: str,
        score: float,
        reason: str,
        context: list[str] | None = None,
    ) -> Decision:
        return Decision(
            action=action,
            score=score,
            reason=reason,
            context=context or [],
        )

    def is_mentioned(self, message: Message, bot_user: User) -> bool:
        text = message.text or message.caption or ""

        entities = (
            message.entities
            or message.caption_entities
            or []
        )

        for entity in entities:
            if entity.type == "text_mention":
                if entity.user and entity.user.id == bot_user.id:
                    return True

        if bot_user.username:
            pattern = rf"(?<!\w)@{re.escape(bot_user.username)}\b"

            if re.search(pattern, text, re.IGNORECASE):
                return True

        if re.search(r"(?<!\w)мембу(?!\w)", text, re.IGNORECASE):
            return True

        return False

    def is_reply_to_bot(self, message: Message, bot_user: User) -> bool:
        reply = message.reply_to_message

        return (
            reply is not None
            and reply.from_user is not None
            and reply.from_user.id == bot_user.id
        )

    def is_question(self, text: str) -> bool:
        if not text:
            return False

        if "?" in text or "？" in text:
            return True

        words = (
            "кто", "что", "где", "когда", "почему",
            "зачем", "как", "какой", "какая", "какие",
            "сколько", "чего", "кому", "чем",
        )

        return text.lower().strip().startswith(words)

    def is_request(self, text: str) -> bool:
        if not text:
            return False

        patterns = [
            r"\bпомоги\b",
            r"\bпомогите\b",
            r"\bподскажи\b",
            r"\bподскажите\b",
            r"\bобъясни\b",
            r"\bрасскажи\b",
            r"\bнайди\b",
            r"\bпосмотри\b",
            r"\bпроверь\b",
            r"\bсделай\b",
            r"\bможешь\b",
            r"\bдавай\b",
            r"\bнужно\b",
            r"\bнадо\b",
        ]

        return any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in patterns
        )

    def has_direct_address(self, text: str) -> bool:
        patterns = [
            r"\bребят\b",
            r"\bнарод\b",
            r"\bпарни\b",
            r"\bдрузья\b",
            r"\bслушайте\b",
            r"\bсмотрите\b",
        ]

        return any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in patterns
        )

    def is_short_reaction(self, text: str) -> bool:
        if not text:
            return False

        normalized = text.lower().strip()

        reactions = {
            "ахах", "ахахах", "хах", "хаха",
            "лол", "кек", "да", "нет", "ага",
            "неа", "пон", "ясно", "ок", "окей",
            "жесть", "мда", "ого", "понятно",
        }

        return normalized in reactions

    def format_reasons(self, reasons: list[str]) -> str:
        if not reasons:
            return "no_significant_signals"

        return ", ".join(reasons)
