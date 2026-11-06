import re
import time

from collections import defaultdict, deque
from dataclasses import dataclass, field

from aiogram.types import Message, User


@dataclass
class Decision:
    action: str
    reason: str
    score: float
    details: list[str] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.action == "reply"

    @property
    def description(self) -> str:
        descriptions = {
            "reply": "Мембу решил бы ответить.",
            "wait": "Мембу пока подождал бы.",
            "ignore": "Мембу проигнорировал бы сообщение.",
        }
        return descriptions[self.action]


class MembuCore:
    def __init__(self):
        self.history = defaultdict(lambda: deque(maxlen=40))
        self.last_reply = defaultdict(float)

    def process(self, message: Message, bot_user: User) -> Decision:
        if not message.from_user:
            return self.make("ignore", "no_sender", 0)

        if message.from_user.is_bot:
            return self.make("ignore", "bot_message", 0)

        text = (message.text or message.caption or "").strip()
        chat_id = message.chat.id
        now = time.monotonic()

        history = self.history[chat_id]

        # Удаляем устаревшие сообщения из контекста
        while history and now - history[0]["time"] > 180:
            history.popleft()

        mentioned = self.is_mentioned(message, bot_user)
        replied = self.is_reply_to_bot(message, bot_user)

        # Сохраняем сообщение в истории
        history.append({
            "user_id": message.from_user.id,
            "text": text,
            "time": now,
        })

        # Личные сообщения
        if message.chat.type == "private":
            return self.make(
                "reply",
                "private_message",
                10,
                ["private_chat"],
            )

        score = 0.0
        reasons = []

        # Прямое обращение
        if mentioned:
            score += 8
            reasons.append("mentioned")

        if replied:
            score += 8
            reasons.append("reply_to_bot")

        # Вопрос
        if self.is_question(text):
            score += 2
            reasons.append("question")

        # Содержательность сообщения
        if self.has_substance(text):
            score += 1.5
            reasons.append("substantial_message")

        # Обращение к группе
        if self.has_direct_address(text):
            score += 0.8
            reasons.append("direct_address")

        # Участие нескольких людей в обсуждении
        recent = [
            item for item in history
            if now - item["time"] < 45
        ]

        participants = {
            item["user_id"] for item in recent
        }

        if len(participants) >= 2:
            score += 0.8
            reasons.append("group_discussion")

        # Активность переписки
        if len(recent) >= 5:
            score += 0.7
            reasons.append("active_chat")

        # Слишком быстрый разговор — лучше не перебивать
        if len(recent) >= 9:
            score -= 1.5
            reasons.append("fast_conversation")

        # Короткие реакции
        if self.is_short_reaction(text):
            score -= 2
            reasons.append("short_reaction")

        # Медиа без текста
        has_media = any([
            message.photo,
            message.video,
            message.document,
            message.animation,
            message.sticker,
            message.voice,
            message.video_note,
        ])

        if has_media and not text:
            score -= 1.5
            reasons.append("media_without_text")

        # Защита от слишком частых ответов
        elapsed = now - self.last_reply[chat_id]

        if elapsed < 20:
            score -= 1.5
            reasons.append("reply_cooldown")
        elif elapsed < 45:
            score -= 0.7
            reasons.append("recent_reply")

        # Прямое обращение всегда имеет приоритет
        if mentioned or replied:
            score = max(score, 8)

        score = round(score, 2)

        if score >= 4.5:
            self.last_reply[chat_id] = now
            return self.make(
                "reply",
                self.format_reasons(reasons),
                score,
                reasons,
            )

        if score >= 1.8:
            return self.make(
                "wait",
                self.format_reasons(reasons),
                score,
                reasons,
            )

        return self.make(
            "ignore",
            self.format_reasons(reasons),
            score,
            reasons,
        )

    def make(
        self,
        action: str,
        reason: str,
        score: float,
        details: list[str] | None = None,
    ) -> Decision:
        return Decision(
            action=action,
            reason=reason,
            score=score,
            details=details or [],
        )

    def is_question(self, text: str) -> bool:
        if not text:
            return False

        if "?" in text or "？" in text:
            return True

        question_words = (
            "кто", "что", "где", "когда", "почему",
            "зачем", "как", "какой", "какая", "какие",
            "сколько", "чего", "кому", "чем", "ли ",
        )

        normalized = text.lower().strip()

        return normalized.startswith(question_words)

    def is_mentioned(self, message: Message, bot_user: User) -> bool:
        text = message.text or message.caption or ""
        entities = message.entities or message.caption_entities or []

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

    def has_substance(self, text: str) -> bool:
        if not text:
            return False

        words = text.split()

        return len(words) >= 8 or (
            len(words) >= 5 and len(text) >= 35
        )

    def has_direct_address(self, text: str) -> bool:
        if not text:
            return False

        patterns = [
            r"\bребят\b",
            r"\bнарод\b",
            r"\bпарни\b",
            r"\bдрузья\b",
            r"\bслушайте\b",
            r"\bсмотрите\b",
            r"\bкстати\b",
        ]

        return any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in patterns
        )

    def is_short_reaction(self, text: str) -> bool:
        if not text:
            return False

        normalized = text.lower().strip()
        words = normalized.split()

        reactions = {
            "ахах", "ахахах", "хах", "хаха",
            "лол", "кек", "да", "нет", "ага",
            "неа", "пон", "ясно", "ок", "окей",
            "жесть", "мда", "ого", "понятно",
        }

        return (
            len(words) <= 2
            and (
                normalized in reactions
                or len(normalized) <= 3
            )
        )

    def format_reasons(self, reasons: list[str]) -> str:
        if not reasons:
            return "no_significant_signals"

        return ", ".join(reasons)
