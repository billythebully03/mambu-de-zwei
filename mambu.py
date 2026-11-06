import asyncio
import random
import re
import time

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from aiogram.types import Message, User


@dataclass
class MessageRecord:
    user_id: int
    username: str
    text: str
    timestamp: float
    message_id: int
    is_question: bool
    is_mentioned: bool
    is_reply_to_bot: bool
    has_media: bool


@dataclass
class Decision:
    action: str
    reason: str
    accepted: bool
    description: str
    score: float = 0.0
    details: list[str] = field(default_factory=list)


class MembuCore:
    def __init__(
        self,
        reporter: Callable[[int, Decision, int], Awaitable[None]],
        batch_delay: float = 3.5,
    ):
        self.reporter = reporter
        self.batch_delay = batch_delay

        self.pending = defaultdict(list)
        self.workers = {}

        self.history = defaultdict(
            lambda: deque(maxlen=40)
        )

        self.last_reply_time = defaultdict(float)
        self.last_activity_time = defaultdict(float)

        self.running = True

    async def submit(self, message: Message, bot_user: User):
        if not self.running:
            return

        if message.from_user is None:
            return

        if message.from_user.is_bot:
            return

        record = self.create_record(message, bot_user)

        chat_id = message.chat.id

        self.pending[chat_id].append(record)
        self.history[chat_id].append(record)

        self.last_activity_time[chat_id] = time.monotonic()

        if chat_id not in self.workers or self.workers[chat_id].done():
            self.workers[chat_id] = asyncio.create_task(
                self.process_batch(chat_id)
            )

    def create_record(
        self,
        message: Message,
        bot_user: User,
    ) -> MessageRecord:
        content = message.text or message.caption or ""

        return MessageRecord(
            user_id=message.from_user.id,
            username=message.from_user.username
            or message.from_user.first_name
            or "unknown",
            text=content.strip(),
            timestamp=time.monotonic(),
            message_id=message.message_id,
            is_question=self.is_question(content),
            is_mentioned=self.is_mentioned(message, bot_user),
            is_reply_to_bot=self.is_reply_to_bot(message, bot_user),
            has_media=any([
                message.photo,
                message.video,
                message.document,
                message.animation,
                message.sticker,
                message.voice,
                message.video_note,
            ]),
        )

    async def process_batch(self, chat_id: int):
        try:
            while self.running:
                await asyncio.sleep(self.batch_delay)

                if not self.pending[chat_id]:
                    break

                batch = self.pending[chat_id]
                self.pending[chat_id] = []

                decision = self.evaluate(chat_id, batch)

                target_message_id = batch[-1].message_id

                await self.reporter(
                    chat_id,
                    decision,
                    target_message_id,
                )

        except asyncio.CancelledError:
            raise

        finally:
            self.workers.pop(chat_id, None)

            if self.running and self.pending[chat_id]:
                self.workers[chat_id] = asyncio.create_task(
                    self.process_batch(chat_id)
                )

    def evaluate(
        self,
        chat_id: int,
        batch: list[MessageRecord],
    ) -> Decision:
        if not batch:
            return self.make(
                "ignore",
                "empty_batch",
                0,
            )

        latest = batch[-1]
        history = list(self.history[chat_id])

        score = 0.0
        reasons = []

        # Прямое обращение к Мембу
        if any(message.is_mentioned for message in batch):
            score += 7.0
            reasons.append("direct_mention")

        # Ответ на сообщение Мембу
        if any(message.is_reply_to_bot for message in batch):
            score += 7.0
            reasons.append("reply_to_bot")

        # Личное сообщение
        # В этой версии личные сообщения не обрабатываются
        # как отдельный режим: решение остаётся частью общей логики.

        # Вопросы
        if any(message.is_question for message in batch):
            score += 1.8
            reasons.append("question")

        # Сообщение содержит обращение к собеседникам
        if any(self.has_direct_address(message.text) for message in batch):
            score += 0.8
            reasons.append("direct_address")

        # Содержательное сообщение
        if any(self.has_substance(message.text) for message in batch):
            score += 1.0
            reasons.append("substantial_message")

        # Короткие реакции обычно не требуют ответа
        if all(self.is_short_reaction(message.text) for message in batch):
            score -= 2.0
            reasons.append("short_reaction")

        # Медиа без текста редко требуют вмешательства
        if all(
            message.has_media and not message.text
            for message in batch
        ):
            score -= 1.5
            reasons.append("media_only")

        # Несколько участников могут означать живое обсуждение
        participants = {
            message.user_id
            for message in batch
        }

        if len(participants) >= 2:
            score += 0.8
            reasons.append("multiple_participants")

        # Активная переписка
        recent = [
            message
            for message in history
            if time.monotonic() - message.timestamp < 45
        ]

        if len(recent) >= 5:
            score += 0.8
            reasons.append("active_conversation")

        # Если разговор идёт слишком быстро, лучше не перебивать
        if len(recent) >= 8:
            score -= 1.2
            reasons.append("fast_conversation")

        # Если последнее сообщение очень короткое
        if len(latest.text) <= 3:
            score -= 0.7
            reasons.append("very_short_message")

        # Случайная небольшая вариативность
        score += random.uniform(-0.35, 0.35)

        # Защита от слишком частых ответов
        elapsed = (
            time.monotonic() - self.last_reply_time[chat_id]
        )

        if elapsed < 20:
            score -= 1.5
            reasons.append("reply_cooldown")

        elif elapsed < 45:
            score -= 0.7
            reasons.append("recent_reply")

        # Если обращение прямое, cooldown не должен блокировать ответ
        direct = any(
            message.is_mentioned or message.is_reply_to_bot
            for message in batch
        )

        if direct:
            score = max(score, 7.0)

        score = round(score, 2)

        if score >= 4.2:
            self.last_reply_time[chat_id] = time.monotonic()

            return self.make(
                "reply",
                self.format_reason(reasons),
                score,
                reasons,
            )

        if score >= 2.0:
            return self.make(
                "wait",
                self.format_reason(reasons),
                score,
                reasons,
            )

        return self.make(
            "ignore",
            self.format_reason(reasons),
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
        descriptions = {
            "reply": "Мембу решил бы ответить.",
            "wait": "Мембу пока подождал бы.",
            "ignore": "Мембу проигнорировал бы сообщение.",
        }

        return Decision(
            action=action,
            reason=reason,
            accepted=action == "reply",
            description=descriptions[action],
            score=score,
            details=details or [],
        )

    def format_reason(self, reasons: list[str]) -> str:
        if not reasons:
            return "no_significant_signals"

        return ", ".join(reasons)

    def is_question(self, text: str) -> bool:
        if not text:
            return False

        question_words = (
            "кто",
            "что",
            "где",
            "когда",
            "почему",
            "зачем",
            "как",
            "какой",
            "какая",
            "какие",
            "сколько",
            "чего",
            "кому",
            "чем",
            "можно",
            "будет",
            "есть ли",
        )

        normalized = text.lower().strip()

        return (
            "?" in normalized
            or normalized.startswith(question_words)
        )

    def is_mentioned(self, message: Message, bot_user: User) -> bool:
        content = message.text or message.caption or ""
        entities = message.entities or message.caption_entities or []

        for entity in entities:
            if entity.type == "text_mention":
                if entity.user and entity.user.id == bot_user.id:
                    return True

            if entity.type == "mention" and bot_user.username:
                mention = content[
                    entity.offset:entity.offset + entity.length
                ]

                if mention.casefold() == f"@{bot_user.username}".casefold():
                    return True

        if re.search(r"\bмембу\b", content, re.IGNORECASE):
            return True

        return False

    def is_reply_to_bot(
        self,
        message: Message,
        bot_user: User,
    ) -> bool:
        reply = message.reply_to_message

        return (
            reply is not None
            and reply.from_user is not None
            and reply.from_user.id == bot_user.id
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

    def has_substance(self, text: str) -> bool:
        if not text:
            return False

        words = text.split()

        if len(words) >= 8:
            return True

        if len(words) >= 5 and len(text) >= 35:
            return True

        return False

    def is_short_reaction(self, text: str) -> bool:
        if not text:
            return False

        normalized = text.lower().strip()
        words = normalized.split()

        reactions = {
            "ахах",
            "ахахах",
            "хах",
            "хаха",
            "лол",
            "кек",
            "да",
            "нет",
            "ага",
            "неа",
            "пон",
            "ясно",
            "ок",
            "окей",
            "жесть",
            "мда",
            "ого",
            "понятно",
        }

        return (
            len(words) <= 2
            and (
                normalized in reactions
                or len(normalized) <= 5
            )
        )

    async def shutdown(self):
        self.running = False

        tasks = list(self.workers.values())

        for task in tasks:
            task.cancel()

        if tasks:
            await asyncio.gather(
                *tasks,
                return_exceptions=True,
            )

        self.workers.clear()
        self.pending.clear()
