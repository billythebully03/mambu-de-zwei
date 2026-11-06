import asyncio
import logging
import random
import re
import time

from collections import defaultdict, deque
from dataclasses import dataclass

from aiogram.types import Message, User


logger = logging.getLogger("membu.core")


@dataclass
class MessageRecord:
    message_id: int
    user_id: int
    name: str
    text: str
    timestamp: float
    mentioned: bool
    replied_to: bool


@dataclass
class Decision:
    action: str
    reason: str
    score: float
    target_message_id: int
    description: str


class MembuCore:

    def __init__(self):

        self.history = defaultdict(
            lambda: deque(maxlen=40)
        )

        self.pending = defaultdict(list)
        self.tasks = {}

        self.last_reply = defaultdict(float)

        # Время сбора сообщений
        self.debounce = 3.5

        # Минимальный интервал между гипотетическими ответами
        self.cooldown = 45.0

        # Контекст для оценки социальной ситуации
        self.context_window = 25.0

        # Вероятностная вариативность
        self.randomness = 0.45

        self.short_reactions = {
            "ок", "окей", "ага", "угу", "да",
            "нет", "ясно", "понятно", "хах",
            "ахах", "лол", "кек", "спс",
            "спасибо", "👍", "👌", "😂"
        }

        logger.info("Membu Core initialized")

    # -----------------------------------------
    # ПОЛУЧЕНИЕ СООБЩЕНИЙ
    # -----------------------------------------

    async def submit(
        self,
        message: Message,
        bot_user: User,
        reporter
    ):

        user = message.from_user

        if not user or user.is_bot:
            return

        text = (
            message.text
            or message.caption
            or "[медиа]"
        ).strip()

        record = MessageRecord(
            message_id=message.message_id,
            user_id=user.id,
            name=user.full_name,
            text=text[:1500],
            timestamp=time.time(),
            mentioned=self.is_mentioned(
                message,
                bot_user
            ),
            replied_to=self.is_reply_to_bot(
                message,
                bot_user
            )
        )

        chat_id = message.chat.id

        self.history[chat_id].append(record)
        self.pending[chat_id].append(record)

        logger.info(
            "MESSAGE | chat=%s | user=%s | text=%s",
            chat_id,
            user.full_name,
            text[:100]
        )

        # Один сборщик на чат.
        # Новые сообщения не перезапускают таймер.

        if chat_id not in self.tasks:
            self.tasks[chat_id] = asyncio.create_task(
                self.worker(
                    chat_id,
                    message.chat.type,
                    reporter
                )
            )

    # -----------------------------------------
    # ОБРАБОТЧИК ПАКЕТОВ
    # -----------------------------------------

    async def worker(
        self,
        chat_id,
        chat_type,
        reporter
    ):

        current_task = asyncio.current_task()

        try:

            while True:

                await asyncio.sleep(self.debounce)

                batch = self.pending[chat_id]

                if not batch:
                    break

                self.pending[chat_id] = []

                decision = self.evaluate(
                    chat_id,
                    batch,
                    chat_type
                )

                logger.info(
                    "DECISION | chat=%s | action=%s | score=%.2f | reason=%s",
                    chat_id,
                    decision.action,
                    decision.score,
                    decision.reason
                )

                await reporter(
                    chat_id,
                    decision
                )

                # Если во время анализа пришли новые сообщения,
                # они попадут в следующую итерацию.

                if not self.pending[chat_id]:
                    break

        except asyncio.CancelledError:
            raise

        except Exception:

            logger.exception(
                "WORKER ERROR | chat=%s",
                chat_id
            )

        finally:

            if self.tasks.get(chat_id) is current_task:
                self.tasks.pop(chat_id, None)

            # Защита от потери сообщения при завершении задачи.
            if self.pending[chat_id] and chat_id not in self.tasks:

                self.tasks[chat_id] = asyncio.create_task(
                    self.worker(
                        chat_id,
                        chat_type,
                        reporter
                    )
                )

    # -----------------------------------------
    # ГЛАВНАЯ ВЫЧИСЛИТЕЛЬНАЯ МАШИНА
    # -----------------------------------------

    def evaluate(
        self,
        chat_id,
        batch,
        chat_type
    ):

        now = time.time()

        recent = [
            item
            for item in self.history[chat_id]
            if now - item.timestamp <= self.context_window
        ]

        participants = {
            item.user_id
            for item in recent
        }

        batch_users = {
            item.user_id
            for item in batch
        }

        target = batch[-1]

        # Берём последнее содержательное сообщение.
        for item in reversed(batch):
            if item.text != "[медиа]":
                target = item
                break

        score = 0.0
        reasons = []

        # -------------------------------------
        # 1. ПРЯМОЕ ОБРАЩЕНИЕ
        # -------------------------------------

        if chat_type == "private":

            score += 4.2
            reasons.append("private_conversation")

        else:

            if any(item.mentioned for item in batch):

                score += 4.8
                reasons.append("direct_mention")

            if any(item.replied_to for item in batch):

                score += 5.0
                reasons.append("reply_to_membu")

        # -------------------------------------
        # 2. ИНТЕРЕСНОСТЬ СООБЩЕНИЯ
        # -------------------------------------

        text = target.text.lower().strip()

        if "?" in text:

            score += 0.8
            reasons.append("question_detected")

        if "!" in text:

            score += 0.25

        if len(text) > 100:

            score += 0.35
            reasons.append("substantial_message")

        if len(text) > 300:

            score += 0.25

        interesting_patterns = [
            r"\bпочему\b",
            r"\bзачем\b",
            r"\bпредставь\b",
            r"\bприкинь\b",
            r"\bинтересно\b",
            r"\bнеужели\b",
            r"\bчто если\b",
            r"\bкак думаете\b",
            r"\bкак считаете\b",
            r"\bкстати\b",
            r"\bлол\b",
            r"\bахах\b",
        ]

        if any(
            re.search(pattern, text)
            for pattern in interesting_patterns
        ):

            score += 0.45
            reasons.append("conversation_hook")

        # -------------------------------------
        # 3. КАЧЕСТВО СООБЩЕНИЯ
        # -------------------------------------

        normalized = re.sub(
            r"[^\w\s]",
            "",
            text
        ).strip()

        if normalized in self.short_reactions:

            score -= 2.2
            reasons.append("short_reaction")

        if len(normalized) <= 3:

            score -= 1.0
            reasons.append("very_short_message")

        if text == "[медиа]":

            score -= 2.0
            reasons.append("media_without_caption")

        # -------------------------------------
        # 4. СОЦИАЛЬНАЯ СИТУАЦИЯ
        # -------------------------------------

        if len(recent) >= 4:

            score -= 0.5
            reasons.append("active_conversation")

        if len(recent) >= 7:

            score -= 0.8
            reasons.append("very_active_conversation")

        if len(participants) >= 3:

            score -= 0.35
            reasons.append("multiple_participants")

        if len(batch) >= 3:

            score -= 0.45
            reasons.append("message_burst")

        if len(batch_users) >= 2:

            score += 0.25
            reasons.append("shared_conversation")

        # -------------------------------------
        # 5. ПОВТОРНАЯ АКТИВНОСТЬ
        # -------------------------------------

        previous_reply = self.last_reply[chat_id]

        elapsed = now - previous_reply

        if previous_reply:

            if elapsed < 15:

                score -= 2.8
                reasons.append("very_recent_intervention")

            elif elapsed < self.cooldown:

                score -= 1.5
                reasons.append("reply_cooldown")

        # -------------------------------------
        # 6. ПЕРЕМЕННАЯ ХАРИЗМА
        # -------------------------------------

        score += random.uniform(
            -self.randomness,
            self.randomness
        )

        # -------------------------------------
        # 7. ПРИНЯТИЕ РЕШЕНИЯ
        # -------------------------------------

        if score >= 4.2:

            action = "reply"

        elif score >= 2.0:

            action = "wait"

        else:

            action = "ignore"

        # Фиксируем гипотетическое вмешательство,
        # даже если реального ответа пока нет.

        if action == "reply":

            self.last_reply[chat_id] = now

        reason = (
            ", ".join(reasons)
            if reasons
            else "no_social_trigger"
        )

        descriptions = {
            "reply": "Мембу бы ответил на этот фрагмент переписки.",
            "wait": "Мембу бы пока подождал развития разговора.",
            "ignore": "Мембу бы проигнорировал эту ситуацию."
        }

        return Decision(
            action=action,
            reason=reason,
            score=round(score, 2),
            target_message_id=target.message_id,
            description=descriptions[action]
        )

    # -----------------------------------------
    # TELEGRAM
    # -----------------------------------------

    def is_mentioned(self, message, bot_user):

        content = message.text or message.caption or ""

        entities = (
            message.entities
            or message.caption_entities
            or []
        )

        for entity in entities:

            if entity.type == "text_mention":

                if entity.user and entity.user.id == bot_user.id:
                    return True

            if entity.type == "mention" and bot_user.username:

                mention = content[
                    entity.offset:
                    entity.offset + entity.length
                ]

                if mention.casefold() == (
                    f"@{bot_user.username}".casefold()
                ):

                    return True

        # Также распознаём обычное написание имени.

        if re.search(
            r"\bмембу\b",
            content,
            re.IGNORECASE
        ):

            return True

        return False

    def is_reply_to_bot(self, message, bot_user):

        reply = message.reply_to_message

        return (
            reply is not None
            and reply.from_user is not None
            and reply.from_user.id == bot_user.id
        )
