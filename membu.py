import random
import time
import uuid

from dataclasses import dataclass


@dataclass
class Decision:
    call_ai: bool
    reason: str
    branch_id: str | None
    direct: bool
    wait_seconds: float = 0.0


class MembuCore:
    def __init__(
        self,
        spontaneous_probability: float = 0.008,
        spontaneous_cooldown: float = 45,
        branch_cooldown: float = 10,
        direct_cooldown: float = 5,
    ):
        self.spontaneous_probability = (
            spontaneous_probability
        )

        self.spontaneous_cooldown = (
            spontaneous_cooldown
        )

        self.branch_cooldown = branch_cooldown
        self.direct_cooldown = direct_cooldown

        self.last_spontaneous: dict[int, float] = {}

    def _cooldown_wait(
        self,
        key: str,
        cooldown: float,
        now: float,
    ) -> float:
        # Задержку между ответами теперь ведёт main.py отдельно для каждого
        # пользователя (дебаунс) и очередь чата. Здесь ожидания нет, и
        # словарь с ключами веток больше не растёт.
        return 0.0

    def inspect(
        self,
        event: dict,
        active_branches: list[dict],
    ) -> Decision:

        now = time.monotonic()

        chat_id = event["chat_id"]

        mentioned = bool(
            event.get("mentioned")
        )

        replied_to_bot = bool(
            event.get("replied_to_bot")
        )

        private_chat = bool(
            event.get("private_chat")
        )

        reply_to_branch_id = event.get(
            "reply_to_branch_id"
        )

        # 1. Прямое обращение.
        # Оно всегда передаётся Gemini.
        if mentioned or replied_to_bot or private_chat:

            branch_id = (
                reply_to_branch_id
                or str(uuid.uuid4())
            )

            wait = self._cooldown_wait(
                f"{chat_id}:{branch_id}",
                self.direct_cooldown,
                now,
            )

            return Decision(
                call_ai=True,
                reason="direct_mention_or_reply",
                branch_id=branch_id,
                direct=True,
                wait_seconds=wait,
            )

        # 2. Ответ на сообщение, уже принадлежащее ветке.
        if reply_to_branch_id:

            wait = self._cooldown_wait(
                f"{chat_id}:{reply_to_branch_id}",
                self.branch_cooldown,
                now,
            )

            return Decision(
                call_ai=True,
                reason="reply_inside_active_branch",
                branch_id=reply_to_branch_id,
                direct=False,
                wait_seconds=wait,
            )

        # 3. В чате есть ровно одна активная ветка.
        # Новое сообщение передаётся Gemini как возможное
        # продолжение разговора. Адресата определит модель.
        if len(active_branches) == 1:

            branch_id = active_branches[0]["branch_id"]

            wait = self._cooldown_wait(
                f"{chat_id}:{branch_id}",
                self.branch_cooldown,
                now,
            )

            return Decision(
                call_ai=True,
                reason="message_during_active_branch",
                branch_id=branch_id,
                direct=False,
                wait_seconds=wait,
            )

        # 4. Несколько активных веток.
        # Не пытаемся угадать, к какой относится сообщение.
        # Передаём Gemini общий контекст с кандидатами.
        if len(active_branches) > 1:

            wait = self._cooldown_wait(
                f"{chat_id}:multiple_branches",
                self.branch_cooldown,
                now,
            )

            return Decision(
                call_ai=True,
                reason="multiple_active_branches",
                branch_id=None,
                direct=False,
                wait_seconds=wait,
            )

        # 5. Спонтанное внимание.
        # Иногда машина предлагает Gemini посмотреть
        # на общий разговор, даже если Мембу не упоминали.
        last_spontaneous = self.last_spontaneous.get(
            chat_id,
            0,
        )

        if (
            now - last_spontaneous
            >= self.spontaneous_cooldown
            and random.random()
            < self.spontaneous_probability
        ):

            self.last_spontaneous[chat_id] = now

            wait = self._cooldown_wait(
                f"{chat_id}:spontaneous",
                self.branch_cooldown,
                now,
            )

            return Decision(
                call_ai=True,
                reason="spontaneous_opportunity",
                branch_id=None,
                direct=False,
                wait_seconds=wait,
            )

        return Decision(
            call_ai=False,
            reason="no_attention_trigger",
            branch_id=None,
            direct=False,
        )
