from dataclasses import dataclass

from aiogram.types import Message, User


@dataclass
class Decision:
    action: str
    reason: str
    accepted: bool
    description: str


class MembuCore:

    def process(self, message: Message, bot_user: User) -> Decision:
        if message.from_user and message.from_user.is_bot:
            return self.make(
                "ignore",
                "bot_message",
                False
            )

        if message.chat.type == "private":
            return self.make(
                "reply",
                "private_message",
                True
            )

        if self.is_reply_to_bot(message, bot_user):
            return self.make(
                "reply",
                "reply_to_membu",
                True
            )

        if self.is_mentioned(message, bot_user):
            return self.make(
                "reply",
                "mentioned",
                True
            )

        return self.make(
            "ignore",
            "ordinary_group_message",
            False
        )

    def make(self, action: str, reason: str, accepted: bool) -> Decision:
        descriptions = {
            "reply": "Мембу бы ответил на это сообщение.",
            "ignore": "Мембу бы проигнорировал это сообщение.",
            "wait": "Мембу бы пока промолчал, но оставил сообщение в поле внимания."
        }

        return Decision(
            action=action,
            reason=reason,
            accepted=accepted,
            description=descriptions[action]
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

        return False

    def is_reply_to_bot(self, message: Message, bot_user: User) -> bool:
        reply = message.reply_to_message

        return (
            reply is not None
            and reply.from_user is not None
            and reply.from_user.id == bot_user.id
        )
