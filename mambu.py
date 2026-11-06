from dataclasses import dataclass

from aiogram.types import Message, User


@dataclass
class Decision:
    accepted: bool
    reason: str
    reply_text: str | None = None


class MembuCore:

    def process(self, message: Message, bot_user: User) -> Decision:
        if message.chat.type == "private":
            return Decision(
                accepted=True,
                reason="private_message",
                reply_text="Мембу на связи. Пока работаю в тестовом режиме."
            )

        if self.is_mentioned(message, bot_user):
            return Decision(
                accepted=True,
                reason="mentioned"
            )

        if self.is_reply_to_bot(message, bot_user):
            return Decision(
                accepted=True,
                reason="reply_to_membu"
            )

        return Decision(
            accepted=False,
            reason="not_relevant"
        )

    def is_mentioned(self, message: Message, bot_user: User) -> bool:
        content = message.text or message.caption or ""
        entities = message.entities or message.caption_entities or []

        for entity in entities:
            if entity.type == "text_mention":
                if entity.user and entity.user.id == bot_user.id:
                    return True

            if entity.type == "mention" and bot_user.username:
                mention = content[entity.offset:entity.offset + entity.length]

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
