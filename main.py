import asyncio
import logging
import os

from aiogram import Bot, Dispatcher, Router
from aiogram.types import Message, User

from mambu import MembuCore, Decision


TOKEN = os.getenv("TOKEN")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger("membu.main")

router = Router()

membu = MembuCore()

bot_user: User | None = None


async def report_decision(
    chat_id: int,
    decision: Decision
):

    icons = {
        "reply": "🟢",
        "wait": "🟡",
        "ignore": "⚫"
    }

    icon = icons[decision.action]

    report = (
        f"{icon} {decision.description}\n\n"
        f"Решение: {decision.action.upper()}\n"
        f"Оценка: {decision.score}\n"
        f"Причина: {decision.reason}"
    )

    await bot.send_message(
        chat_id=chat_id,
        text=report,
        reply_to_message_id=decision.target_message_id,
        allow_sending_without_reply=True
    )


@router.message()
async def handle_message(message: Message):

    if not bot_user:
        return

    try:

        await membu.submit(
            message=message,
            bot_user=bot_user,
            reporter=report_decision
        )

    except Exception:

        logger.exception(
            "MESSAGE ERROR | chat=%s",
            message.chat.id
        )


async def main():

    global bot_user, bot

    if not TOKEN:
        raise RuntimeError(
            "TOKEN environment variable is missing"
        )

    bot = Bot(token=TOKEN)

    dp = Dispatcher()

    dp.include_router(router)

    try:

        bot_user = await bot.get_me()

        logger.info(
            "MEMBU STARTED | @%s | id=%s",
            bot_user.username,
            bot_user.id
        )

        await bot.delete_webhook(
            drop_pending_updates=True
        )

        await dp.start_polling(
            bot,
            allowed_updates=dp.resolve_used_update_types()
        )

    finally:

        await membu.shutdown()

        await bot.session.close()

        logger.info("MEMBU STOPPED")


if __name__ == "__main__":

    asyncio.run(main())
