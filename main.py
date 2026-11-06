import asyncio
import logging
import os

from aiogram import Bot, Dispatcher, Router
from aiogram.types import Message, User

from mambu import MembuCore


TOKEN = os.getenv("TOKEN")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

router = Router()

membu = MembuCore()
bot_user: User | None = None


@router.message()
async def handle_message(message: Message):
    if bot_user is None:
        return

    if message.from_user and message.from_user.is_bot:
        return

    decision = membu.process(message, bot_user)

    username = (
        message.from_user.username
        or message.from_user.first_name
        if message.from_user
        else "Unknown"
    )

    logging.info(
        "MEMBU | chat=%s | user=%s | action=%s | score=%s | reason=%s",
        message.chat.id,
        username,
        decision.action,
        decision.score,
        decision.reason,
    )

    icons = {
        "reply": "🟢",
        "wait": "🟡",
        "ignore": "⚫",
    }

    diagnostic = (
        f"{icons[decision.action]} MEMBU DECISION\n\n"
        f"Действие: {decision.action.upper()}\n"
        f"Оценка: {decision.score}\n"
        f"Причина: {decision.reason}"
    )

    await message.answer(diagnostic)


async def main():
    global bot_user

    if not TOKEN:
        raise RuntimeError("TOKEN environment variable is missing")

    bot = Bot(token=TOKEN)
    bot_user = await bot.get_me()

    dp = Dispatcher()
    dp.include_router(router)

    logging.info("Membu started as @%s", bot_user.username)

    await bot.delete_webhook(drop_pending_updates=True)

    try:
        await dp.start_polling(
            bot,
            allowed_updates=dp.resolve_used_update_types(),
        )
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
