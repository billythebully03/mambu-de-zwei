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
    decision = membu.process(message, bot_user)

    if not decision.accepted:
        return

    if decision.reply_text:
        await message.answer(decision.reply_text)
        return

    logging.info(
        "MEMBU EVENT | chat_id=%s | user_id=%s | reason=%s",
        message.chat.id,
        message.from_user.id if message.from_user else None,
        decision.reason
    )


async def main():
    global bot_user

    if not TOKEN:
        raise RuntimeError("TOKEN environment variable is missing")

    bot = Bot(token=TOKEN)
    bot_user = await bot.get_me()

    dp = Dispatcher()
    dp.include_router(router)

    logging.info("Membu started as @%s", bot_user.username)

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
