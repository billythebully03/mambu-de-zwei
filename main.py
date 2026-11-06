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

    logging.info(
        "MEMBU | chat=%s | user=%s | decision=%s | reason=%s",
        message.chat.id,
        message.from_user.id if message.from_user else None,
        decision.action,
        decision.reason
    )

    if message.chat.type == "private":
        await message.answer(decision.description)
        return

    if decision.accepted:
        await message.answer(decision.description)


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

    await dp.start_polling(
        bot,
        allowed_updates=dp.resolve_used_update_types()
    )


if __name__ == "__main__":
    asyncio.run(main())
