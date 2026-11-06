import asyncio
import logging
import os
import re
from typing import Any

from aiogram import Bot, Dispatcher, Router
from aiogram.filters import Command
from aiogram.types import Message, User
from aiogram.exceptions import TelegramAPIError

from membu import MembuCore
from storage import MembuStorage
from gemini_agent import GeminiAgent


TOKEN = os.getenv("TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite",
)

DEBOUNCE_SECONDS = float(
    os.getenv("MEMBU_DEBOUNCE_SECONDS", "2.4")
)

BRANCH_TTL = int(
    os.getenv("MEMBU_BRANCH_TTL_SECONDS", "300")
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

router = Router()

membu = MembuCore()

bot_user: User | None = None
storage: MembuStorage | None = None
agent: GeminiAgent | None = None

scheduled_tasks: dict[tuple[int, str], asyncio.Task] = {}


def is_mentioned(message: Message, current_bot: User) -> bool:
    text = message.text or message.caption or ""

    for entity in (
        message.entities or []
    ) + (
        message.caption_entities or []
    ):
        if (
            entity.type == "text_mention"
            and entity.user
            and entity.user.id == current_bot.id
        ):
            return True

    username = current_bot.username

    if not username:
        return False

    pattern = rf"(?<![\w])@{re.escape(username)}(?!\w)"

    return bool(
        re.search(pattern, text, flags=re.IGNORECASE)
    )


def is_reply_to_bot(message: Message, current_bot: User) -> bool:
    replied = message.reply_to_message

    if not replied or not replied.from_user:
        return False

    return replied.from_user.id == current_bot.id


async def process_after_pause(
    chat_id: int,
    branch_id: str | None,
    reason: str,
    direct: bool,
    wait_seconds: float,
    task_key: tuple[int, str],
):
    current_task = asyncio.current_task()

    try:
        await asyncio.sleep(
            DEBOUNCE_SECONDS + max(0, wait_seconds)
        )

        if storage is None or agent is None:
            return

        recent = await asyncio.to_thread(
            storage.recent_messages,
            chat_id,
            18,
        )

        branch_messages = []

        if branch_id:
            branch_messages = await asyncio.to_thread(
                storage.branch_messages,
                chat_id,
                branch_id,
                25,
            )

        active_branches = await asyncio.to_thread(
            storage.active_branches,
            chat_id,
            BRANCH_TTL,
        )

        # Каждой активной ветке передаём собственный контекст.
        for branch in active_branches:
            branch["messages"] = await asyncio.to_thread(
                storage.branch_messages,
                chat_id,
                branch["branch_id"],
                12,
            )

        await agent.process(
            chat_id=chat_id,
            branch_id=branch_id,
            reason=reason,
            direct=direct,
            recent_messages=recent,
            branch_messages=branch_messages,
            active_branches=active_branches,
        )

    except asyncio.CancelledError:
        raise

    except Exception:
        logging.exception(
            "Ошибка при обработке контекста чата %s",
            chat_id,
        )

    finally:
        if scheduled_tasks.get(task_key) is current_task:
            scheduled_tasks.pop(task_key, None)


def schedule_agent(
    chat_id: int,
    branch_id: str | None,
    reason: str,
    direct: bool,
    wait_seconds: float,
):
    key = (
        chat_id,
        branch_id or "__general__",
    )

    old_task = scheduled_tasks.get(key)

    if old_task and not old_task.done():
        old_task.cancel()

    task = asyncio.create_task(
        process_after_pause(
            chat_id=chat_id,
            branch_id=branch_id,
            reason=reason,
            direct=direct,
            wait_seconds=wait_seconds,
            task_key=key,
        )
    )

    scheduled_tasks[key] = task


@router.message(Command("privacy"))
async def privacy_command(message: Message):
    await message.reply(
        "Я могу получать сообщения этого чата, если Telegram "
        "передаёт их мне. Контекст отдельных разговоров может "
        "отправляться в Google Gemini для анализа. История и "
        "выбранные воспоминания хранятся в Supabase. "
        "Не отправляйте сюда пароли, документы и личные секреты."
    )


@router.message()
async def handle_message(message: Message):
    global bot_user, storage

    if bot_user is None or storage is None:
        return

    if message.from_user is None:
        return

    # Не обрабатываем сообщения самого Мембу и других ботов.
    if message.from_user.is_bot:
        return

    text = message.text or message.caption or ""

    if not text.strip():
        return

    chat_id = message.chat.id

    mentioned = is_mentioned(message, bot_user)
    replied_to_bot = is_reply_to_bot(message, bot_user)

    # В личной переписке каждое сообщение адресовано боту.
    private_chat = message.chat.type == "private"

    direct = (
        mentioned
        or replied_to_bot
        or private_chat
    )

    reply_to_message_id = None
    reply_to_user_id = None
    reply_to_branch_id = None

    if message.reply_to_message:
        reply_to_message_id = (
            message.reply_to_message.message_id
        )

        if message.reply_to_message.from_user:
            reply_to_user_id = (
                message.reply_to_message.from_user.id
            )

        reply_to_branch_id = await asyncio.to_thread(
            storage.get_branch_for_message,
            chat_id,
            reply_to_message_id,
        )

    active_branches = await asyncio.to_thread(
        storage.active_branches,
        chat_id,
        BRANCH_TTL,
    )

    event: dict[str, Any] = {
        "chat_id": chat_id,
        "message_id": message.message_id,
        "sender_id": message.from_user.id,
        "sender_name": (
            message.from_user.full_name or ""
        ),
        "username": (
            message.from_user.username or ""
        ),
        "text": text[:3000],
        "reply_to_user_id": reply_to_user_id,
        "reply_to_message_id": reply_to_message_id,
        "reply_to_branch_id": reply_to_branch_id,
        "mentioned": mentioned,
        "replied_to_bot": replied_to_bot,
        "private_chat": private_chat,
        "ts": message.date.timestamp(),
    }

    decision = membu.inspect(
        event=event,
        active_branches=active_branches,
    )

    event["branch_id"] = decision.branch_id

    # Если началась новая ветка, сохраняем её отдельно.
    if decision.direct and decision.branch_id:
        existing_ids = {
            branch["branch_id"]
            for branch in active_branches
        }

        if decision.branch_id not in existing_ids:
            await asyncio.to_thread(
                storage.create_branch,
                chat_id,
                decision.branch_id,
                message.message_id,
                message.from_user.id,
            )

    await asyncio.to_thread(
        storage.add_message,
        event,
    )

    if decision.branch_id:
        await asyncio.to_thread(
            storage.touch_branch,
            chat_id,
            decision.branch_id,
        )

    logging.info(
        "CHAT=%s USER=%s DIRECT=%s AI=%s REASON=%s BRANCH=%s",
        chat_id,
        message.from_user.id,
        decision.direct,
        decision.call_ai,
        decision.reason,
        decision.branch_id,
    )

    if not decision.call_ai:
        return

    schedule_agent(
        chat_id=chat_id,
        branch_id=decision.branch_id,
        reason=decision.reason,
        direct=decision.direct,
        wait_seconds=decision.wait_seconds,
    )


async def main():
    global bot_user, storage, agent

    required = {
        "TOKEN": TOKEN,
        "GEMINI_API_KEY": GEMINI_API_KEY,
        "SUPABASE_URL": SUPABASE_URL,
        "SUPABASE_SERVICE_ROLE_KEY": SUPABASE_SERVICE_ROLE_KEY,
    }

    missing = [
        name
        for name, value in required.items()
        if not value
    ]

    if missing:
        raise RuntimeError(
            "Не заданы переменные окружения: "
            + ", ".join(missing)
        )

    bot = Bot(token=TOKEN)

    try:
        bot_user = await bot.get_me()

        storage = MembuStorage(
            url=SUPABASE_URL,
            key=SUPABASE_SERVICE_ROLE_KEY,
        )

        agent = GeminiAgent(
            api_key=GEMINI_API_KEY,
            storage=storage,
            bot=bot,
            bot_user=bot_user,
            model=GEMINI_MODEL,
        )

        dp = Dispatcher()
        dp.include_router(router)

        logging.info(
            "Мембу запущен: @%s | model=%s",
            bot_user.username,
            GEMINI_MODEL,
        )

        await bot.delete_webhook(
            drop_pending_updates=True
        )

        await dp.start_polling(
            bot,
            allowed_updates=dp.resolve_used_update_types(),
        )

    finally:
        for task in scheduled_tasks.values():
            task.cancel()

        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
