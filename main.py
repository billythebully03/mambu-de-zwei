import asyncio
import logging
import os
import re
from dataclasses import dataclass
from functools import partial
from typing import Awaitable, Callable

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ChatAction
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command
from aiogram.types import ChatMemberUpdated, Message, User

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

# Задержка на каждого пользователя отдельно: после его последнего
# сообщения Мембу ждёт тишины от этого же человека столько секунд.
DEBOUNCE_SECONDS = float(
    os.getenv("MEMBU_DEBOUNCE_SECONDS", "3")
)

# Если человек пишет без остановки, ответ не откладывается дольше этого срока.
MAX_DEBOUNCE_WAIT_SECONDS = float(
    os.getenv("MEMBU_MAX_DEBOUNCE_WAIT_SECONDS", "12")
)

# Очередь ответов одного чата. Ответы выполняются строго по очереди.
MAX_CHAT_QUEUE = int(os.getenv("MEMBU_MAX_CHAT_QUEUE", "30"))

# Сколько последних сообщений чата видит модель в каждом контексте.
# Без этого «вспомни, что обсуждали» работает только если модель сама вызовет поиск.
RECENT_MESSAGES_FOR_CONTEXT = int(os.getenv("MEMBU_RECENT_MESSAGES", "60"))

BRANCH_TTL = int(
    os.getenv("MEMBU_BRANCH_TTL_SECONDS", "300")
)

# Через сколько тишины Мембу получит возможность самому начать разговор.
IDLE_INITIATIVE_SECONDS = max(
    60,
    int(os.getenv("MEMBU_IDLE_SECONDS", "600")),
)

# Даже если чат продолжает молчать, повторно предлагать инициативу
# Gemini будет не чаще одного раза за этот промежуток.
IDLE_INITIATIVE_COOLDOWN_SECONDS = max(
    IDLE_INITIATIVE_SECONDS,
    int(os.getenv("MEMBU_IDLE_COOLDOWN_SECONDS", "1800")),
)

# Статус самого Мембу в каждом чате: in_chat или removed, кто и когда.
BOT_IN_CHAT_STATUSES = {"member", "administrator", "creator"}

# Telegram показывает typing максимум 5 секунд. Обновляем чаще этого срока.
TYPING_REFRESH_SECONDS = 4.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

router = Router()
membu = MembuCore()


@dataclass
class PendingTrigger:
    """Ожидающий запуска ответ на сообщения одного пользователя."""

    task: asyncio.Task | None
    started_at: float
    chat_id: int
    branch_id: str | None
    reason: str
    direct: bool


bot_user: User | None = None
storage: MembuStorage | None = None
agent: GeminiAgent | None = None
bot: Bot | None = None

# Ключ: (chat_id, user_id). У каждого пользователя свой таймер задержки.
pending_triggers: dict[tuple[int, int], PendingTrigger] = {}

# Статус Мембу в чатах, чтобы модель знала, что с ним происходило.
chat_bot_status: dict[int, dict] = {}

# Очередь и воркер на каждый чат: ответы идут по очереди, без отмены друг друга.
chat_queues: dict[int, asyncio.Queue] = {}
chat_workers: dict[int, asyncio.Task] = {}

# Мониторим тишину только в группах, где Мембу уже видел активность.
group_chat_ids: set[int] = set()
chat_last_activity: dict[int, float] = {}
chat_user_activity_version: dict[int, int] = {}
chat_idle_events: dict[int, asyncio.Event] = {}
chat_idle_tasks: dict[int, asyncio.Task] = {}
chat_last_idle_attempt: dict[int, float] = {}


def is_mentioned(message: Message, current_bot: User) -> bool:
    text = message.text or message.caption or ""

    for entity in (message.entities or []) + (message.caption_entities or []):
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
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def is_reply_to_bot(message: Message, current_bot: User) -> bool:
    replied = message.reply_to_message
    if not replied or not replied.from_user:
        return False
    return replied.from_user.id == current_bot.id


def mark_chat_activity(chat_id: int, *, human_activity: bool = True) -> None:
    """Обновляет таймер тишины. Вызов должен происходить внутри event loop."""
    if chat_id not in group_chat_ids:
        return

    loop = asyncio.get_running_loop()
    chat_last_activity[chat_id] = loop.time()

    if human_activity:
        chat_user_activity_version[chat_id] = (
            chat_user_activity_version.get(chat_id, 0) + 1
        )

    event = chat_idle_events.get(chat_id)
    if event is not None:
        event.set()

    task = chat_idle_tasks.get(chat_id)
    if task is None or task.done():
        chat_idle_tasks[chat_id] = asyncio.create_task(
            idle_initiative_monitor(chat_id)
        )


def register_group_activity(chat_id: int) -> None:
    group_chat_ids.add(chat_id)
    if chat_id not in chat_user_activity_version:
        chat_user_activity_version[chat_id] = 0
    if chat_id not in chat_last_activity:
        chat_last_activity[chat_id] = asyncio.get_running_loop().time()
    mark_chat_activity(chat_id, human_activity=True)


def enqueue_chat_job(
    chat_id: int,
    factory: Callable[[], Awaitable[None]],
) -> asyncio.Future:
    """Ставит ответ в очередь чата. Future завершается, когда ответ обработан."""
    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()

    queue = chat_queues.get(chat_id)
    if queue is None:
        queue = asyncio.Queue(maxsize=MAX_CHAT_QUEUE)
        chat_queues[chat_id] = queue

    worker = chat_workers.get(chat_id)
    if worker is None or worker.done():
        chat_workers[chat_id] = asyncio.create_task(
            chat_worker(chat_id, queue)
        )

    try:
        queue.put_nowait((factory, fut))
    except asyncio.QueueFull:
        logging.warning(
            "MEMBU | chat=%s | queue_full | job dropped", chat_id
        )
        fut.set_result(None)

    return fut


async def chat_worker(chat_id: int, queue: asyncio.Queue) -> None:
    """Обрабатывает ответы чата строго по одному."""
    while True:
        factory, fut = await queue.get()

        try:
            await factory()
            if not fut.done():
                fut.set_result(None)

        except asyncio.CancelledError:
            if not fut.done():
                fut.cancel()
            raise

        except Exception:
            logging.exception("Ошибка при обработке очереди чата %s", chat_id)
            if not fut.done():
                fut.set_result(None)

        finally:
            queue.task_done()


async def typing_indicator(chat_id: int) -> None:
    """Показывает штатный статус Telegram «печатает…», без отдельного сообщения."""
    if bot is None:
        return

    while True:
        try:
            await bot.send_chat_action(
                chat_id=chat_id,
                action=ChatAction.TYPING,
            )
        except TelegramAPIError as exc:
            logging.debug(
                "Не удалось обновить typing status в чате %s: %s",
                chat_id,
                exc,
            )
        await asyncio.sleep(TYPING_REFRESH_SECONDS)


async def run_agent_with_context(
    chat_id: int,
    branch_id: str | None,
    reason: str,
    direct: bool,
    *,
    is_stale: Callable[[], bool] | None = None,
) -> None:
    if storage is None or agent is None:
        return

    if is_stale and is_stale():
        logging.info(
            "MEMBU | chat=%s | initiative_cancelled=stale_context",
            chat_id,
        )
        return

    typing_task = asyncio.create_task(typing_indicator(chat_id))

    try:
        # После перезапуска статус берём из истории, а не ждём нового события.
        if chat_id not in chat_bot_status:
            chat_bot_status[chat_id] = await asyncio.to_thread(
                storage.last_bot_membership,
                chat_id,
            )

        recent = await asyncio.to_thread(
            storage.recent_messages,
            chat_id,
            RECENT_MESSAGES_FOR_CONTEXT,
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

        if is_stale and is_stale():
            logging.info(
                "MEMBU | chat=%s | initiative_cancelled=stale_context_before_gemini",
                chat_id,
            )
            return

        await agent.process(
            chat_id=chat_id,
            branch_id=branch_id,
            reason=reason,
            direct=direct,
            recent_messages=recent,
            branch_messages=branch_messages,
            active_branches=active_branches,
            is_stale=is_stale,
            bot_status=chat_bot_status.get(chat_id),
        )

    finally:
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass


async def fire_trigger(key: tuple[int, int], record: PendingTrigger) -> None:
    """Ждёт тишины от пользователя и ставит ответ в очередь чата."""
    await asyncio.sleep(DEBOUNCE_SECONDS)

    if pending_triggers.get(key) is record:
        pending_triggers.pop(key, None)

    enqueue_chat_job(
        record.chat_id,
        partial(
            run_agent_with_context,
            chat_id=record.chat_id,
            branch_id=record.branch_id,
            reason=record.reason,
            direct=record.direct,
        ),
    )


def schedule_agent(
    chat_id: int,
    branch_id: str | None,
    reason: str,
    direct: bool,
    sender_id: int,
) -> None:
    key = (chat_id, sender_id)
    now = asyncio.get_running_loop().time()

    old = pending_triggers.get(key)

    if old is not None and not old.task.done():
        # Непрерывный поток от одного человека не откладывает ответ вечно:
        # первый таймер уже идёт, новый сигнал только дополняет его.
        if now - old.started_at >= MAX_DEBOUNCE_WAIT_SECONDS:
            old.branch_id = branch_id or old.branch_id
            old.direct = old.direct or direct
            return

        old.task.cancel()

    record = PendingTrigger(
        task=None,  # задача создаётся ниже, она ссылается на саму запись
        started_at=now,
        chat_id=chat_id,
        branch_id=branch_id,
        reason=reason,
        direct=direct,
    )
    record.task = asyncio.create_task(fire_trigger(key, record))
    pending_triggers[key] = record


async def idle_initiative_monitor(chat_id: int) -> None:
    """Даёт Gemini шанс начать разговор после длительной тишины."""
    event = chat_idle_events.setdefault(chat_id, asyncio.Event())
    loop = asyncio.get_running_loop()

    try:
        while True:
            # Снимок версии позволяет отменить устаревшую инициативу,
            # если кто-то написал, пока Gemini обдумывает сообщение.
            event.clear()
            activity_version = chat_user_activity_version.get(chat_id, 0)
            last_activity = chat_last_activity.get(chat_id, loop.time())
            last_attempt = chat_last_idle_attempt.get(chat_id)

            next_allowed = (
                last_attempt + IDLE_INITIATIVE_COOLDOWN_SECONDS
                if last_attempt is not None
                else 0.0
            )
            wake_at = max(
                last_activity + IDLE_INITIATIVE_SECONDS,
                next_allowed,
            )
            timeout = max(0.0, wake_at - loop.time())

            try:
                await asyncio.wait_for(event.wait(), timeout=timeout)
                continue
            except asyncio.TimeoutError:
                pass

            if activity_version != chat_user_activity_version.get(chat_id, 0):
                continue

            if storage is None or agent is None:
                await asyncio.sleep(5)
                continue

            logging.info(
                "MEMBU | chat=%s | initiative=checking_after_silence | idle_seconds=%s",
                chat_id,
                int(loop.time() - chat_last_activity.get(chat_id, loop.time())),
            )

            fut = enqueue_chat_job(
                chat_id,
                partial(
                    run_agent_with_context,
                    chat_id=chat_id,
                    branch_id=None,
                    reason="long_silence_initiative",
                    direct=False,
                    is_stale=partial(
                        _activity_changed,
                        chat_id,
                        activity_version,
                    ),
                ),
            )

            try:
                await fut
            except asyncio.CancelledError:
                raise
            except Exception:
                logging.exception(
                    "Ошибка инициативы Мембу в чате %s",
                    chat_id,
                )

            # Не ставим кулдаун, если за время размышления кто-то написал:
            # следующую возможность можно будет оценить после новой тишины.
            if not _activity_changed(chat_id, activity_version):
                chat_last_idle_attempt[chat_id] = loop.time()

    except asyncio.CancelledError:
        raise
    finally:
        if chat_idle_tasks.get(chat_id) is asyncio.current_task():
            chat_idle_tasks.pop(chat_id, None)


def _activity_changed(chat_id: int, version: int) -> bool:
    return version != chat_user_activity_version.get(chat_id, 0)


@router.my_chat_member()
async def handle_bot_membership(update: ChatMemberUpdated) -> None:
    """Замечает, что Мембу добавили в чат или удалили из него."""
    if storage is None or bot_user is None:
        return

    if update.chat.type not in {"group", "supergroup"}:
        return

    chat_id = update.chat.id
    was_in = update.old_chat_member.status in BOT_IN_CHAT_STATUSES
    is_in = update.new_chat_member.status in BOT_IN_CHAT_STATUSES

    if was_in == is_in:
        return

    actor = update.from_user
    actor_name = format_member(actor) if actor else "неизвестно кто"
    changed_at = update.date.isoformat()

    if is_in:
        reason = "bot_added"
        text = f"[СОБЫТИЕ ЧАТА] Мембу добавили в чат; добавил: {actor_name}"
        chat_bot_status[chat_id] = {
            "status": "in_chat",
            "changed_at": changed_at,
            "by": actor_name,
        }
        register_group_activity(chat_id)
    else:
        reason = "bot_removed"
        self_left = actor is None or actor.id == bot_user.id
        who = "сам вышел" if self_left else f"удалил: {actor_name}"
        text = f"[СОБЫТИЕ ЧАТА] Мембу удалили из чата ({who})"
        chat_bot_status[chat_id] = {
            "status": "removed",
            "changed_at": changed_at,
            "by": actor_name,
        }

    # У этого события нет message_id. Берём отрицательный ключ из времени,
    # он не пересекается с обычными сообщениями Telegram (их ID положительные).
    synthetic_id = -int(update.date.timestamp() * 1000)

    await asyncio.to_thread(
        storage.add_message,
        {
            "chat_id": chat_id,
            "message_id": synthetic_id,
            "sender_id": actor.id if actor else None,
            "sender_name": actor.full_name if actor else "Telegram",
            "username": (actor.username or "") if actor else "",
            "text": text,
            "reply_to_user_id": None,
            "reply_to_message_id": None,
            "branch_id": None,
            "ts": update.date.timestamp(),
        },
    )

    logging.info("MEMBU | chat=%s | bot_membership=%s | %s", chat_id, reason, text)

    # Удалённого Мембу ответ не может достичь: писать ему некуда, поэтому ИИ
    # вызываем только при добавлении.
    if reason == "bot_added":
        schedule_agent(
            chat_id=chat_id,
            branch_id=None,
            reason=reason,
            direct=False,
            sender_id=actor.id if actor else 0,
        )


@router.message(Command("privacy"))
async def privacy_command(message: Message) -> None:
    if message.chat.type in {"group", "supergroup"}:
        register_group_activity(message.chat.id)

    await message.reply(
        "Я могу получать сообщения этого чата, если Telegram "
        "передаёт их мне. Контекст отдельных разговоров может "
        "отправляться в Google Gemini для анализа. История, долговременная "
        "память и профиль стиля хранятся в Supabase. "
        "Не отправляйте сюда пароли, документы и личные секреты."
    )


def format_member(user: User) -> str:
    name = user.full_name or user.first_name or "участник"
    if user.username:
        return f"{name} (@{user.username})"
    return name


@router.message(F.new_chat_members | F.left_chat_member)
async def handle_membership_event(message: Message) -> None:
    """Сохраняет события входа/выхода и передаёт их Мембу на оценку."""
    if storage is None or bot_user is None:
        return

    if message.chat.type not in {"group", "supergroup"}:
        return

    register_group_activity(message.chat.id)

    joined = [
        user
        for user in (message.new_chat_members or [])
        if user.id != bot_user.id
    ]
    left = message.left_chat_member
    if left and left.id == bot_user.id:
        left = None

    # Событие может сообщать только о добавлении самого Мембу.
    if not joined and left is None:
        logging.info(
            "MEMBU | chat=%s | membership_event=bot_join_or_leave",
            message.chat.id,
        )
        return

    descriptions: list[str] = []
    if joined:
        actor_id = message.from_user.id if message.from_user else None
        joined_descriptions = []
        for user in joined:
            if actor_id == user.id:
                joined_descriptions.append(
                    f"сам присоединился: {format_member(user)}"
                )
            else:
                joined_descriptions.append(
                    f"добавлен в чат: {format_member(user)}"
                )
        descriptions.extend(joined_descriptions)

    if left is not None:
        actor_id = message.from_user.id if message.from_user else None
        if actor_id == left.id:
            descriptions.append(
                f"сам покинул чат: {format_member(left)}"
            )
        else:
            descriptions.append(
                f"удалён из чата: {format_member(left)}"
            )

    event_text = "[СОБЫТИЕ ЧАТА] " + "; ".join(descriptions)
    actor = message.from_user

    await asyncio.to_thread(
        storage.add_message,
        {
            "chat_id": message.chat.id,
            "message_id": message.message_id,
            "sender_id": actor.id if actor else None,
            "sender_name": actor.full_name if actor else "Telegram",
            "username": actor.username if actor and actor.username else "",
            "text": event_text,
            "reply_to_user_id": None,
            "reply_to_message_id": None,
            "branch_id": None,
            "ts": message.date.timestamp(),
        },
    )

    reason = "member_joined" if joined and not left else "member_left"
    if joined and left:
        reason = "membership_changed"

    logging.info(
        "MEMBU | chat=%s | membership_event=%s | %s",
        message.chat.id,
        reason,
        event_text,
    )

    schedule_agent(
        chat_id=message.chat.id,
        branch_id=None,
        reason=reason,
        direct=False,
        sender_id=actor.id if actor else 0,
    )


@router.message()
async def handle_message(message: Message) -> None:
    global bot_user, storage

    if bot_user is None or storage is None:
        return

    if message.from_user is None:
        return

    # Сообщения самого Мембу и других ботов не запускают новую обработку.
    if message.from_user.is_bot:
        return

    chat_id = message.chat.id
    if message.chat.type in {"group", "supergroup"}:
        # Считаем активностью и фото/стикеры/голосовые, даже если их
        # текстовая подпись пуста и текущая AI-логика их не анализирует.
        register_group_activity(chat_id)

    text = message.text or message.caption or ""
    if not text.strip():
        return

    mentioned = is_mentioned(message, bot_user)
    replied_to_bot = is_reply_to_bot(message, bot_user)
    private_chat = message.chat.type == "private"

    reply_to_message_id = None
    reply_to_user_id = None
    reply_to_branch_id = None

    if message.reply_to_message:
        reply_to_message_id = message.reply_to_message.message_id

        if message.reply_to_message.from_user:
            reply_to_user_id = message.reply_to_message.from_user.id

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

    event: dict = {
        "chat_id": chat_id,
        "message_id": message.message_id,
        "sender_id": message.from_user.id,
        "sender_name": message.from_user.full_name or "",
        "username": message.from_user.username or "",
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
        sender_id=message.from_user.id,
    )


async def main() -> None:
    global bot_user, storage, agent, bot

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
            activity_callback=lambda chat_id: mark_chat_activity(
                chat_id,
                human_activity=False,
            ),
        )

        dp = Dispatcher()
        dp.include_router(router)

        logging.info(
            "Мембу запущен: @%s | model=%s | debounce=%ss | idle=%ss | idle_cooldown=%ss",
            bot_user.username,
            GEMINI_MODEL,
            DEBOUNCE_SECONDS,
            IDLE_INITIATIVE_SECONDS,
            IDLE_INITIATIVE_COOLDOWN_SECONDS,
        )

        # Не сбрасываем сообщения, пришедшие во время простоя или деплоя.
        await bot.delete_webhook(
            drop_pending_updates=False,
        )

        await dp.start_polling(
            bot,
            allowed_updates=dp.resolve_used_update_types(),
        )

    finally:
        all_tasks: list[asyncio.Task] = []

        for record in pending_triggers.values():
            all_tasks.append(record.task)

        all_tasks.extend(chat_workers.values())
        all_tasks.extend(chat_idle_tasks.values())

        for task in all_tasks:
            task.cancel()

        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)

        if bot is not None:
            await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
