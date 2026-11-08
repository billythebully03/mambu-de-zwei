import asyncio
import logging
import os
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import Awaitable, Callable

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ChatAction
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    ChatMemberUpdated,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    User,
)

from membu import MembuCore
from storage import MembuStorage
from gemini_agent import GeminiAgent
from web_server import start_web_server

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
RECENT_MESSAGES_FOR_CONTEXT = int(os.getenv("MEMBU_RECENT_MESSAGES", "45"))

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

# Команды управления доступны только в личке и только этому пользователю.
OWNER_USERNAMES = {
    name.strip().lstrip("@").lower()
    for name in os.getenv("MEMBU_OWNER_USERNAMES", "ibartyom,Ribapibaa").split(",")
    if name.strip()
}
MEMORY_PAGE_SIZE = 8
DUBAI_TZ = timezone(timedelta(hours=4))
STARTED_AT = datetime.now(timezone.utc)

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


GREETING_MORNING = re.compile(
    r"^\W*(доброе утро|доброго утра|доброе утречко|доброго утречка|с добрым утром|"
    r"утречка|утречко|утра|утро|good morning)(\W+\w{1,10})?\W*$",
    re.IGNORECASE,
)
GREETING_NIGHT = re.compile(
    r"^\W*(спокойной ночи|спокойной ночки|спокойной ночь|сладких снов|доброй ночи|"
    r"доброй ночки|приятных снов|спокойного сна|спать пора|ночи|good night)"
    r"(\W+\w{1,10})?\W*$",
    re.IGNORECASE,
)
_greeting_marks: set[tuple] = set()


def greeting_moment(text: str, chat_id: int) -> bool:
    """Утреннее (7–9) или ночное (21–23) пожелание по Дубаю. Один раз на чат в день
    и окно: тогда Мембу решает, подключаться ли."""
    body = (text or "").strip()
    if not body or len(body) > 60:
        return False

    now = datetime.now(timezone(timedelta(hours=4)))

    if 7 <= now.hour < 9:
        window, pattern = "morning", GREETING_MORNING
    elif 21 <= now.hour < 23:
        window, pattern = "night", GREETING_NIGHT
    else:
        return False

    if not pattern.search(body):
        return False

    key = (chat_id, now.date().isoformat(), window)
    if key in _greeting_marks:
        return False

    _greeting_marks.add(key)
    return True


def is_mentioned(message: Message, current_bot: User) -> bool:
    text = message.text or message.caption or ""

    for entity in (message.entities or []) + (message.caption_entities or []):
        if (
            entity.type == "text_mention"
            and entity.user
            and entity.user.id == current_bot.id
        ):
            return True

    # Имя в тексте тоже считается обращением: запрос уходит в ИИ, а он сам решает,
    # ответить, поставить реакцию или промолчать.
    name_pattern = r"(?<!\w)(мемб|мэмб|memb)\w*"
    if re.search(name_pattern, text, flags=re.IGNORECASE):
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

    # Инициатива в тишине отключена: Мембу сам пишет только утром и вечером (daily_window_monitor).


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

        except Exception as exc:
            if agent is not None:
                agent.note_api_error(exc)
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


def is_owner_dm(user: User | None, chat_type: str) -> bool:
    return bool(
        user
        and user.username
        and chat_type == "private"
        and user.username.lower() in OWNER_USERNAMES
    )


def format_dubai(dt: datetime) -> str:
    return dt.astimezone(DUBAI_TZ).strftime("%d.%m %H:%M")


def memory_label(row: dict) -> str:
    content = str(row.get("content", "")).replace("\n", " ")
    people = row.get("people") or []
    label = f"{', '.join(people)}: {content}" if people else content
    return label if len(label) <= 60 else label[:59] + "…"


async def memory_page_view(page: int) -> tuple[str, InlineKeyboardMarkup]:
    rows = await asyncio.to_thread(storage.list_all_memories, 500)

    if not rows:
        return (
            "Настройка памяти\n\nМембу пока ничего не запомнил.",
            InlineKeyboardMarkup(inline_keyboard=[tabs_row("mem")]),
        )

    total_pages = math.ceil(len(rows) / MEMORY_PAGE_SIZE)
    page = max(0, min(page, total_pages - 1))
    chunk = rows[page * MEMORY_PAGE_SIZE:(page + 1) * MEMORY_PAGE_SIZE]

    keyboard = [
        [
            InlineKeyboardButton(
                text=memory_label(row),
                callback_data=f"mem:view:{row['id']}:{page}",
            )
        ]
        for row in chunk
    ]

    keyboard.insert(0, tabs_row("mem"))
    keyboard.append([
        InlineKeyboardButton(
            text="◀",
            callback_data=f"mem:page:{page - 1}" if page > 0 else "mem:noop",
        ),
        InlineKeyboardButton(
            text=f"{page + 1}/{total_pages}",
            callback_data="mem:noop",
        ),
        InlineKeyboardButton(
            text="▶",
            callback_data=(
                f"mem:page:{page + 1}"
                if page < total_pages - 1
                else "mem:noop"
            ),
        ),
    ])

    text = (
        "Настройка памяти · Основные\n\n"
        "Правила и выводы, которые запомнил Мембу. "
        "Нажми на запись, чтобы открыть её полностью, удалить или вернуться назад. "
        f"Всего: {len(rows)}."
    )
    return text, InlineKeyboardMarkup(inline_keyboard=keyboard)


def parse_db_time(value) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


async def memory_detail_view(
    memory_id: int,
    page: int,
) -> tuple[str, InlineKeyboardMarkup] | None:
    row = await asyncio.to_thread(storage.get_memory, memory_id)

    if row is None:
        return None

    people = ", ".join(row.get("people") or []) or "—"
    updated = parse_db_time(row.get("updated_at"))
    updated_text = format_dubai(updated) if updated else "—"

    text = (
        f"Воспоминание #{row['id']}\n"
        f"Категория: {row.get('category')}\n"
        f"Важность: {row.get('importance')}/5\n"
        f"Люди: {people}\n"
        f"Чат: {row.get('chat_id')}\n"
        f"Обновлено: {updated_text} (Дубай)\n\n"
        f"{row.get('content', '')}"
    )

    markup = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Удалить",
                    callback_data=f"mem:del:{row['id']}:{page}",
                ),
                InlineKeyboardButton(
                    text="Назад",
                    callback_data=f"mem:page:{page}",
                ),
            ]
        ]
    )
    return text, markup


async def build_status_text() -> str:
    now = datetime.now(timezone.utc)
    lines = ["Статус Мембу"]

    lines.append(f"Модель: {GEMINI_MODEL}")
    lines.append(f"Запущен: {format_dubai(STARTED_AT)} (Дубай)")

    last_ok = agent.last_ok_at if agent else None
    lines.append(
        "Последний успешный ответ модели: "
        + (format_dubai(last_ok) if last_ok else "пока не было")
    )

    if agent and agent.using_backup:
        since = format_dubai(agent.switched_at) if agent.switched_at else "—"
        lines.append(
            f"Ключ: работает РЕЗЕРВНЫЙ ключ с {since} (Дубай). "
            "Основной вернётся сам примерно через 30 минут, если начнёт отвечать."
        )
    else:
        lines.append("Ключ: основной.")

    if agent and agent.quota_reset_at and agent.quota_reset_at > now:
        lines.append(
            "Лимит токенов исчерпан. Сбросится примерно: "
            f"{format_dubai(agent.quota_reset_at)} (Дубай, UTC+4)."
        )
    else:
        lines.append("Лимит токенов: переполнения сейчас нет.")

    if agent and agent.last_error:
        err = agent.last_error
        lines.append(
            f"Последняя ошибка ({format_dubai(err['at'])}): {err['message'][:200]}"
        )
    else:
        lines.append("Последних ошибок нет.")

    count = 0
    if storage is not None:
        count = len(await asyncio.to_thread(storage.list_all_memories, 1000))
    lines.append(f"Воспоминаний: {count}")

    return "\n".join(lines)


@router.message(Command("memory"))
async def memory_command(message: Message) -> None:
    if not is_owner_dm(message.from_user, message.chat.type) or storage is None:
        return

    text, markup = await memory_page_view(0)
    await message.answer(text, reply_markup=markup)


@router.message(Command("clearall"))
async def clearall_command(message: Message) -> None:
    if not is_owner_dm(message.from_user, message.chat.type):
        return

    markup = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Да, стереть всё", callback_data="clr:yes"),
                InlineKeyboardButton(text="Отмена", callback_data="clr:no"),
            ]
        ]
    )
    await message.answer(
        "Ты точно хочешь стереть воспоминания Мембу? Будет стёрто: обычные воспоминания "
        "(важность 1–3), вся история сообщений, сессионная память ИИ. Важные воспоминания "
        "(важность 4–5) и личность Мембу останутся. Это нельзя будет отменить.",
        reply_markup=markup,
    )


async def key_view() -> tuple[str, InlineKeyboardMarkup]:
    results = await agent.probe_keys()
    active = agent.active_key_index
    lines = ["Ключи Gemini"]
    keyboard = []

    for index, info in enumerate(results):
        if info["ok"]:
            mark = "🟢 без лимита"
        elif info["error"] == "лимит":
            if info["reset_at"]:
                mark = f"🔴 лимит, обновится {format_dubai(info['reset_at'])} (Дубай)"
            else:
                mark = "🔴 лимит"
        else:
            mark = f"🔴 ошибка: {info['error']}"

        flag = " · активный" if index == active else ""
        lines.append(f"{info['label']}: {mark}{flag}")
        keyboard.append([
            InlineKeyboardButton(
                text=f"Переключить на {info['label']}",
                callback_data=f"key:use:{index}",
            )
        ])

    keyboard.append([InlineKeyboardButton(text="Проверить снова", callback_data="key:check")])
    lines.append("")
    lines.append("При лимите ключи переключаются сами. Ручное переключение держится, пока не упрётся в лимит.")
    return "\n".join(lines), InlineKeyboardMarkup(inline_keyboard=keyboard)


@router.message(Command("key"))
async def key_command(message: Message) -> None:
    if not is_owner_dm(message.from_user, message.chat.type) or agent is None:
        return

    text, markup = await key_view()
    await message.answer(text, reply_markup=markup)


async def style_list_view() -> tuple[str, InlineKeyboardMarkup]:
    rows = await asyncio.to_thread(storage.list_style_profiles, 20)

    if not rows:
        return (
            "Выученной модели общения пока нет. Она появится, когда в чатах накопится переписка.",
            InlineKeyboardMarkup(inline_keyboard=[]),
        )

    keyboard = []
    for row in rows:
        label = short_label(
            f"чат {row.get('chat_id')}: ",
            f"{row.get('observed_messages') or 0} сообщений",
        )
        keyboard.append([
            InlineKeyboardButton(text=label, callback_data=f"sty:view:{row['chat_id']}")
        ])

    text = "Модель общения по чатам. Нажми на чат, чтобы увидеть профиль и обновить его сейчас."
    return text, InlineKeyboardMarkup(inline_keyboard=keyboard)


async def style_chat_view(chat_id: int) -> tuple[str, InlineKeyboardMarkup]:
    row = await asyncio.to_thread(storage.get_chat_style, chat_id)

    updated = parse_db_time(row.get("updated_at"))
    updated_text = format_dubai(updated) if updated else "—"
    profile = (row.get("style_profile") or "").strip() or "пока пусто"

    text = (
        f"Модель общения, чат {chat_id}\n"
        f"Сообщений учтено: {row.get('observed_messages') or 0}\n"
        f"Обновлено: {updated_text} (Дубай)\n\n"
        f"{profile[:3500]}"
    )

    markup = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Обновить сейчас", callback_data=f"sty:refresh:{chat_id}")],
            [InlineKeyboardButton(text="Назад", callback_data="sty:list")],
        ]
    )
    return text, markup


@router.message(Command("style"))
async def style_command(message: Message) -> None:
    if not is_owner_dm(message.from_user, message.chat.type) or storage is None:
        return

    text, markup = await style_list_view()
    await message.answer(text, reply_markup=markup)


@router.message(Command("reset"))
async def reset_command(message: Message) -> None:
    if not is_owner_dm(message.from_user, message.chat.type):
        return

    markup = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Да, перезапустить", callback_data="rst:yes"),
                InlineKeyboardButton(text="Отмена", callback_data="rst:no"),
            ]
        ]
    )
    await message.answer(
        "Полный перезапуск Мембу с нуля. Будет стёрто: обычные воспоминания (важность 1–3), "
        "вся история сообщений, сессионная память ИИ, выученный стиль чата и мнения о людях. "
        "Важные воспоминания (важность 4–5) и личность Мембу останутся. Это нельзя будет отменить.",
        reply_markup=markup,
    )


@router.message(Command("status"))
async def status_command(message: Message) -> None:
    if not is_owner_dm(message.from_user, message.chat.type):
        return

    await message.answer(await build_status_text())


def tabs_row(active: str) -> list[InlineKeyboardButton]:
    def label(name: str, text: str) -> str:
        return ("• " if active == name else "") + text

    return [
        InlineKeyboardButton(text=label("mem", "Основные"), callback_data="tab:mem"),
        InlineKeyboardButton(text=label("msg", "Все сообщения"), callback_data="tab:msg"),
    ]


def nav_row(prefix: str, page: int, total_pages: int) -> list[InlineKeyboardButton]:
    return [
        InlineKeyboardButton(
            text="◀",
            callback_data=f"{prefix}:page:{page - 1}" if page > 0 else f"{prefix}:noop",
        ),
        InlineKeyboardButton(text=f"{page + 1}/{total_pages}", callback_data=f"{prefix}:noop"),
        InlineKeyboardButton(
            text="▶",
            callback_data=(
                f"{prefix}:page:{page + 1}" if page < total_pages - 1 else f"{prefix}:noop"
            ),
        ),
    ]


def short_label(prefix: str, text: str) -> str:
    label = f"{prefix}{text}".replace("\n", " ")
    return label if len(label) <= 60 else label[:59] + "…"


async def message_page_view(page: int) -> tuple[str, InlineKeyboardMarkup]:
    rows, total = await asyncio.to_thread(
        storage.list_messages_page, max(0, page), MEMORY_PAGE_SIZE
    )
    if total and not rows:
        page = math.ceil(total / MEMORY_PAGE_SIZE) - 1
        rows, total = await asyncio.to_thread(
            storage.list_messages_page, page, MEMORY_PAGE_SIZE
        )

    total_pages = max(1, math.ceil(total / MEMORY_PAGE_SIZE))
    page = max(0, min(page, total_pages - 1))

    keyboard = [tabs_row("msg")]
    for row in rows:
        label = short_label(
            f"{row.get('sender_name') or 'участник'}: ",
            str(row.get("text") or ""),
        )
        keyboard.append([
            InlineKeyboardButton(
                text=label,
                callback_data=f"msg:view:{row['chat_id']}:{row['message_id']}:{page}",
            )
        ])

    if total:
        keyboard.append(nav_row("msg", page, total_pages))
        text = (
            "Настройка памяти · Все сообщения\n\n"
            "Все сообщения из базы, от новых к старым. Нажми на сообщение, чтобы открыть "
            f"его полностью или удалить. Всего: {total}."
        )
    else:
        text = "Настройка памяти · Все сообщения\n\nВ базе пока нет сообщений."

    return text, InlineKeyboardMarkup(inline_keyboard=keyboard)


async def message_detail_view(
    chat_id: int,
    message_id: int,
    page: int,
) -> tuple[str, InlineKeyboardMarkup] | None:
    row = await asyncio.to_thread(storage.get_message, chat_id, message_id)
    if row is None:
        return None

    when = parse_db_time(row.get("created_at"))
    when_text = format_dubai(when) if when else "—"

    text = (
        f"Сообщение #{row.get('message_id')}\n"
        f"От: {row.get('sender_name') or 'участник'}\n"
        f"Чат: {row.get('chat_id')}\n"
        f"Время: {when_text} (Дубай)\n\n"
        f"{str(row.get('text') or '')[:3500]}"
    )

    markup = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Удалить",
                    callback_data=f"msg:del:{chat_id}:{message_id}:{page}",
                ),
                InlineKeyboardButton(text="Назад", callback_data=f"msg:page:{page}"),
            ]
        ]
    )
    return text, markup


async def people_page_view(page: int) -> tuple[str, InlineKeyboardMarkup]:
    rows, total = await asyncio.to_thread(
        storage.list_people_page, max(0, page), MEMORY_PAGE_SIZE
    )
    if total == 0:
        return (
            "Мнений о людях пока нет. Мембу формирует их по ходу общения.",
            InlineKeyboardMarkup(inline_keyboard=[]),
        )

    total_pages = math.ceil(total / MEMORY_PAGE_SIZE)
    page = max(0, min(page, total_pages - 1))
    if not rows:
        rows, total = await asyncio.to_thread(
            storage.list_people_page, page, MEMORY_PAGE_SIZE
        )

    keyboard = []
    for row in rows:
        label = short_label(
            f"{row.get('display_name') or 'участник'}: ",
            str(row.get("opinion") or ""),
        )
        keyboard.append([
            InlineKeyboardButton(
                text=label,
                callback_data=f"ppl:view:{row['chat_id']}:{row['person_id']}:{page}",
            )
        ])
    keyboard.append(nav_row("ppl", page, total_pages))

    text = (
        "Мнение Мембу о людях\n\n"
        "Нажми на человека, чтобы прочитать мнение целиком. "
        f"Всего: {total}."
    )
    return text, InlineKeyboardMarkup(inline_keyboard=keyboard)


async def person_detail_view(
    chat_id: int,
    person_id: int,
    page: int,
) -> tuple[str, InlineKeyboardMarkup] | None:
    row = await asyncio.to_thread(storage.get_person, chat_id, person_id)
    if row is None:
        return None

    updated = parse_db_time(row.get("updated_at"))
    updated_text = format_dubai(updated) if updated else "—"

    text = (
        f"{row.get('display_name') or 'участник'}\n"
        f"Чат: {row.get('chat_id')}\n"
        f"Сообщений в недавней переписке: {row.get('message_count')}\n"
        f"Обновлено: {updated_text} (Дубай)\n\n"
        "Мнение Мембу:\n"
        f"{row.get('opinion') or '—'}"
    )

    markup = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Назад", callback_data=f"ppl:page:{page}")]
        ]
    )
    return text, markup


@router.message(Command("people"))
async def people_command(message: Message) -> None:
    if not is_owner_dm(message.from_user, message.chat.type) or storage is None:
        return

    text, markup = await people_page_view(0)
    await message.answer(text, reply_markup=markup)


@router.callback_query(
    F.data.startswith(("mem:", "msg:", "ppl:", "tab:", "clr:", "rst:", "sty:", "key:"))
)
async def handle_owner_callbacks(callback: CallbackQuery) -> None:
    message = callback.message

    if message is None or not is_owner_dm(callback.from_user, message.chat.type):
        await callback.answer()
        return

    parts = (callback.data or "").split(":")

    try:
        kind = parts[0]

        if kind == "tab":
            if parts[1] == "msg":
                text, markup = await message_page_view(0)
            else:
                text, markup = await memory_page_view(0)
            await message.edit_text(text, reply_markup=markup)
            await callback.answer()
            return

        if kind == "key":
            action = parts[1]

            if action == "use":
                agent.set_active_key(int(parts[2]))
                await callback.answer("Переключено")

            text, markup = await key_view()
            await message.edit_text(text, reply_markup=markup)
            if action != "use":
                await callback.answer()
            return

        if kind == "sty":
            action = parts[1]

            if action == "list":
                text, markup = await style_list_view()
                await message.edit_text(text, reply_markup=markup)
                await callback.answer()
                return

            if action == "view":
                text, markup = await style_chat_view(int(parts[2]))
                await message.edit_text(text, reply_markup=markup)
                STYLE_VIEW_MESSAGES[message.message_id] = int(parts[2])
                await callback.answer()
                return

            if action == "refresh":
                chat_id = int(parts[2])
                await callback.answer("Обновляю…")
                await agent.force_style_refresh(chat_id)
                text, markup = await style_chat_view(chat_id)
                await message.edit_text(text, reply_markup=markup)
                STYLE_VIEW_MESSAGES[message.message_id] = chat_id
                return

            await callback.answer()
            return

        if kind == "rst":
            if parts[1] == "yes":
                memories = await asyncio.to_thread(storage.delete_unimportant_memories, 4)
                messages = await asyncio.to_thread(storage.delete_all_messages)
                branches = await asyncio.to_thread(storage.close_all_branches)
                people = await asyncio.to_thread(storage.delete_all_people)
                await asyncio.to_thread(storage.reset_chat_styles)
                await asyncio.to_thread(storage.reset_digest_counts)
                agent.reset_learned_state()
                await message.edit_text(
                    "Перезапуск выполнен.\n"
                    f"Обычных воспоминаний стёрто: {memories}\n"
                    f"Сообщений стёрто: {messages}\n"
                    f"Веток разговора закрыто: {branches}\n"
                    f"Мнений о людях стёрто: {people}\n\n"
                    "Стиль чата сброшен. Важные воспоминания (4–5) и личность Мембу на месте."
                )
            else:
                await message.edit_text("Отменено. Ничего не сброшено.")
            await callback.answer()
            return

        if kind == "clr":
            if parts[1] == "yes":
                memories = await asyncio.to_thread(storage.delete_unimportant_memories, 4)
                messages = await asyncio.to_thread(storage.delete_all_messages)
                branches = await asyncio.to_thread(storage.close_all_branches)
                await asyncio.to_thread(storage.reset_digest_counts)
                agent.reset_session_state()
                await message.edit_text(
                    "Готово.\n"
                    f"Обычных воспоминаний стёрто: {memories}\n"
                    f"Сообщений стёрто: {messages}\n"
                    f"Веток разговора закрыто: {branches}\n\n"
                    "Важные воспоминания (4–5) и личность Мембу на месте."
                )
            else:
                await message.edit_text("Отменено. Воспоминания на месте.")
            await callback.answer()
            return

        if kind == "mem":
            action = parts[1]

            if action == "noop":
                await callback.answer()
                return

            if action == "view":
                memory_id = int(parts[2])
                page = int(parts[3])
                view = await memory_detail_view(memory_id, page)

                if view is None:
                    await callback.answer("Запись уже удалена")
                    text, markup = await memory_page_view(page)
                    await message.edit_text(text, reply_markup=markup)
                    return

                text, markup = view
                await message.edit_text(text, reply_markup=markup)
                await callback.answer()
                return

            if action == "page":
                page = int(parts[2])
            elif action == "del":
                memory_id = int(parts[2])
                page = int(parts[3])
                deleted_chat = await asyncio.to_thread(storage.delete_memory, memory_id)
                if deleted_chat is not None:
                    agent.invalidate_memory_cache(deleted_chat)
                await callback.answer("Удалено из памяти")
            else:
                await callback.answer()
                return

            text, markup = await memory_page_view(page)
            await message.edit_text(text, reply_markup=markup)
            if action == "page":
                await callback.answer()
            return

        if kind == "msg":
            action = parts[1]

            if action == "noop":
                await callback.answer()
                return

            if action == "page":
                text, markup = await message_page_view(int(parts[2]))
                await message.edit_text(text, reply_markup=markup)
                await callback.answer()
                return

            if action == "view":
                chat_id = int(parts[2])
                message_id = int(parts[3])
                page = int(parts[4])
                view = await message_detail_view(chat_id, message_id, page)

                if view is None:
                    await callback.answer("Сообщение уже удалено")
                    text, markup = await message_page_view(page)
                    await message.edit_text(text, reply_markup=markup)
                    return

                text, markup = view
                await message.edit_text(text, reply_markup=markup)
                await callback.answer()
                return

            if action == "del":
                chat_id = int(parts[2])
                message_id = int(parts[3])
                page = int(parts[4])
                await asyncio.to_thread(storage.delete_message, chat_id, message_id)
                await callback.answer("Сообщение удалено")
                text, markup = await message_page_view(page)
                await message.edit_text(text, reply_markup=markup)
                return

            await callback.answer()
            return

        if kind == "ppl":
            action = parts[1]

            if action == "noop":
                await callback.answer()
                return

            if action == "page":
                text, markup = await people_page_view(int(parts[2]))
                await message.edit_text(text, reply_markup=markup)
                await callback.answer()
                return

            if action == "view":
                chat_id = int(parts[2])
                person_id = int(parts[3])
                page = int(parts[4])
                view = await person_detail_view(chat_id, person_id, page)

                if view is None:
                    await callback.answer("Мнение не найдено")
                    text, markup = await people_page_view(page)
                    await message.edit_text(text, reply_markup=markup)
                    return

                text, markup = view
                await message.edit_text(text, reply_markup=markup)
                await callback.answer()
                return

        await callback.answer()

    except TelegramAPIError as exc:
        # Например, «message is not modified», если страница не изменилась.
        logging.debug("Кнопка управления памятью: %s", exc)
    except (ValueError, IndexError):
        await callback.answer("Не понял кнопку.")


STYLE_VIEW_MESSAGES: dict[int, int] = {}


def _is_style_reply(message: Message) -> bool:
    return (
        is_owner_dm(message.from_user, message.chat.type)
        and message.reply_to_message is not None
        and message.reply_to_message.message_id in STYLE_VIEW_MESSAGES
    )


@router.message(F.text, _is_style_reply)
async def style_edit_reply(message: Message) -> None:
    if agent is None:
        return

    chat_id = STYLE_VIEW_MESSAGES[message.reply_to_message.message_id]
    instruction = message.text.strip()
    if not instruction:
        return

    status = await message.answer("Правлю профиль…")
    ok = await agent.apply_style_edit(chat_id, instruction)
    if not ok:
        await status.edit_text("Не получилось поправить профиль. Попробуй ещё раз.")
        return

    text, markup = await style_chat_view(chat_id)
    await status.edit_text(text, reply_markup=markup)
    STYLE_VIEW_MESSAGES[status.message_id] = chat_id


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
        if message.sticker is not None and storage is not None:
            # Стикер — немой ответ: сохраняем как «[стикер]» и ИИ не запускаем.
            await asyncio.to_thread(
                storage.add_message,
                {
                    "chat_id": chat_id,
                    "message_id": message.message_id,
                    "sender_id": message.from_user.id,
                    "sender_name": message.from_user.full_name or "",
                    "username": message.from_user.username or "",
                    "text": "[стикер]",
                    "reply_to_user_id": None,
                    "reply_to_message_id": None,
                    "branch_id": None,
                    "ts": message.date.timestamp(),
                },
            )
        return

    mentioned = is_mentioned(message, bot_user) or greeting_moment(
        message.text or message.caption or "",
        message.chat.id,
    )
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


_daily_marks: set[tuple] = set()
daily_tasks: set[asyncio.Task] = set()


async def daily_window_monitor() -> None:
    """Раз в окно (утро 7–9, вечер 21–23 по Дубаю) и на чат даёт модели шанс
    сам пожелать доброго утра или сладких снов. Решает модель."""
    await asyncio.sleep(60)

    while True:
        try:
            now = datetime.now(timezone(timedelta(hours=4)))

            if 7 <= now.hour < 9:
                window = "morning"
            elif 21 <= now.hour < 23:
                window = "evening"
            else:
                window = None

            if window and agent is not None and storage is not None:
                for chat_id in list(group_chat_ids):
                    key = (chat_id, now.date().isoformat(), window)
                    if key in _daily_marks:
                        continue

                    _daily_marks.add(key)
                    fut = enqueue_chat_job(
                        chat_id,
                        partial(
                            run_agent_with_context,
                            chat_id=chat_id,
                            branch_id=None,
                            reason=f"daily_window_{window}",
                            direct=False,
                        ),
                    )
                    fut.add_done_callback(
                        lambda f: f.cancelled() or f.exception()
                    )
        except Exception:
            logging.exception("Ошибка монитора утра и вечера")

        await asyncio.sleep(300)


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

    daily_task = asyncio.create_task(daily_window_monitor())
    daily_tasks.add(daily_task)
    daily_task.add_done_callback(daily_tasks.discard)

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

        # Web-сервис Render требует открытый порт. Поднимаем Flask в фоне.
        start_web_server()

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
