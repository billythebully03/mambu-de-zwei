import asyncio
import logging
import os
import re
from typing import Callable

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

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
DEBOUNCE_SECONDS = float(os.getenv("MEMBU_DEBOUNCE_SECONDS", "2.4"))
BRANCH_TTL = int(os.getenv("MEMBU_BRANCH_TTL_SECONDS", "300"))
IDLE_INITIATIVE_SECONDS = max(60, int(os.getenv("MEMBU_IDLE_SECONDS", "1200")))
IDLE_INITIATIVE_COOLDOWN_SECONDS = max(
    IDLE_INITIATIVE_SECONDS,
    int(os.getenv("MEMBU_IDLE_COOLDOWN_SECONDS", "3600")),
)
TYPING_REFRESH_SECONDS = 4.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

router = Router()
membu = MembuCore()

bot_user: User | None = None
storage: MembuStorage | None = None
agent: GeminiAgent | None = None
bot: Bot | None = None

scheduled_tasks: dict[tuple[int, str], asyncio.Task] = {}
group_chat_ids: set[int] = set()
chat_last_activity: dict[int, float] = {}
chat_user_activity_version: dict[int, int] = {}
chat_idle_events: dict[int, asyncio.Event] = {}
chat_idle_tasks: dict[int, asyncio.Task] = {}
chat_last_idle_attempt: dict[int, float] = {}


def is_explicit_private_request(text: str, bot_username: str | None = None) -> bool:
    """Распознаёт прямую просьбу в начале сообщения, чтобы не реагировать на цитаты."""
    text = (text or "").strip()
    if not text:
        return False

    bot_prefix = ""
    if bot_username:
        bot_prefix = rf"(?:@{re.escape(bot_username)}[\s,:!—-]*)?"

    pattern = re.compile(
        r"^\s*"
        + bot_prefix
        + r"(?:мембу[\s,:!—-]*)?"
        + r"(?:а\s+)?(?:пожалуйста[\s,:!—-]+)?"
        + r"(?:напиши|напишешь|напишите|пиши|отправь|скинь|перейди|давай|можешь)\b"
        + r".{0,60}\b(?:мне\s+)?(?:в\s+)?(?:лс|личк[уе]|личные\s+сообщения)\b",
        re.IGNORECASE,
    )
    return bool(pattern.search(text))



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


def _duration(value) -> str:
    try:
        seconds = max(0, int(value))
        minutes, seconds = divmod(seconds, 60)
        return f"{minutes}:{seconds:02d}" if minutes else f"{seconds} сек."
    except (TypeError, ValueError):
        return "длительность неизвестна"


def describe_message_content(message: Message) -> tuple[str, bool]:
    """Возвращает текст для контекста и признак вложения. Сам файл не скачивается."""
    plain_text = (message.text or "").strip()
    caption = (message.caption or "").strip()

    if plain_text:
        return plain_text, False

    media: list[str] = []

    if message.photo:
        photo = message.photo[-1]
        media.append(
            f"фотография ({getattr(photo, 'width', '?')}×{getattr(photo, 'height', '?')})"
        )

    if message.animation:
        animation = message.animation
        media.append(
            f"GIF/анимация ({_duration(getattr(animation, 'duration', None))})"
        )

    if message.sticker:
        sticker = message.sticker
        if getattr(sticker, "is_video", False):
            sticker_format = "видеостикер"
        elif getattr(sticker, "is_animated", False):
            sticker_format = "анимированный стикер"
        else:
            sticker_format = "статичный стикер"
        metadata = [sticker_format]
        if getattr(sticker, "emoji", None):
            metadata.append(f"эмодзи из метаданных: {sticker.emoji}")
        if getattr(sticker, "set_name", None):
            metadata.append(f"набор: {sticker.set_name}")
        media.append("; ".join(metadata))

    if message.video:
        video = message.video
        media.append(
            f"видео ({_duration(getattr(video, 'duration', None))}, "
            f"{getattr(video, 'width', '?')}×{getattr(video, 'height', '?')})"
        )

    if getattr(message, "live_photo", None):
        media.append("Live Photo")

    if message.video_note:
        media.append(
            f"видеосообщение-кружок ({_duration(getattr(message.video_note, 'duration', None))})"
        )

    if message.voice:
        media.append(
            f"голосовое сообщение ({_duration(getattr(message.voice, 'duration', None))})"
        )

    if message.audio:
        audio = message.audio
        details = [f"аудиофайл ({_duration(getattr(audio, 'duration', None))})"]
        if getattr(audio, "title", None):
            details.append(f"название: {audio.title}")
        if getattr(audio, "performer", None):
            details.append(f"исполнитель: {audio.performer}")
        media.append("; ".join(details))

    if message.document and not message.animation:
        document = message.document
        mime = (getattr(document, "mime_type", None) or "").lower()
        filename = getattr(document, "file_name", None)
        if mime == "image/gif":
            kind = "GIF-файл"
        elif mime.startswith("image/"):
            kind = "файл-изображение"
        elif mime.startswith("video/"):
            kind = "видеофайл"
        elif mime.startswith("audio/"):
            kind = "аудиофайл"
        else:
            kind = "документ или файл"
        label = kind + (f" ({filename})" if filename else "")
        media.append(label)

    if getattr(message, "paid_media", None):
        media.append("платное медиа")

    if message.poll:
        poll = message.poll
        options = [option.text for option in (poll.options or [])[:10]]
        poll_text = f"опрос: {poll.question}"
        if options:
            poll_text += "; варианты: " + " / ".join(options)
        media.append(poll_text)

    if message.contact:
        media.append("контактная карточка (контактные данные не раскрываются в контексте)")

    if message.location:
        media.append("геолокация (точные координаты не сохраняются в текстовом описании)")

    if message.venue:
        media.append("место/заведение, отправленное через Telegram")

    if message.dice:
        media.append(f"анимация с результатом: {message.dice.emoji} {message.dice.value}")

    if getattr(message, "story", None):
        media.append("пересланная история Telegram")

    if not media:
        return caption, False

    summary = "[ВЛОЖЕНИЕ: " + "; ".join(media) + "]"
    if caption:
        summary += "\nПодпись отправителя: " + caption
    summary += (
        "\n[Важно: Мембу получил только факт отправки и перечисленные метаданные. "
        "Само изображение, GIF, видео или аудио не анализировалось.]"
    )
    return summary[:3000], True


def mark_chat_activity(chat_id: int, *, human_activity: bool = True) -> None:
    if chat_id not in group_chat_ids:
        return

    loop = asyncio.get_running_loop()
    chat_last_activity[chat_id] = loop.time()
    if human_activity:
        chat_user_activity_version[chat_id] = chat_user_activity_version.get(chat_id, 0) + 1

    event = chat_idle_events.get(chat_id)
    if event is not None:
        event.set()

    task = chat_idle_tasks.get(chat_id)
    if task is None or task.done():
        chat_idle_tasks[chat_id] = asyncio.create_task(idle_initiative_monitor(chat_id))


def register_group_activity(chat_id: int) -> None:
    group_chat_ids.add(chat_id)
    chat_user_activity_version.setdefault(chat_id, 0)
    chat_last_activity.setdefault(chat_id, asyncio.get_running_loop().time())
    mark_chat_activity(chat_id, human_activity=True)


async def typing_indicator(chat_id: int) -> None:
    if bot is None:
        return
    while True:
        try:
            await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        except TelegramAPIError as exc:
            logging.debug("Не удалось обновить typing status в чате %s: %s", chat_id, exc)
        await asyncio.sleep(TYPING_REFRESH_SECONDS)


async def run_agent_with_context(
    chat_id: int,
    branch_id: str | None,
    reason: str,
    direct: bool,
    *,
    is_stale: Callable[[], bool] | None = None,
    trigger_user: dict | None = None,
    allow_private_message: bool = False,
) -> None:
    if storage is None or agent is None:
        return
    if is_stale and is_stale():
        logging.info("MEMBU | chat=%s | initiative_cancelled=stale_context", chat_id)
        return

    typing_task = asyncio.create_task(typing_indicator(chat_id))
    try:
        recent = await asyncio.to_thread(storage.recent_messages, chat_id, 18)
        branch_messages = []
        if branch_id:
            branch_messages = await asyncio.to_thread(storage.branch_messages, chat_id, branch_id, 25)

        active_branches = await asyncio.to_thread(storage.active_branches, chat_id, BRANCH_TTL)
        for branch in active_branches:
            branch["messages"] = await asyncio.to_thread(
                storage.branch_messages,
                chat_id,
                branch["branch_id"],
                12,
            )

        if is_stale and is_stale():
            logging.info("MEMBU | chat=%s | initiative_cancelled=stale_context_before_gemini", chat_id)
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
            trigger_user=trigger_user,
            allow_private_message=allow_private_message,
        )
    finally:
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass


async def process_after_pause(
    chat_id: int,
    branch_id: str | None,
    reason: str,
    direct: bool,
    wait_seconds: float,
    task_key: tuple[int, str],
    trigger_user: dict | None = None,
    allow_private_message: bool = False,
) -> None:
    current_task = asyncio.current_task()
    try:
        await asyncio.sleep(DEBOUNCE_SECONDS + max(0, wait_seconds))
        await run_agent_with_context(
            chat_id=chat_id,
            branch_id=branch_id,
            reason=reason,
            direct=direct,
            trigger_user=trigger_user,
            allow_private_message=allow_private_message,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        logging.exception("Ошибка при обработке контекста чата %s", chat_id)
    finally:
        if scheduled_tasks.get(task_key) is current_task:
            scheduled_tasks.pop(task_key, None)


def schedule_agent(
    chat_id: int,
    branch_id: str | None,
    reason: str,
    direct: bool,
    wait_seconds: float,
    trigger_user: dict | None = None,
    allow_private_message: bool = False,
) -> None:
    key = (chat_id, branch_id or "__general__")
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
            trigger_user=trigger_user,
            allow_private_message=allow_private_message,
        )
    )
    scheduled_tasks[key] = task


async def idle_initiative_monitor(chat_id: int) -> None:
    event = chat_idle_events.setdefault(chat_id, asyncio.Event())
    loop = asyncio.get_running_loop()
    try:
        while True:
            event.clear()
            activity_version = chat_user_activity_version.get(chat_id, 0)
            last_activity = chat_last_activity.get(chat_id, loop.time())
            last_attempt = chat_last_idle_attempt.get(chat_id)
            next_allowed = (
                last_attempt + IDLE_INITIATIVE_COOLDOWN_SECONDS
                if last_attempt is not None else 0.0
            )
            wake_at = max(last_activity + IDLE_INITIATIVE_SECONDS, next_allowed)
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
            is_stale = lambda: activity_version != chat_user_activity_version.get(chat_id, 0)
            try:
                await run_agent_with_context(
                    chat_id=chat_id,
                    branch_id=None,
                    reason="long_silence_initiative",
                    direct=False,
                    is_stale=is_stale,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logging.exception("Ошибка инициативы Мембу в чате %s", chat_id)

            if not is_stale():
                chat_last_idle_attempt[chat_id] = loop.time()
    except asyncio.CancelledError:
        raise
    finally:
        if chat_idle_tasks.get(chat_id) is asyncio.current_task():
            chat_idle_tasks.pop(chat_id, None)


@router.message(Command("privacy"))
async def privacy_command(message: Message) -> None:
    if message.chat.type in {"group", "supergroup"}:
        register_group_activity(message.chat.id)
    elif message.chat.type == "private" and message.from_user and storage is not None:
        await asyncio.to_thread(
            storage.register_private_user,
            message.from_user.id,
            message.chat.id,
            message.from_user.full_name or message.from_user.first_name or "",
            message.from_user.username or "",
        )
        await asyncio.to_thread(
            storage.add_message,
            {
                "chat_id": message.chat.id,
                "message_id": message.message_id,
                "sender_id": message.from_user.id,
                "sender_name": message.from_user.full_name or "",
                "username": message.from_user.username or "",
                "text": message.text or "/privacy",
                "reply_to_user_id": None,
                "reply_to_message_id": None,
                "branch_id": None,
                "ts": message.date.timestamp(),
            },
        )

    await message.reply(
        "Я могу получать сообщения этого чата, если Telegram передаёт их мне. "
        "Контекст разговоров может отправляться в Google Gemini для анализа. "
        "История и выбранные воспоминания хранятся в Supabase. "
        "Личные сообщения хранятся отдельно от групповых. Не отправляйте сюда "
        "пароли, документы и личные секреты. Я не анализирую содержимое фото, "
        "видео и аудио в этой версии."
    )


def _member_status(member) -> str:
    status = getattr(member, "status", "")
    return str(getattr(status, "value", status))


async def _is_group_admin(chat_id: int, user_id: int) -> bool:
    if bot is None:
        return False
    try:
        member = await bot.get_chat_member(chat_id, user_id)
        return _member_status(member) in {"creator", "administrator"}
    except TelegramAPIError:
        return False


@router.message(Command("membu_contact"))
async def set_membu_contact(message: Message) -> None:
    if (
        storage is None
        or message.from_user is None
        or message.chat.type not in {"group", "supergroup"}
    ):
        return

    user_id = message.from_user.id
    if not await _is_group_admin(message.chat.id, user_id):
        await message.reply("Назначить контакт Мембу может только администратор чата.")
        return

    opted_in = await asyncio.to_thread(storage.has_private_chat_opt_in, user_id)
    if not opted_in:
        username = bot_user.username if bot_user else None
        link = f"https://t.me/{username}?start=private" if username else "открой @membu_bot"
        await message.reply(
            "Сначала открой личный чат с Мембу и нажми Start, чтобы он мог написать тебе. "
            f"Ссылка: {link}\nПотом повтори /membu_contact в этой группе."
        )
        return

    await asyncio.to_thread(
        storage.set_chat_contact,
        message.chat.id,
        user_id,
        message.from_user.full_name or "",
        message.from_user.username or "",
    )
    await message.reply(
        "Готово. Если Мембу удалят из чата, он сможет один раз написать тебе в личку. "
        "Повторные просьбы и рассылки отключены."
    )


@router.message(Command("membu_contact_off"))
async def remove_membu_contact(message: Message) -> None:
    if (
        storage is None
        or message.from_user is None
        or message.chat.type not in {"group", "supergroup"}
    ):
        return
    if not await _is_group_admin(message.chat.id, message.from_user.id):
        await message.reply("Отключить контакт Мембу может только администратор чата.")
        return
    await asyncio.to_thread(storage.remove_chat_contact, message.chat.id)
    await message.reply("Личный контакт для уведомлений Мембу отключён.")


@router.my_chat_member()
async def handle_bot_membership_update(update: ChatMemberUpdated) -> None:
    """Если Мембу удалили, он может один раз обратиться к назначенному админу."""
    if storage is None or agent is None or bot_user is None:
        return
    if update.new_chat_member.user.id != bot_user.id:
        return

    old_status = _member_status(update.old_chat_member)
    new_status = _member_status(update.new_chat_member)

    # В личном чате пользователь мог заблокировать или разблокировать бота.
    if update.chat.type == "private":
        person_id = update.from_user.id if update.from_user else update.chat.id
        if new_status in {"kicked", "left"}:
            await asyncio.to_thread(storage.set_private_user_enabled, person_id, False)
        return

    if update.chat.type not in {"group", "supergroup"}:
        return
    was_present = old_status in {"member", "administrator", "restricted"}
    was_removed = new_status in {"left", "kicked"}
    if not (was_present and was_removed):
        return

    chat_id = update.chat.id
    logging.info(
        "MEMBU | chat=%s | bot_membership_changed=%s->%s",
        chat_id,
        old_status,
        new_status,
    )

    try:
        contact = await asyncio.to_thread(storage.claim_chat_contact_notice, chat_id)
        if not contact:
            logging.info("MEMBU | chat=%s | removal_notice=no_contact_or_cooldown", chat_id)
            return

        user_id = int(contact["user_id"])
        if not await asyncio.to_thread(storage.has_private_chat_opt_in, user_id):
            logging.info("MEMBU | chat=%s | removal_notice=contact_has_no_private_opt_in", chat_id)
            return

        recent = await asyncio.to_thread(storage.recent_messages, chat_id, 18)
        trigger_user = {
            "user_id": user_id,
            "name": contact.get("display_name") or "администратор чата",
            "username": contact.get("username") or "",
        }

        await agent.process(
            chat_id=chat_id,
            branch_id=None,
            reason="bot_removed_from_chat",
            direct=True,
            recent_messages=recent,
            branch_messages=[],
            active_branches=[],
            trigger_user=trigger_user,
            allow_private_message=True,
        )
    except Exception:
        logging.exception("Не удалось обработать удаление Мембу из чата %s", chat_id)


async def _handle_membership_message(message: Message) -> None:
    if storage is None or bot_user is None:
        return
    if message.chat.type not in {"group", "supergroup"}:
        return

    register_group_activity(message.chat.id)
    joined = [u for u in (message.new_chat_members or []) if u.id != bot_user.id]
    left = message.left_chat_member
    if left and left.id == bot_user.id:
        left = None
    if not joined and left is None:
        return

    descriptions = []
    actor_id = message.from_user.id if message.from_user else None
    for user in joined:
        name = user.full_name or user.first_name or "участник"
        username = f" (@{user.username})" if user.username else ""
        action = "сам присоединился" if actor_id == user.id else "добавлен в чат"
        descriptions.append(f"{action}: {name}{username}")
    if left is not None:
        name = left.full_name or left.first_name or "участник"
        username = f" (@{left.username})" if left.username else ""
        action = "сам покинул чат" if actor_id == left.id else "удалён из чата"
        descriptions.append(f"{action}: {name}{username}")

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
    schedule_agent(
        chat_id=message.chat.id,
        branch_id=None,
        reason=reason,
        direct=False,
        wait_seconds=0,
    )


@router.message(F.new_chat_members | F.left_chat_member)
async def handle_membership_event(message: Message) -> None:
    await _handle_membership_message(message)


@router.message()
async def handle_message(message: Message) -> None:
    global bot_user, storage
    if bot_user is None or storage is None or message.from_user is None:
        return
    if message.from_user.is_bot:
        return

    chat_id = message.chat.id
    is_group = message.chat.type in {"group", "supergroup"}
    private_chat = message.chat.type == "private"

    if is_group:
        register_group_activity(chat_id)
    elif private_chat:
        # Любое входящее сообщение в личке подтверждает, что пользователь
        # сам открыл диалог; это разрешает отвечать ему из группового сценария.
        await asyncio.to_thread(
            storage.register_private_user,
            message.from_user.id,
            message.chat.id,
            message.from_user.full_name or message.from_user.first_name or "",
            message.from_user.username or "",
        )

    content_text, has_media = describe_message_content(message)
    if not content_text.strip() and not has_media:
        return

    raw_text = message.text or message.caption or ""
    mentioned = is_mentioned(message, bot_user)
    replied_to_bot = is_reply_to_bot(message, bot_user)
    private_request = is_group and is_explicit_private_request(raw_text, bot_user.username)
    direct = mentioned or replied_to_bot or private_chat or private_request

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

    active_branches = await asyncio.to_thread(storage.active_branches, chat_id, BRANCH_TTL)

    event = {
        "chat_id": chat_id,
        "message_id": message.message_id,
        "sender_id": message.from_user.id,
        "sender_name": message.from_user.full_name or "",
        "username": message.from_user.username or "",
        "text": content_text[:3000],
        "reply_to_user_id": reply_to_user_id,
        "reply_to_message_id": reply_to_message_id,
        "reply_to_branch_id": reply_to_branch_id,
        "mentioned": mentioned,
        "replied_to_bot": replied_to_bot,
        "private_chat": private_chat,
        "private_message_request": private_request,
        "has_media": has_media,
        "ts": message.date.timestamp(),
    }

    decision = membu.inspect(event=event, active_branches=active_branches)
    event["branch_id"] = decision.branch_id

    if decision.direct and decision.branch_id:
        existing_ids = {branch["branch_id"] for branch in active_branches}
        if decision.branch_id not in existing_ids:
            await asyncio.to_thread(
                storage.create_branch,
                chat_id,
                decision.branch_id,
                message.message_id,
                message.from_user.id,
            )

    await asyncio.to_thread(storage.add_message, event)

    if decision.branch_id:
        await asyncio.to_thread(storage.touch_branch, chat_id, decision.branch_id)

    logging.info(
        "CHAT=%s USER=%s DIRECT=%s AI=%s REASON=%s BRANCH=%s MEDIA=%s",
        chat_id,
        message.from_user.id,
        decision.direct,
        decision.call_ai,
        decision.reason,
        decision.branch_id,
        has_media,
    )

    if not decision.call_ai:
        return

    requester = None
    allow_private = False
    if is_group:
        requester = {
            "user_id": message.from_user.id,
            "name": message.from_user.full_name or "участник",
            "username": message.from_user.username or "",
        }
        allow_private = private_request

    schedule_agent(
        chat_id=chat_id,
        branch_id=decision.branch_id,
        reason=decision.reason,
        direct=decision.direct,
        wait_seconds=decision.wait_seconds,
        trigger_user=requester,
        allow_private_message=allow_private,
    )


async def main() -> None:
    global bot_user, storage, agent, bot

    required = {
        "TOKEN": TOKEN,
        "GEMINI_API_KEY": GEMINI_API_KEY,
        "SUPABASE_URL": SUPABASE_URL,
        "SUPABASE_SERVICE_ROLE_KEY": SUPABASE_SERVICE_ROLE_KEY,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError("Не заданы переменные окружения: " + ", ".join(missing))

    bot = Bot(token=TOKEN)
    try:
        bot_user = await bot.get_me()
        storage = MembuStorage(url=SUPABASE_URL, key=SUPABASE_SERVICE_ROLE_KEY)
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
            "Мембу запущен: @%s | model=%s | idle=%ss | idle_cooldown=%ss",
            bot_user.username,
            GEMINI_MODEL,
            IDLE_INITIATIVE_SECONDS,
            IDLE_INITIATIVE_COOLDOWN_SECONDS,
        )
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(
            bot,
            allowed_updates=dp.resolve_used_update_types(),
        )
    finally:
        for task in scheduled_tasks.values():
            task.cancel()
        for task in chat_idle_tasks.values():
            task.cancel()
        all_tasks = list(scheduled_tasks.values()) + list(chat_idle_tasks.values())
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)
        if bot is not None:
            await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
