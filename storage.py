from datetime import datetime, timedelta, timezone

from supabase import create_client


class MembuStorage:
    def __init__(self, url: str, key: str):
        self.client = create_client(url, key)

    @staticmethod
    def _utc_now():
        return datetime.now(timezone.utc).isoformat()

    def add_message(self, event: dict):
        created_at = datetime.fromtimestamp(
            event.get("ts", datetime.now().timestamp()),
            timezone.utc,
        ).isoformat()

        data = {
            "chat_id": event["chat_id"],
            "message_id": event["message_id"],
            "sender_id": event.get("sender_id"),
            "sender_name": event.get("sender_name", ""),
            "username": event.get("username", ""),
            "text": event.get("text", "")[:3000],
            "reply_to_user_id": event.get("reply_to_user_id"),
            "reply_to_message_id": event.get("reply_to_message_id"),
            "branch_id": event.get("branch_id"),
            "created_at": created_at,
        }

        self.client.table("membu_messages").upsert(
            data,
            on_conflict="chat_id,message_id",
            ignore_duplicates=True,
        ).execute()

    def recent_messages(
        self,
        chat_id: int,
        limit: int = 18,
    ) -> list[dict]:
        result = (
            self.client.table("membu_messages")
            .select("*")
            .eq("chat_id", chat_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )

        return list(reversed(result.data or []))

    def get_branch_for_message(
        self,
        chat_id: int,
        message_id: int,
    ) -> str | None:
        result = (
            self.client.table("membu_messages")
            .select("branch_id")
            .eq("chat_id", chat_id)
            .eq("message_id", message_id)
            .limit(1)
            .execute()
        )

        rows = result.data or []

        if not rows:
            return None

        return rows[0].get("branch_id")

    def create_branch(
        self,
        chat_id: int,
        branch_id: str,
        root_message_id: int,
        root_sender_id: int,
    ):
        data = {
            "branch_id": branch_id,
            "chat_id": chat_id,
            "root_message_id": root_message_id,
            "root_sender_id": root_sender_id,
            "active": True,
            "last_activity_at": self._utc_now(),
        }

        self.client.table("membu_branches").upsert(
            data,
            on_conflict="branch_id",
        ).execute()

    def active_branches(
        self,
        chat_id: int,
        ttl_seconds: int = 300,
    ) -> list[dict]:
        cutoff = (
            datetime.now(timezone.utc)
            - timedelta(seconds=ttl_seconds)
        ).isoformat()

        self.close_stale_branches(
            chat_id,
            ttl_seconds,
        )

        result = (
            self.client.table("membu_branches")
            .select("*")
            .eq("chat_id", chat_id)
            .eq("active", True)
            .gte("last_activity_at", cutoff)
            .order("last_activity_at", desc=True)
            .limit(5)
            .execute()
        )

        return result.data or []

    def branch_messages(
        self,
        chat_id: int,
        branch_id: str,
        limit: int = 25,
    ) -> list[dict]:
        result = (
            self.client.table("membu_messages")
            .select("*")
            .eq("chat_id", chat_id)
            .eq("branch_id", branch_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )

        return list(reversed(result.data or []))

    def touch_branch(
        self,
        chat_id: int,
        branch_id: str,
    ):
        self.client.table("membu_branches").update(
            {
                "last_activity_at": self._utc_now(),
                "active": True,
            }
        ).eq(
            "chat_id",
            chat_id,
        ).eq(
            "branch_id",
            branch_id,
        ).execute()

    def close_stale_branches(
        self,
        chat_id: int,
        ttl_seconds: int = 300,
    ):
        cutoff = (
            datetime.now(timezone.utc)
            - timedelta(seconds=ttl_seconds)
        ).isoformat()

        self.client.table("membu_branches").update(
            {"active": False}
        ).eq(
            "chat_id",
            chat_id,
        ).eq(
            "active",
            True,
        ).lt(
            "last_activity_at",
            cutoff,
        ).execute()

    def set_branch_ai_time(
        self,
        chat_id: int,
        branch_id: str,
    ):
        self.client.table("membu_branches").update(
            {"last_ai_at": self._utc_now()}
        ).eq(
            "chat_id",
            chat_id,
        ).eq(
            "branch_id",
            branch_id,
        ).execute()

    def search_memories(
        self,
        chat_id: int,
        query: str,
        category: str | None = None,
        limit: int = 5,
    ) -> list[dict]:
        result = self.client.rpc(
            "search_membu_memory",
            {
                "p_chat_id": chat_id,
                "p_query": query,
                "p_category": category,
                "p_limit": max(1, min(limit, 8)),
            },
        ).execute()

        return result.data or []

    def save_memory(
        self,
        chat_id: int,
        category: str,
        content: str,
        importance: int = 3,
        people: list[str] | None = None,
    ):
        data = {
            "chat_id": chat_id,
            "category": category,
            "content": content[:1500],
            "importance": max(1, min(importance, 5)),
            "people": (people or [])[:10],
            "updated_at": self._utc_now(),
        }

        self.client.table("membu_memories").upsert(
            data,
            on_conflict="chat_id,category,content",
        ).execute()

    def get_chat_style(
        self,
        chat_id: int,
    ) -> dict:
        result = (
            self.client.table("membu_chat_profiles")
            .select(
                "chat_id,style_profile,observed_messages,"
                "last_analyzed_message_count,updated_at"
            )
            .eq("chat_id", chat_id)
            .limit(1)
            .execute()
        )

        rows = result.data or []

        if not rows:
            return {
                "chat_id": chat_id,
                "style_profile": "",
                "observed_messages": 0,
                "last_analyzed_message_count": 0,
                "new_messages_since_analysis": 0,
            }

        profile = rows[0]

        observed = int(
            profile.get("observed_messages") or 0
        )

        analyzed = int(
            profile.get("last_analyzed_message_count") or 0
        )

        profile["new_messages_since_analysis"] = max(
            0,
            observed - analyzed,
        )

        return profile

    def update_chat_style(
        self,
        chat_id: int,
        style_profile: str,
        minimum_new_messages: int = 15,
    ) -> dict:
        style_profile = str(style_profile or "").strip()

        if not style_profile:
            return {
                "updated": False,
                "reason": "empty_profile",
            }

        current = self.get_chat_style(chat_id)

        observed = int(
            current.get("observed_messages") or 0
        )

        analyzed = int(
            current.get("last_analyzed_message_count") or 0
        )

        new_messages = observed - analyzed

        if new_messages < minimum_new_messages:
            return {
                "updated": False,
                "reason": "not_enough_new_messages",
                "new_messages": new_messages,
                "required": minimum_new_messages,
            }

        data = {
            "chat_id": chat_id,
            "style_profile": style_profile[:2500],
            "last_analyzed_message_count": observed,
            "updated_at": self._utc_now(),
        }

        self.client.table("membu_chat_profiles").upsert(
            data,
            on_conflict="chat_id",
        ).execute()

        return {
            "updated": True,
            "observed_messages": observed,
            "profile_length": len(style_profile[:2500]),
        }

    # =========================================
    # ПОИСК СОБСТВЕННЫХ СООБЩЕНИЙ МЕМБУ
    # =========================================

    def search_own_messages(
        self,
        chat_id: int,
        sender_id: int,
        query: str,
        limit: int = 5,
    ) -> list[dict]:
        """
        Ищет только сообщения Мембу в указанном чате.

        Поиск буквальный: запрос должен встречаться в тексте
        сообщения как подстрока.
        """

        query = str(query or "").strip()[:120]

        if len(query) < 2:
            return []

        # Экранируем символы шаблона LIKE, чтобы введённые
        # пользователем проценты и подчёркивания не превращались
        # в произвольные маски поиска.
        escaped_query = (
            query
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )

        pattern = f"%{escaped_query}%"

        result = (
            self.client.table("membu_messages")
            .select(
                "message_id,text,created_at,"
                "reply_to_message_id,branch_id"
            )
            .eq("chat_id", chat_id)
            .eq("sender_id", sender_id)
            .ilike("text", pattern)
            .order("created_at", desc=True)
            .limit(max(1, min(limit, 8)))
            .execute()
        )

        messages = result.data or []

        # Не передаём модели чрезмерно длинные фрагменты истории.
        for message in messages:
            message["text"] = str(
                message.get("text") or ""
            )[:1500]

        return messages

    # =========================================
    # ПРОВЕРКА ИСТОРИИ И ПРАВ НА РЕДАКТИРОВАНИЕ
    # =========================================

    def get_own_message(
        self,
        chat_id: int,
        message_id: int,
        sender_id: int,
    ) -> dict | None:
        """
        Возвращает сообщение только в том случае,
        если оно действительно принадлежит Мембу.
        """

        result = (
            self.client.table("membu_messages")
            .select(
                "chat_id,message_id,sender_id,text,created_at,branch_id"
            )
            .eq("chat_id", chat_id)
            .eq("message_id", message_id)
            .eq("sender_id", sender_id)
            .limit(1)
            .execute()
        )

        rows = result.data or []

        if not rows:
            return None

        return rows[0]

    def has_message_been_edited(
        self,
        chat_id: int,
        message_id: int,
    ) -> bool:
        """
        Проверяет, редактировалось ли сообщение раньше.
        """

        result = (
            self.client.table("membu_message_edits")
            .select("id")
            .eq("chat_id", chat_id)
            .eq("message_id", message_id)
            .limit(1)
            .execute()
        )

        return bool(result.data)

    def recent_message_edits(
        self,
        chat_id: int,
        bot_user_id: int,
        since_minutes: int = 60,
        limit: int = 3,
    ) -> list[dict]:
        """
        Возвращает последние правки Мембу в чате
        за указанный промежуток времени.
        """

        since_minutes = max(
            1,
            min(int(since_minutes), 60),
        )

        limit = max(
            1,
            min(int(limit), 3),
        )

        cutoff = (
            datetime.now(timezone.utc)
            - timedelta(minutes=since_minutes)
        ).isoformat()

        result = (
            self.client.table("membu_message_edits")
            .select("message_id,edited_at")
            .eq("chat_id", chat_id)
            .eq("bot_user_id", bot_user_id)
            .gte("edited_at", cutoff)
            .order("edited_at", desc=True)
            .limit(limit)
            .execute()
        )

        return result.data or []

    def record_message_edit(
        self,
        chat_id: int,
        message_id: int,
        bot_user_id: int,
        old_text: str,
        new_text: str,
        reason: str,
    ) -> dict:
        """
        Атомарно обновляет текст сообщения в истории
        и записывает журнал редактирования.

        SQL-функция выполняет обе операции в одной транзакции.
        """

        result = self.client.rpc(
            "record_membu_message_edit",
            {
                "p_chat_id": chat_id,
                "p_message_id": message_id,
                "p_bot_user_id": bot_user_id,
                "p_old_text": old_text,
                "p_new_text": new_text,
                "p_reason": reason[:300],
            },
        ).execute()

        data = result.data

        if isinstance(data, dict):
            return data

        if isinstance(data, list) and data:
            first = data[0]

            if isinstance(first, dict):
                return first

        return {
            "recorded": bool(data),
            "reason": None if data else "empty_rpc_result",
        }
