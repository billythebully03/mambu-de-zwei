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

    def recent_messages(self, chat_id: int, limit: int = 18) -> list[dict]:
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
            datetime.now(timezone.utc) - timedelta(seconds=ttl_seconds)
        ).isoformat()
        self.close_stale_branches(chat_id, ttl_seconds)
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

    def touch_branch(self, chat_id: int, branch_id: str):
        self.client.table("membu_branches").update(
            {
                "last_activity_at": self._utc_now(),
                "active": True,
            }
        ).eq("chat_id", chat_id).eq("branch_id", branch_id).execute()

    def close_stale_branches(
        self,
        chat_id: int,
        ttl_seconds: int = 300,
    ):
        cutoff = (
            datetime.now(timezone.utc) - timedelta(seconds=ttl_seconds)
        ).isoformat()
        self.client.table("membu_branches").update(
            {"active": False}
        ).eq("chat_id", chat_id).eq("active", True).lt(
            "last_activity_at", cutoff
        ).execute()

    def set_branch_ai_time(self, chat_id: int, branch_id: str):
        self.client.table("membu_branches").update(
            {"last_ai_at": self._utc_now()}
        ).eq("chat_id", chat_id).eq("branch_id", branch_id).execute()

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

    def is_style_frozen(self, chat_id: int) -> bool:
        try:
            result = (
                self.client.table("membu_chat_profiles")
                .select("style_frozen")
                .eq("chat_id", chat_id)
                .limit(1)
                .execute()
            )
        except Exception:
            return False
        rows = result.data or []
        return bool(rows and rows[0].get("style_frozen"))

    def set_style_frozen(self, chat_id: int, frozen: bool) -> None:
        self.client.table("membu_chat_profiles").upsert(
            {"chat_id": chat_id, "style_frozen": frozen},
            on_conflict="chat_id",
        ).execute()

    def get_chat_style(self, chat_id: int) -> dict:
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
        observed = int(profile.get("observed_messages") or 0)
        analyzed = int(profile.get("last_analyzed_message_count") or 0)
        profile["new_messages_since_analysis"] = max(0, observed - analyzed)
        return profile

    def update_chat_style(
        self,
        chat_id: int,
        style_profile: str,
        minimum_new_messages: int = 15,
    ) -> dict:
        style_profile = str(style_profile or "").strip()
        if not style_profile:
            return {"updated": False, "reason": "empty_profile"}

        current = self.get_chat_style(chat_id)
        observed = int(current.get("observed_messages") or 0)
        analyzed = int(current.get("last_analyzed_message_count") or 0)
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

    def load_memories(self, chat_id: int, limit: int = 60) -> list[dict]:
        """Самые важные воспоминания чата. Нужны для кэша в памяти процесса."""
        result = (
            self.client.table("membu_memories")
            .select("id,category,content,importance,people,updated_at")
            .eq("chat_id", chat_id)
            .order("importance", desc=True)
            .order("updated_at", desc=True)
            .limit(max(1, min(limit, 200)))
            .execute()
        )
        return result.data or []

    def search_own_messages(
        self,
        chat_id: int,
        bot_id: int,
        query: str,
        limit: int = 5,
    ) -> list[dict]:
        """Буквальный поиск (подстрока без учёта регистра) по сообщениям самого Мембу."""
        escaped = (
            query.replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        result = (
            self.client.table("membu_messages")
            .select("message_id,text,created_at")
            .eq("chat_id", chat_id)
            .eq("sender_id", bot_id)
            .ilike("text", f"%{escaped}%")
            .order("created_at", desc=True)
            .limit(max(1, min(limit, 8)))
            .execute()
        )
        return result.data or []

    def search_history(
        self,
        chat_id: int,
        query: str,
        limit: int = 8,
        before: str | None = None,
    ) -> list[dict]:
        """Поиск по всей истории чата (люди и Мембу) с учётом окончаний слов."""
        params = {
            "p_chat_id": chat_id,
            "p_query": query,
            "p_limit": max(1, min(limit, 20)),
            "p_before": before,
        }
        result = self.client.rpc("search_membu_history", params).execute()
        return result.data or []

    def get_digest_state(self, chat_id: int) -> dict:
        """Сколько сообщений чата накопилось и сколько уже вошло в дайджест."""
        result = (
            self.client.table("membu_chat_profiles")
            .select("observed_messages,last_digested_count")
            .eq("chat_id", chat_id)
            .limit(1)
            .execute()
        )
        rows = result.data or []
        if not rows:
            return {"observed": 0, "digested": 0}
        return {
            "observed": int(rows[0].get("observed_messages") or 0),
            "digested": int(rows[0].get("last_digested_count") or 0),
        }

    def mark_digested(self, chat_id: int, observed: int) -> None:
        self.client.table("membu_chat_profiles").update(
            {"last_digested_count": observed}
        ).eq("chat_id", chat_id).execute()

    def history_page(
        self,
        chat_id: int,
        before: str | None = None,
        limit: int = 30,
    ) -> tuple[list[dict], bool]:
        """Страница истории по времени, от новых к старым.

        Возвращает сообщения в хронологическом порядке и признак,
        есть ли более ранние сообщения.
        """
        query = (
            self.client.table("membu_messages")
            .select(
                "message_id,sender_name,text,created_at,"
                "reply_to_message_id,branch_id"
            )
            .eq("chat_id", chat_id)
        )

        if before:
            query = query.lt("created_at", before)

        rows = (
            query.order("created_at", desc=True)
            .limit(limit + 1)
            .execute()
            .data
            or []
        )

        has_more = len(rows) > limit
        rows = rows[:limit]
        return list(reversed(rows)), has_more

    def last_bot_membership(self, chat_id: int) -> dict:
        """Последнее событие добавления или удаления Мембу из истории чата.

        Нужно, чтобы после перезапуска бот знал свой статус без новых событий.
        """
        result = (
            self.client.table("membu_messages")
            .select("text,created_at")
            .eq("chat_id", chat_id)
            .like("text", "[СОБЫТИЕ ЧАТА] Мембу%")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        rows = result.data or []
        if not rows:
            return {}

        text = str(rows[0].get("text", ""))
        changed_at = str(rows[0].get("created_at", ""))

        if "добавили в чат" in text:
            status = "in_chat"
        elif "удалили из чата" in text:
            status = "removed"
        else:
            return {}

        by = ""
        for marker in ("добавил: ", "удалил: "):
            if marker in text:
                by = text.split(marker, 1)[1].rstrip(")")
                break

        return {"status": status, "changed_at": changed_at, "by": by}

    def list_all_memories(self, limit: int = 500) -> list[dict]:
        """Все воспоминания бота из всех чатов, свежие сверху."""
        result = (
            self.client.table("membu_memories")
            .select("id,chat_id,category,content,people,importance,updated_at")
            .order("updated_at", desc=True)
            .limit(max(1, min(limit, 1000)))
            .execute()
        )
        return result.data or []

    def delete_memory(self, memory_id: int) -> int | None:
        """Удаляет одно воспоминание. Возвращает chat_id, чтобы сбросить кэш."""
        result = (
            self.client.table("membu_memories")
            .delete()
            .eq("id", memory_id)
            .execute()
        )
        rows = result.data or []
        return int(rows[0]["chat_id"]) if rows else None

    def delete_unimportant_memories(self, keep_from_importance: int = 4) -> int:
        """Стирает только не важные воспоминания (важность ниже порога).
        Важные (4 и 5) и личность (PERSONALITY.md) не трогает."""
        result = (
            self.client.table("membu_memories")
            .delete()
            .lt("importance", keep_from_importance)
            .execute()
        )
        return len(result.data or [])

    def delete_all_messages(self) -> int:
        """Стирает историю сообщений всех чатов (общая память)."""
        result = (
            self.client.table("membu_messages")
            .delete()
            .neq("chat_id", 0)
            .execute()
        )
        return len(result.data or [])

    def close_all_branches(self) -> int:
        """Закрывает все ветки разговоров (сессионная память ИИ)."""
        result = (
            self.client.table("membu_branches")
            .delete()
            .neq("branch_id", "")
            .execute()
        )
        return len(result.data or [])

    def delete_all_people(self) -> int:
        """Стирает мнения о людях всех чатов."""
        result = (
            self.client.table("membu_people")
            .delete()
            .neq("chat_id", 0)
            .execute()
        )
        return len(result.data or [])

    def list_style_profiles(self, limit: int = 20) -> list[dict]:
        """Профили стиля по чатам, свежие сверху."""
        result = (
            self.client.table("membu_chat_profiles")
            .select("chat_id,style_profile,observed_messages,updated_at")
            .order("updated_at", desc=True)
            .limit(max(1, min(limit, 50)))
            .execute()
        )
        return result.data or []

    def reset_chat_styles(self) -> None:
        """Сбрасывает выученный стиль чата: Мембу снова опирается только на PERSONALITY."""
        (
            self.client.table("membu_chat_profiles")
            .update({
                "style_profile": "",
                "observed_messages": 0,
                "last_analyzed_message_count": 0,
            })
            .neq("chat_id", 0)
            .execute()
        )

    def reset_digest_counts(self) -> None:
        """Сбрасывает счётчик дайджеста: после стирания сообщений считать заново."""
        (
            self.client.table("membu_chat_profiles")
            .update({"last_digested_count": 0})
            .neq("chat_id", 0)
            .execute()
        )

    def get_memory(self, memory_id: int) -> dict | None:
        """Одна запись памяти для подробного просмотра."""
        result = (
            self.client.table("membu_memories")
            .select("id,chat_id,category,content,people,importance,updated_at")
            .eq("id", memory_id)
            .limit(1)
            .execute()
        )
        rows = result.data or []
        return rows[0] if rows else None

    def list_messages_page(self, page: int, size: int) -> tuple[list[dict], int]:
        """Все сообщения из базы, от новых к старым, с общим количеством."""
        start = max(0, page) * size
        result = (
            self.client.table("membu_messages")
            .select(
                "message_id,chat_id,sender_name,text,created_at",
                count="exact",
            )
            .order("created_at", desc=True)
            .range(start, start + size - 1)
            .execute()
        )
        return result.data or [], int(result.count or 0)

    def get_message(self, chat_id: int, message_id: int) -> dict | None:
        result = (
            self.client.table("membu_messages")
            .select("message_id,chat_id,sender_name,text,created_at")
            .eq("chat_id", chat_id)
            .eq("message_id", message_id)
            .limit(1)
            .execute()
        )
        rows = result.data or []
        return rows[0] if rows else None

    def delete_message(self, chat_id: int, message_id: int) -> bool:
        result = (
            self.client.table("membu_messages")
            .delete()
            .eq("chat_id", chat_id)
            .eq("message_id", message_id)
            .execute()
        )
        return bool(result.data)

    def list_people_page(self, page: int, size: int) -> tuple[list[dict], int]:
        start = max(0, page) * size
        result = (
            self.client.table("membu_people")
            .select(
                "chat_id,person_id,display_name,opinion,message_count,updated_at",
                count="exact",
            )
            .order("updated_at", desc=True)
            .range(start, start + size - 1)
            .execute()
        )
        return result.data or [], int(result.count or 0)

    def get_person(self, chat_id: int, person_id: int) -> dict | None:
        result = (
            self.client.table("membu_people")
            .select("chat_id,person_id,display_name,opinion,message_count,updated_at")
            .eq("chat_id", chat_id)
            .eq("person_id", person_id)
            .limit(1)
            .execute()
        )
        rows = result.data or []
        return rows[0] if rows else None

    def list_people_for_chat(self, chat_id: int, limit: int = 200) -> list[dict]:
        result = (
            self.client.table("membu_people")
            .select("person_id,display_name,opinion")
            .eq("chat_id", chat_id)
            .limit(max(1, min(limit, 500)))
            .execute()
        )
        return result.data or []

    def upsert_person(
        self,
        chat_id: int,
        person_id: int,
        display_name: str,
        opinion: str,
        message_count: int,
    ) -> None:
        self.client.table("membu_people").upsert(
            {
                "chat_id": chat_id,
                "person_id": person_id,
                "display_name": display_name[:200],
                "opinion": opinion[:800],
                "message_count": message_count,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            on_conflict="chat_id,person_id",
        ).execute()
