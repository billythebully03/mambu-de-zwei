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
            "reply_to_user_id": event.get(
                "reply_to_user_id"
            ),
            "reply_to_message_id": event.get(
                "reply_to_message_id"
            ),
            "branch_id": event.get("branch_id"),
            "created_at": created_at,
        }

        self.client.table(
            "membu_messages"
        ).upsert(
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

        self.client.table(
            "membu_branches"
        ).upsert(
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

        # Старые ветки больше не считаются активными.
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

        self.client.table(
            "membu_branches"
        ).update(
            {
                "last_activity_at": self._utc_now(),
                "active": True,
            }
        ).eq(
            "chat_id", chat_id
        ).eq(
            "branch_id", branch_id
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

        self.client.table(
            "membu_branches"
        ).update(
            {"active": False}
        ).eq(
            "chat_id", chat_id
        ).eq(
            "active", True
        ).lt(
            "last_activity_at", cutoff
        ).execute()

    def set_branch_ai_time(
        self,
        chat_id: int,
        branch_id: str,
    ):

        self.client.table(
            "membu_branches"
        ).update(
            {"last_ai_at": self._utc_now()}
        ).eq(
            "chat_id", chat_id
        ).eq(
            "branch_id", branch_id
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

        self.client.table(
            "membu_memories"
        ).upsert(
            data,
            on_conflict="chat_id,category,content",
        ).execute()
