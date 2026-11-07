-- MEMBU: отдельный профиль манеры общения каждого чата.
-- Запустить один раз в Supabase SQL Editor после основной supabase.sql.

create table if not exists public.membu_chat_profiles (
    chat_id bigint primary key,
    style_profile text not null default '',
    observed_messages bigint not null default 0,
    last_analyzed_message_count bigint not null default 0,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

alter table public.membu_chat_profiles enable row level security;

revoke all on public.membu_chat_profiles
from anon, authenticated;

grant all on public.membu_chat_profiles
to service_role;


-- Считаем новые сообщения участников, но не сообщения самого Мембу.

create or replace function public.bump_membu_chat_profile_message_count()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
    if coalesce(new.sender_name, '') <> 'Мембу' then
        insert into public.membu_chat_profiles (
            chat_id,
            observed_messages,
            updated_at
        )
        values (
            new.chat_id,
            1,
            now()
        )
        on conflict (chat_id)
        do update set
            observed_messages =
                public.membu_chat_profiles.observed_messages + 1,
            updated_at = now();
    end if;

    return new;
end;
$$;


drop trigger if exists membu_count_chat_messages
on public.membu_messages;

create trigger membu_count_chat_messages
after insert on public.membu_messages
for each row
execute function public.bump_membu_chat_profile_message_count();


revoke all on function public.bump_membu_chat_profile_message_count()
from public, anon, authenticated;
