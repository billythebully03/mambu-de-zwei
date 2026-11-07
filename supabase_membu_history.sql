-- MEMBU: поиск по истории чата и по памяти с учётом русских окончаний,
-- плюс учёт дайджеста памяти. Запустить в Supabase SQL Editor
-- после supabase.sql и supabase_chat_style.sql.

-- 1. Основа слова: первые 5 букв каждого слова (3+ букв).
--    «обсуждали», «обсуждение» и «обсудить» дают одну основу «обсуд».

create or replace function public.membu_query_terms(p_query text)
returns text[]
language sql
immutable
as $$
    select coalesce(array_agg(left(w, 5)), '{}'::text[])
    from (
        select regexp_replace(lower(t), '[^[:alnum:]]', '', 'g') as w
        from regexp_split_to_table(coalesce(p_query, ''), '\s+') as t
    ) s
    where length(w) >= 3;
$$;


-- 2. Поиск по всей истории чата (и людей, и Мембу).
--    Все основы должны встречаться в сообщении.
--    p_before позволяет листать назад по времени.

create or replace function public.search_membu_history(
    p_chat_id bigint,
    p_query text,
    p_limit integer default 8,
    p_before timestamptz default null
)
returns table (
    message_id bigint,
    sender_name text,
    text text,
    created_at timestamptz
)
language plpgsql
stable
security definer
set search_path = public
as $$
declare
    terms text[] := public.membu_query_terms(p_query);
begin
    if coalesce(array_length(terms, 1), 0) = 0 then
        return;
    end if;

    return query
    select m.message_id, m.sender_name, m.text, m.created_at
    from public.membu_messages m
    where m.chat_id = p_chat_id
      and (p_before is null or m.created_at < p_before)
      and not exists (
          select 1
          from unnest(terms) as t
          where lower(m.text) not like '%' || t || '%'
      )
    order by m.created_at desc
    limit greatest(1, least(coalesce(p_limit, 8), 20));
end;
$$;


-- 3. Поиск по долговременной памяти: тот же контракт, что и раньше,
--    но с учётом окончаний. Сортировка по важности.

create or replace function public.search_membu_memory(
    p_chat_id bigint,
    p_query text,
    p_category text default null,
    p_limit integer default 5
)
returns table (
    id bigint,
    category text,
    content text,
    importance integer,
    people text[]
)
language plpgsql
stable
security definer
set search_path = public
as $$
declare
    terms text[] := public.membu_query_terms(p_query);
begin
    if coalesce(array_length(terms, 1), 0) = 0 then
        return;
    end if;

    return query
    select m.id, m.category, m.content, m.importance, m.people
    from public.membu_memories m
    where m.chat_id = p_chat_id
      and (p_category is null or m.category = p_category)
      and not exists (
          select 1
          from unnest(terms) as t
          where lower(m.content || ' ' || array_to_string(m.people, ' '))
                not like '%' || t || '%'
      )
    order by m.importance desc, m.updated_at desc
    limit greatest(1, least(coalesce(p_limit, 5), 8));
end;
$$;


-- 4. Дайджест: сколько сообщений уже превращено в память.

alter table public.membu_chat_profiles
add column if not exists last_digested_count bigint not null default 0;


-- 5. Права: только service_role, как и остальные функции.

revoke all on function public.membu_query_terms(text)
from public, anon, authenticated;

revoke all on function public.search_membu_history(bigint, text, integer, timestamptz)
from public, anon, authenticated;

grant execute on function public.search_membu_history(bigint, text, integer, timestamptz)
to service_role;

revoke all on function public.search_membu_memory(bigint, text, text, integer)
from public, anon, authenticated;

grant execute on function public.search_membu_memory(bigint, text, text, integer)
to service_role;
