-- MEMBU: мнения Мембу о людях. Запустить в Supabase SQL Editor.

create table if not exists public.membu_people (
    chat_id bigint not null,
    person_id bigint not null,
    display_name text not null default '',
    opinion text not null default '',
    message_count integer not null default 0,
    updated_at timestamptz not null default now(),
    primary key (chat_id, person_id)
);

create index if not exists membu_people_updated
on public.membu_people (updated_at desc);
