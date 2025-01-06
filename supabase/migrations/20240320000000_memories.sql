create table memories (
    id int8 primary key,
    created_at timestamp with time zone default now(),
    user_id uuid references public.users(id),
    content text default ''::text,
    chunks text,
    embeddings vector,
    meeting_id uuid references public.late_meeting(id),
    centroid vector
);

-- Create foreign key constraints with explicit names
alter table memories
    add constraint memories_user_id_fkey 
    foreign key (user_id) 
    references public.users(id);

alter table memories
    add constraint memories_meeting_id_fkey 
    foreign key (meeting_id) 
    references public.late_meeting(id);

-- Create indexes for foreign keys
create index memories_user_id_idx on memories(user_id);
create index memories_meeting_id_idx on memories(meeting_id); 