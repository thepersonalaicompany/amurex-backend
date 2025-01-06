create table meetings (
  id serial primary key,
  user_id uuid not null references auth.users(id),
  meeting_id text not null,
  transcript text,
  context_files text,
  created_at timestamp with time zone default timezone('utc'::text, now()),
  updated_at timestamp with time zone default timezone('utc'::text, now()),
  embeddings vector,
  generated_prompt jsonb,
  chunks text,
  suggestion_count int2 default '0'::smallint
);

-- Add a constraint name for the foreign key
alter table meetings add constraint meetings_user_id_fkey 
  foreign key (user_id) references auth.users(id);