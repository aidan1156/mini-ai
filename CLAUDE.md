# mini

An agent layer that sits between the user and many Claude Code sessions (and
possibly Codex sessions later). The user chats with one orchestrator agent,
not with each coding session directly: they ask things like "how's the auth
refactor going?" or "can you get something to fix the flaky test?". The
orchestrator answers by reading the relevant session's transcript, or by
starting a session / sending an existing one a task.

## How it works

- **Workers** (`mini/workers/`) are Claude Code sessions run as `claude -p`
  subprocesses (stream-json output, `--permission-mode auto`). Starting a
  session or sending it a message returns immediately; the turn runs in a
  background asyncio task and every streamed message is saved to the database
  as it arrives. A conversation's `status` is `running` / `idle` / `error`.
- **Orchestrator** (planned): for each user message it gets
  - the recent orchestrator chat history (capped at N tokens),
  - the N most recently used worker conversations (id + description),

  and has tools to:
  - start a new worker conversation (prompt, description, cwd),
  - send a message to an existing conversation,
  - read a conversation's recent messages,
  - rename a conversation's description (keeps descriptions accurate as the
    work changes, since search depends on them),
  - search conversations by description to find older ones.

  When the orchestrator reads a transcript, it only gets the text messages:
  the prompts sent to the worker and the worker's text replies, like "I'll
  install x, then do this before finishing". Tool calls, tool results and raw
  JSON are left out, and the result is capped to the most recent N tokens.

## Layout

- `mini/database/models.py` holds the SQLAlchemy models.
- `mini/database/migrations/` holds the numbered `.sql` files, which are the
  source of truth for the schema.
- `mini/workers/worker.py` has `create_conversation`, `send_message` and
  `get_recent_messages`.
- `mini/orchestrator/worker_updates.py` decides whether a worker's reply gets
  posted to the user's chat, either in the owner message's thread
  (`WorkerMessage.owner_message_id`) or as a new top-level message.
- `mini/mcp_servers/` holds MCP servers that give workers extra tools (e.g.
  `browser.py`, which hands tasks to a Browser Use cloud agent, capped at $1
  a task). `worker_mcp_args()` returns the `claude` flags that load them,
  passed on every turn since MCP config isn't saved with the session.
- `database/database.db` is the SQLite database (`DATABASE_PATH`).
- `old/` is the previous version of the project (Gemini-based), kept for
  reference only.

## Conventions

- **Schema changes:** add a new numbered migration *and* update the model to
  match. Never call `Base.metadata.create_all()`, and never edit a migration
  that has already been applied to a real database.
- **Setup:** `python -m mini.database.migrate` applies pending migrations.
  `init_db()` also runs them and turns on SQLite foreign keys.
- **Session ids:** we generate each session id ourselves (uuid4) and pass it
  via `--session-id`. Resuming with `--resume` only works from the same cwd,
  which is why each conversation stores its `cwd`.
