# mini

One assistant to chat with instead of juggling many Claude Code sessions. You
message mini (over Discord or its API), and it starts Claude Code sessions
("workers") in your projects, sends them follow-ups, checks on them, and
reports back when they're done.

- **Orchestrator:** an LLM agent (OpenAI by default, behind a provider-agnostic
  adapter) with tools to start, message, read, rename and search workers, and
  to reply in the chat, inline or in a thread.
- **Workers:** `claude -p` sessions running in a repo in your projects folder
  (or in the folder itself for general tasks), with extra MCP tools such as
  web browsing.
- **Routines:** instructions mini saves for itself and runs on a cron schedule
  or on request, e.g. a morning brief.
- **Attachments:** files go both ways; workers can read yours, and mini can
  send you theirs.

## Setup

Needs Python 3.12, [uv](https://docs.astral.sh/uv/) and the `claude` CLI
(logged in).

```
uv sync
```

`config.json` (or the file at `MINI_CONFIG`):

```json
{
    "projects_dir": "/path/to/your/repos",
    "discord": true
}
```

`.env`:

| Variable | |
|---|---|
| `MINI_API_TOKEN` | Bearer token for the API. Long and random. |
| `OPENAI_API_KEY` | For the orchestrator. `OPENAI_MODEL` optionally picks the model. |
| `DISCORD_BOT_TOKEN`, `DISCORD_CHANNEL_ID` | The bot and the channel mini lives in. |
| `BROWSER_USE_API_KEY` | For workers' web browsing tool. |
| `MINI_API_URL` | Optional, for the bot: where the API is (default `http://127.0.0.1:8000`). |
| `MINI_API_HOST`, `MINI_API_PORT` | Optional, for the API server (default `127.0.0.1:8000`). |

## Running

```
python -m mini            # API server + Discord bot, restarted if either exits
python -m mini.api        # just the API server
python -m mini.discord    # just the Discord bot
```

Data (the SQLite database and attachments) lives in `data/`. Migrations run on
startup.

## API

Every request needs `Authorization: Bearer <MINI_API_TOKEN>`.

- `POST /messages`: send mini a message (`content`, optional `parent_id` to
  reply in a thread, `attachment_ids`). Replies arrive on the event stream.
- `GET /events`: Server-Sent Events of every new chat message. Reconnect with
  `Last-Event-ID` (or `?after=<id>`) to catch up.
- `GET /messages?page=N`: the chat, newest first, 20 a page.
- `POST /attachments`, `GET /attachments/{id}`: upload and download files.

## Deploying

Pushing to `main` deploys to `miniapi.aidanba.com`: GitHub Actions SSHes into
the server with a key that can only run `deploy/deploy-mini`, which pulls,
syncs dependencies, waits for running workers and restarts the `mini` systemd
service (`deploy/mini.service`). Caddy proxies the domain to the API.
