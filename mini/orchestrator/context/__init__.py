"""Builds the text the orchestrator sees: chat history, worker transcripts, etc.

Use it as `from mini.orchestrator import context` and `context.chat_history(...)`.
"""

from mini.orchestrator.context.chat_history import (
    chat_history,
    describe_thread,
    label,
    render,
    strip_label,
    thread_has_replies,
)
from mini.orchestrator.context.conversations import (
    describe,
    owned_conversations,
    recent_conversations,
    run_conversations,
)
from mini.orchestrator.context.projects import project_names, project_path, projects
from mini.orchestrator.context.transcripts import transcript

__all__ = [
    "chat_history",
    "describe",
    "describe_thread",
    "label",
    "owned_conversations",
    "project_names",
    "project_path",
    "projects",
    "recent_conversations",
    "render",
    "run_conversations",
    "strip_label",
    "thread_has_replies",
    "transcript",
]
