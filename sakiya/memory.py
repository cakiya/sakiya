"""Rolling conversation stores.

Always reach for these through the module (``memory.channel_memory``) rather than
importing the names directly, so a rebind here can never desync the callers.
"""

from .settings import settings

channel_memory: dict[int, list[dict[str, str]]] = {}
global_user_memory: dict[int, list[str]] = {}


def append_channel_message(channel_id: int, entry: dict[str, str]) -> None:
    channel_memory.setdefault(channel_id, []).append(entry)
    if len(channel_memory[channel_id]) > settings.max_channel_memory:
        channel_memory[channel_id].pop(0)


def replace_channel_history(channel_id: int, entries: list[dict[str, str]]) -> None:
    channel_memory[channel_id] = entries


def append_user_exchange(user_id: int, entry: str) -> None:
    global_user_memory.setdefault(user_id, []).append(entry)
    if len(global_user_memory[user_id]) > settings.max_user_memory:
        global_user_memory[user_id].pop(0)


def clear_channel_memory(channel_id: int) -> None:
    channel_memory[channel_id] = []


def clear_user_memory(user_id: int) -> None:
    if user_id in global_user_memory:
        global_user_memory[user_id] = []
