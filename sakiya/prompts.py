import re

from .settings import FALLBACK_SYSTEM_PROMPT, settings

JAPANESE_PATTERN = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]')

ENGLISH_DIRECTIVE = "\n\n[CRITICAL DIRECTIVE: The user spoke English. You MUST reply in English.]"
JAPANESE_DIRECTIVE = "\n\n[CRITICAL DIRECTIVE: The user spoke Japanese. You MUST reply in Japanese.]"

_system_prompt = FALLBACK_SYSTEM_PROMPT


def load_system_prompt() -> str:
    global _system_prompt
    try:
        with open(settings.prompt_path, encoding="utf-8") as f:
            _system_prompt = f.read().strip()
        print(f"Successfully loaded system prompt from {settings.prompt_path}.")
    except FileNotFoundError:
        print(f"WARNING: {settings.prompt_path} not found. Using default system prompt.")
        _system_prompt = FALLBACK_SYSTEM_PROMPT
    return _system_prompt


def get_system_prompt() -> str:
    return _system_prompt


def is_japanese(text: str) -> bool:
    return bool(JAPANESE_PATTERN.search(text))


def language_directive(user_input: str) -> str:
    return JAPANESE_DIRECTIVE if is_japanese(user_input) else ENGLISH_DIRECTIVE


def display_name(author) -> str:
    """Falls back to the ASCII handle when a display name is written in Japanese."""
    name = author.display_name
    if is_japanese(name):
        name = author.name if not is_japanese(author.name) else "User"
    return name
