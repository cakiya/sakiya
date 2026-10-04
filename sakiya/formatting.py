import re

from .settings import settings


def sanitize_response(text: str) -> str:
    """Cleans up periods and stray quotes to match casual style."""
    if not text:
        return ""
    text = re.sub(r'(?i)^sakiya:\s*', '', text)
    text = text.replace('"', '')
    text = re.sub(r'\n{3,}', '\n', text)

    if len(text) < 160:
        text = text.replace("...", "<ELLIPSIS>")
        text = re.sub(r'\.(?!\S)', '', text)
        text = text.replace("。", "")
        text = text.replace("<ELLIPSIS>", "...")
    return text.strip()


def split_chunks(text: str) -> list[str]:
    size = settings.send_chunk_size
    return [text[i:i + size] for i in range(0, len(text), size)]


def empty_check(text: str) -> str:
    """if the string is empty, return "_ _" to reply with nothing"""
    if text.strip() == "":
        return "_ _"
    return text
