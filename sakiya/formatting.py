import re

from .settings import settings


def sanitize_response(text: str) -> str:
    """Cleans up periods and stray quotes to match casual style."""
    if not text:
        return ""
    text = re.sub(r'(?i)^sakiya:\s*', '', text)
    text = text.replace('"', '')
    text = re.sub(r'<[^>]+>', '\n', text)
    text = re.sub(r'\n{3,}', '\n', text)

    if len(text) < 160:
        text = text.replace("...", "<ELLIPSIS>")
        text = re.sub(r'\.(?!\S)', '', text)
        text = text.replace("。", "")
        text = text.replace("<ELLIPSIS>", "...")

    final_text = text.strip()
    return final_text if final_text else "_ _"


def split_chunks(text: str) -> list[str]:
    size = settings.send_chunk_size
    return [text[i:i + size] for i in range(0, len(text), size)]
