from typing import List

def safe_truncate(text: str, n: int = 1200) -> str:
    if len(text) <= n:
        return text
    return text[:n] + "…"
