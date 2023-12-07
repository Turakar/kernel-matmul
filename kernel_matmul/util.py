from typing import Any


def format_dict(d: dict[str, Any]) -> str:
    content = ", ".join(f"{k}={v}" for k, v in d.items())
    return f"dict({content})"
