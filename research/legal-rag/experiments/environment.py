"""Load user-supplied credentials without copying or displaying their values."""
import os
from pathlib import Path

from dotenv import load_dotenv


def load_environment(path: Path) -> None:
    load_dotenv(path)
    if not os.environ.get("OPENAI_API_KEY") and os.environ.get("OPEN_AI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["OPEN_AI_API_KEY"]


def record_response(response, records: list) -> None:
    """Keep model/version and token accounting; never request headers or credentials."""
    response.read()
    try:
        payload = response.json()
    except ValueError:
        return
    records.append({k: payload.get(k) for k in
                    ("id", "model", "status", "usage", "incomplete_details", "created_at")})
