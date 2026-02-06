import sys
from typing import Any
from pathlib import Path
from openai import OpenAI
import os
import time

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

retries=3


def summarize_text(text: str, model_config_name: str = "x-ai/grok-4.1-fast") -> str:
    messages = [
        {
            "role": "system",
            "content": "You summarize long web page text to a factual summary for downstream QA. Preserve key facts, dates, numbers, names, and definitions. Write in the same language as the input. your summary should be as detailed as possible.",
        },
        {"role": "user", "content": text},
    ]

    api_key = os.getenv("OPENROUTER_API_KEY")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model_config_name,
                messages=messages,
                extra_body={"reasoning": {"effort": "minimal"}},
            )
            content = resp.choices[0].message.content
            return content
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(1 + attempt)
                continue
            raise

    if last_error is not None:
        raise last_error
    return ""


if __name__ == "__main__":
    raw = sys.stdin.read()
    sys.stdout.write(summarize_text(raw))
