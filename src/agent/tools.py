import asyncio
import functools
import json
import os
import sys
import threading
import time
import traceback
from typing import Any, Awaitable, Callable
from src.agent import content_summary
import tiktoken
import aiohttp
from loguru import logger
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt


def _count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def _truncate_tokens(text: str, max_tokens: int) -> str:
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.decode(enc.encode(text)[:max_tokens])


class RateLimiter:
    def __init__(self, interval: float):
        self.interval = interval
        self._lock = threading.Lock()
        self._next_time = 0.0

    async def wait(self):
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_time - now)
            self._next_time = max(self._next_time, now) + self.interval
        if wait > 0:
            await asyncio.sleep(wait)
_parallel_search_limiter = RateLimiter(interval=0.15)
_exa_contents_limiter = RateLimiter(interval=0.025)
_glm_web_search_concurrency = asyncio.Semaphore(40)


class InternalResponse(BaseModel):
    data: object | None = None
    """The data of the response."""

    error: str | None = None
    """The error message of the response."""

    system_error: str | None = None
    """The system error message of the response."""

    extra: dict | None = None


def return_error(error_msg: str, verbose: bool, req: str, context: str):
    warning_msg = f"req={req}, context={context}"
    logger.warning(f"error_msg={error_msg}, {warning_msg}")
    if not verbose:
        return error_msg
    else:
        return error_msg + f"\n{warning_msg}"


# search tools
def timeout_handler(timeout: int = 120):
    def decorator(
        func: Callable[..., Awaitable[InternalResponse]],
    ) -> Callable[..., Awaitable[InternalResponse]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                resp = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                return resp
            except asyncio.TimeoutError:
                return InternalResponse(error="TimeoutError")
            except Exception:
                return InternalResponse(system_error=traceback.format_exc())

        return wrapper

    return decorator


def abort_after_retries(retry_state):
    logger.error(f"Function {retry_state.fn.__name__} failed after {retry_state.attempt_number} retries. Aborting...")
    logger.error(f"Last exception: {retry_state.outcome.exception()}")
    sys.exit(1)


@timeout_handler(timeout=120)
@retry(stop=stop_after_attempt(3), retry_error_callback=abort_after_retries)
async def web_search_internationl(
    query: str,
):
    if not query:
        return InternalResponse(
            error=return_error(
                error_msg="error: query is empty",
                verbose=True,
                req=query,
                context="",
            )
        )

    api_key = os.getenv("PARALLEL_API_KEY")
    await _parallel_search_limiter.wait()

    payload = {
        "search_queries": [query],
        "max_results": 5,
        "excerpts": {"max_chars_per_result": 1000},
    }
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "parallel-beta": "search-extract-2025-10-10",
        "accept": "application/json",
    }
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=120)
    ) as session:
        async with session.post(
            "https://api.parallel.ai/v1beta/search",
            headers=headers,
            json=payload,
        ) as response:
            response.raise_for_status()
            result = await response.json()

    domain_filter = ["taiwan", "huggingface"]
    if isinstance(result, dict):
        items = result.get("results")
        if isinstance(items, list):
            filtered_items = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                url = item.get("url", "")
                if url and any(domain in url.lower() for domain in domain_filter):
                    continue
                filtered_items.append(item)
            result["results"] = filtered_items

    data = json.dumps(result, ensure_ascii=False, indent=2)
    return InternalResponse(
        data=data,
        extra={"content_lines": data.splitlines()},
    )

@timeout_handler(timeout=120)
@retry(stop=stop_after_attempt(3), retry_error_callback=abort_after_retries)
async def web_search_chinese(
    query: str,
):
    if not query:
        return InternalResponse(
            error=return_error(
                error_msg="error: query is empty",
                verbose=True,
                req=query,
                context="",
            )
        )

    api_key = os.getenv("GLM_API_KEY")

    payload = {
        "search_query": query,
        "search_engine": "search_std",
        "count": 5,
        "content_size": "medium"
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "accept": "application/json",
    }
    async with _glm_web_search_concurrency:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120)
        ) as session:
            async with session.post(
                "https://open.bigmodel.cn/api/paas/v4/web_search",
                headers=headers,
                json=payload,
            ) as response:
                response.raise_for_status()
                result = await response.json()

    domain_filter = ["taiwan", "huggingface"]
    if isinstance(result, dict):
        items = result.get("search_result")
        if isinstance(items, list):
            filtered_items = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                url = item.get("link", "")
                if url and any(domain in url.lower() for domain in domain_filter):
                    continue
                filtered_items.append(item)
            result["search_result"] = filtered_items

    data = json.dumps(result, ensure_ascii=False, indent=2)
    return InternalResponse(
        data=data,
        extra={"content_lines": data.splitlines()},
    )

@timeout_handler(timeout=120)
@retry(stop=stop_after_attempt(3), retry_error_callback=abort_after_retries)
async def get_content(url: str):
    if not url:
        return InternalResponse(error="error: url is empty")

    api_key = os.getenv("EXA_API_KEY")
    await _exa_contents_limiter.wait()

    payload = {
        "urls": [url],
        "extras": {
            "links": 0
        },
        "subpages": 0,
        "summary": True,
        "text": {
            "verbosity": "standard"
        }
    }
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "accept": "application/json",
    }

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=120)
    ) as session:
        async with session.post(
            "https://api.exa.ai/contents",
            headers=headers,
            json=payload,
        ) as response:
            response.raise_for_status()
            result = await response.json()

    if not isinstance(result, dict):
        return InternalResponse(error="Invalid response from Exa contents API")

    statuses = result.get("statuses") or []
    for status in statuses:
        if not isinstance(status, dict):
            continue
        if status.get("id") != url:
            continue
        if status.get("status") == "error":
            err = status.get("error") or {}
            tag = err.get("tag", "")
            http_code = err.get("httpStatusCode", "")
            return InternalResponse(error=f"Exa crawl error: {tag} ({http_code})")

    results = result.get("results") or []
    if not results or not isinstance(results, list) or not isinstance(results[0], dict):
        return InternalResponse(error="No content returned from Exa contents API")

    max_text_tokens = 5000
    for item in results:
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        token_count = _count_tokens(text)
        if token_count <= max_text_tokens:
            continue

        try:
            summary = await asyncio.to_thread(content_summary.summarize_text, text)
            item["text"] = summary
        except Exception:
            item["text"] = _truncate_tokens(text, max_text_tokens)
            logger.warning(
                f"content summary failed for url={url}, token_count={token_count}: {traceback.format_exc()}"
            )

    data = json.dumps(result, ensure_ascii=False, indent=2)
    return InternalResponse(
        data=data,
        extra={"content_lines": data.splitlines()},
    )


_default_tools = {
    "web_search_internationl": web_search_internationl,
    "web_search_chinese": web_search_chinese,
    "get_content": get_content,
}


#test get_content
async def test_get_content():
    """Test get_content function and print results to terminal."""
    test_url = "https://zh.wikipedia.org/wiki/%E9%9A%8B%E6%9C%9D"
    result = await get_content(test_url)
    print(f"Testing get_content with URL: {test_url}")
    print("=" * 80)
    print(result.data)

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    asyncio.run(test_get_content())
