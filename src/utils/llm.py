# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import copy
import os
import re
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterable, List, Optional, Union

import yaml
from loguru import logger
from openai import OpenAI
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from tenacity import retry, stop_after_attempt, wait_incrementing

from src.agent.schema import LLMOutputItem, ModelResponse, ToolCall

_MODELS_YAML_PATH = Path(__file__).resolve().parents[2] / "models.yaml"
_EVALUATORS_YAML_PATH = Path(__file__).resolve().parents[2] / "evaluators.yaml"
_YAML_CACHE: dict[Path, dict[str, Any]] = {}
_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")
_OPENAI_CLIENT_CACHE: dict[tuple[str, str | None], OpenAI] = {}

_TOKEN_USAGE_COLLECTOR: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "token_usage_collector", default=None
)


def set_token_usage_collector(collector: list[dict[str, Any]] | None):
    return _TOKEN_USAGE_COLLECTOR.set(collector)


def reset_token_usage_collector(token) -> None:
    _TOKEN_USAGE_COLLECTOR.reset(token)


def get_token_usage_collector() -> list[dict[str, Any]] | None:
    return _TOKEN_USAGE_COLLECTOR.get()


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    mtime = path.stat().st_mtime
    cached = _YAML_CACHE.get(path)
    if cached is not None and cached.get("mtime") == mtime:
        return cached["data"]

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid YAML format in {path}: root must be a dict, got {type(data)}"
        )

    _YAML_CACHE[path] = {"mtime": mtime, "data": data}
    return data


def _read_models_yaml() -> dict:
    return _read_yaml(_MODELS_YAML_PATH)


def _get_config_entry(path: Path, name: str) -> dict:
    data = _read_yaml(path)
    models = data.get("models", [])
    if not isinstance(models, list):
        raise ValueError(f"Invalid YAML format in {path}: `models` must be a list")
    for m in models:
        if isinstance(m, dict) and m.get("name") == name:
            return m
    raise KeyError(f"Config name {name} not found in {path}")


def get_model_config_no_key(config_name: str, is_evaluator: bool = False) -> dict:
    """Get model configuration without api_key."""
    path = _EVALUATORS_YAML_PATH if is_evaluator else _MODELS_YAML_PATH
    try:
        entry = _get_config_entry(path, config_name)
        # Deep copy to avoid modifying cache
        entry = copy.deepcopy(entry)
        entry.pop("api_key", None)
        return entry
    except (KeyError, FileNotFoundError, ValueError) as e:
        logger.warning(f"Failed to get config for {config_name}: {e}")
        return {"name": config_name, "error": str(e)}


def _iter_model_entries() -> list[dict]:
    data = _read_models_yaml()
    models = data.get("models", [])
    if not isinstance(models, list):
        raise ValueError("Invalid models.yaml format: `models` must be a list")
    out: list[dict] = []
    for m in models:
        if isinstance(m, dict):
            out.append(m)
    return out


def _resolve_env_placeholders(obj: Any) -> Any:
    if isinstance(obj, str):

        def repl(match: re.Match[str]) -> str:
            var = match.group(1)
            val = os.getenv(var)
            if val is None:
                raise ValueError(f"Missing required environment variable: {var}")
            return val

        return _ENV_PATTERN.sub(repl, obj)
    if isinstance(obj, list):
        return [_resolve_env_placeholders(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _resolve_env_placeholders(v) for k, v in obj.items()}
    return obj


def _get_model_entry(model_config_name: str) -> dict:
    for entry in _iter_model_entries():
        if entry.get("name") == model_config_name:
            return entry
    raise KeyError(f"model_config_name {model_config_name} not found in models.yaml")


def _get_model_config(model_config_name: str) -> dict:
    raw = _get_model_entry(model_config_name)
    entry = _resolve_env_placeholders(raw)

    if "name" not in entry:
        raise ValueError(f"Invalid model entry: missing `name` for {model_config_name}")
    if "base_url" not in entry:
        raise ValueError(
            f"Invalid model entry: missing `base_url` for {model_config_name}"
        )
    if "api_key" not in entry:
        raise ValueError(
            f"Invalid model entry: missing `api_key` for {model_config_name}"
        )

    generate_kwargs = {
        k: v
        for k, v in entry.items()
        if k
        not in {
            "name",
            "base_url",
            "api_key",
            "default_system_prompt",
            "is_claude_thinking",
        }
    }

    return {
        "model_name": entry["name"],
        "base_url": entry["base_url"],
        "api_key": entry.get("api_key"),
        "generate_kwargs": generate_kwargs,
        "default_system_prompt": entry.get("default_system_prompt", ""),
        "is_claude_thinking": bool(entry.get("is_claude_thinking", False)),
    }


def _add_prompt_caching(
    messages: List[dict], model_name: str
) -> List[dict]:
    """
    Add ephemeral caching to the most recent messages for Anthropic and Minimax models.
    """
    # Only apply caching for Anthropic models or Minimax models
    if not ("minimax" in model_name.lower() or "claude" in model_name.lower()):
        return messages

    # Create a deep copy to avoid modifying the original messages
    cached_messages = copy.deepcopy(messages)

    # Add cache_control to the most recent 3 messages
    for n in range(len(cached_messages)):
        if n >= len(cached_messages) - 3:
            msg = cached_messages[n]

            # Ensure content is in the expected format (list of dicts with type and text)
            if isinstance(msg.get("content"), str):
                msg["content"] = [
                    {
                        "type": "text",
                        "text": msg["content"],
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            elif isinstance(msg.get("content"), list):
                # Add cache_control to each content item
                for content_item in msg["content"]:
                    if isinstance(content_item, dict) and "type" in content_item:
                        content_item["cache_control"] = {"type": "ephemeral"}

    return cached_messages


def _get_openai_client(base_url: str, api_key: str | None) -> OpenAI:
    key = (base_url, api_key)
    cached = _OPENAI_CLIENT_CACHE.get(key)
    if cached is not None:
        return cached
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=300)
    _OPENAI_CLIENT_CACHE[key] = client
    return client


def abort_after_retries(retry_state):
    logger.error(f"Function {retry_state.fn.__name__} failed after {retry_state.attempt_number} retries. Aborting...")
    last_exc = retry_state.outcome.exception()
    logger.error(f"Last exception: {last_exc}")
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Function {retry_state.fn.__name__} failed after retries")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_incrementing(8, 8),
    retry_error_callback=abort_after_retries,
)
def openai_complete(
    base_url: str,
    api_key: Optional[str],
    messages: Iterable[dict],
    tools: Optional[Iterable[dict]] = None,
    model_name: str = "openai/gpt-5.2",
    **generate_kwargs,
) -> Optional[ChatCompletionMessage]:
    """Complete a prompt with OpenAI APIs."""

    openai_client = _get_openai_client(base_url=base_url, api_key=api_key)
    logger.debug(f"messages: {messages}")
    logger.debug(f"tools: {tools}")
    logger.debug(generate_kwargs)

    completion = openai_client.chat.completions.create(
        messages=messages,  # type: ignore
        model=model_name,
        tools=tools,  # type: ignore
        **generate_kwargs,
    )

    collector = get_token_usage_collector()
    if collector is not None:
        usage = getattr(completion, "usage", None)
        if usage is not None:
            usage_dict: dict[str, Any]
            if isinstance(usage, dict):
                usage_dict = copy.deepcopy(usage)
            elif hasattr(usage, "model_dump"):
                usage_dict = usage.model_dump()  # type: ignore[attr-defined]
            elif hasattr(usage, "dict"):
                usage_dict = usage.dict()  # type: ignore[attr-defined]
            else:
                try:
                    usage_dict = dict(usage)  # type: ignore[arg-type]
                except Exception:
                    usage_dict = getattr(usage, "__dict__", {}) or {"usage": str(usage)}

            usage_dict["model"] = model_name
            collector.append(usage_dict)

    return completion.choices[0].message


def get_is_claude_thinking(model_config_name: str) -> bool:
    return _get_model_config(model_config_name).get("is_claude_thinking", False)


def get_default_system_prompt_insert(model_config_name: str) -> str:
    return _get_model_config(model_config_name).get("default_system_prompt", "")


def llm_completion(
    messages: Union[str, List[dict]],
    tools: Optional[List[dict]] = None,
    model_config_name: str = "openai/gpt-5.2",
) -> Optional[ChatCompletionMessage]:
    """Complete a prompt with given LLM, raise error if the request failed."""

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    cfg = _get_model_config(model_config_name)
    model_name = cfg["model_name"]
    base_url = cfg["base_url"]
    api_key = cfg["api_key"]
    generate_kwargs = cfg.get("generate_kwargs", {})

    logger.debug(
        f"model_config_name: {model_config_name}, model_name: {model_name}, generate_kwargs: {generate_kwargs}"
    )

    # Add prompt caching for Anthropic and Minimax models
    messages = _add_prompt_caching(messages, model_config_name)

    response = openai_complete(
        base_url=base_url,
        api_key=api_key,
        messages=messages,
        tools=tools,
        model_name=model_name,
        **generate_kwargs,
    )

    return response


def transform_model_response(response: Any | None) -> ModelResponse:
    out = ModelResponse()
    if response is None:
        out.error_marker = {"message": "Calling LLM failed."}
        return out

    # Set fields.
    item = LLMOutputItem(content=response.content)
    # Convert into dict to get optional fields.
    resp_dict = response.model_dump()
    if resp_dict.get("reasoning_details") is not None:
        item.reasoning_details = resp_dict["reasoning_details"]
    elif resp_dict.get("reasoning_content"):
        item.reasoning_content = resp_dict["reasoning_content"]
    if resp_dict.get("signature"):
        item.signature = resp_dict["signature"]

    if response.tool_calls:
        item.tool_calls = []
        for tool_call in response.tool_calls:
            item.tool_calls.append(
                ToolCall(
                    tool_name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                    # TODO: Randomly generate the ID if not provided.
                    tool_call_id=tool_call.id,
                )
            )
    out.outputs.append(item)
    return out
