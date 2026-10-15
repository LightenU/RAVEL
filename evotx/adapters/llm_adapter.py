from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple


# Official general-purpose APIs only.
# No Coding Plan endpoint and no Anthropic-compatible endpoint are used.
ZHIPU_GENERAL_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
MINIMAX_GENERAL_BASE_URL = "https://api.minimaxi.com/v1"

MINIMAX_MAX_COMPLETION_TOKENS = 32768


def resolve_adaptive_llm_route(
    *,
    model: Optional[str],
    provider: Optional[str],
    thinking: Optional[str],
    adaptive_model: Optional[str] = None,
    adaptive_provider: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve the optional second LLM tier for adaptive-thinking roles."""
    use_adaptive_tier = str(thinking or "").strip().lower() == "adaptive"
    resolved_model = adaptive_model if use_adaptive_tier and adaptive_model else model
    resolved_provider = (
        adaptive_provider
        if use_adaptive_tier and adaptive_provider
        else provider
    )
    return resolved_model, resolved_provider


class OpenAICompatibleLLM:
    """Minimal EvoTx adapter for GLM and MiniMax normal APIs.

    Both providers are accessed through their ordinary OpenAI-compatible
    Chat Completions endpoints. Coding Plan / coding-specific endpoints and
    Anthropic-compatible transports are intentionally not supported here.

    The public class name and main methods are kept compatible with the
    previous adapter so existing EvoTx call sites do not need to change.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        provider: Optional[str] = None,
        timeout: Optional[float] = None,
        max_tokens: Optional[int] = None,
        minimax_thinking: Optional[str] = None,
    ):
        try:
            from dotenv import load_dotenv
        except ImportError as exc:
            raise ImportError(
                "OpenAICompatibleLLM requires python-dotenv. "
                "Install the EvoTx runtime dependencies first."
            ) from exc

        try:
            from openai import DefaultHttpxClient, OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAICompatibleLLM requires the openai package. "
                "Install the EvoTx runtime dependencies first."
            ) from exc

        load_dotenv()

        self.temperature = temperature
        self.provider = _infer_provider(model=model, provider=provider)
        self.model = _resolve_model(model, self.provider)
        self.transport = "openai"
        self.openai_api_route = "chat_completions"

        self.api_key = api_key or _resolve_api_key(self.provider)
        if not self.api_key:
            raise ValueError(
                f"No API key found for provider '{self.provider}'. "
                "Set it in .env or pass api_key explicitly."
            )

        self.base_url = base_url or _resolve_base_url(self.provider)
        self.timeout = _resolve_timeout(timeout)
        self.max_tokens = _resolve_max_tokens(
            self.provider,
            max_tokens,
        )

        # Keep the old constructor argument name for compatibility with
        # existing EvoTx call sites. For GLM it is interpreted as a stage
        # thinking override; for MiniMax it controls MiniMax thinking.
        self.minimax_thinking = _resolve_minimax_thinking_type(
            provider=self.provider,
            value=minimax_thinking,
        )
        self.glm_thinking_override = _resolve_glm_stage_thinking_override(
            value=minimax_thinking,
        )
        self.stage_thinking = (
            self.glm_thinking_override
            if self.provider == "glm"
            else self.minimax_thinking
        )

        self.last_usage: Dict[str, Any] = {}
        self.last_thinking_metadata: Dict[str, Any] = {
            "thinking_present": None,
            "thinking_chars": None,
            "thinking_tokens_estimated": None,
            "thinking_tokens": None,
            "thinking_tokens_source": None,
        }
        self.last_finish_reason = ""
        self.last_request_config: Dict[str, Any] = {}

        # Preserve the old adapter's behavior for GLM/MiniMax: ignore shell
        # proxy variables and connect directly unless callers override the
        # networking outside this adapter.
        http_client = DefaultHttpxClient(trust_env=False)
        self.http_client_trust_env = bool(http_client.trust_env)
        self.network_route = "direct"
        self.network_route_scope = "application_http_client"
        self.application_proxy = "disabled"

        client_kwargs: Dict[str, Any] = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "http_client": http_client,
        }
        if self.timeout is not None:
            client_kwargs["timeout"] = self.timeout

        self.client = OpenAI(**client_kwargs)

        print(
            "[LLM RUNTIME NETWORK DEBUG]",
            "adapter_file=", __file__,
            "provider=", self.provider,
            "transport=", self.transport,
            "api_route=", self.openai_api_route,
            "base_url=", self.base_url,
            "trust_env=", self.http_client_trust_env,
            flush=True,
        )

    def clone_for_worker(self) -> "OpenAICompatibleLLM":
        """Return an independent adapter instance for concurrent judge workers."""
        return self.clone_with_max_tokens(self.max_tokens)

    def clone_with_max_tokens(
        self,
        max_tokens: Optional[int],
    ) -> "OpenAICompatibleLLM":
        """Return an equivalent adapter with a stage-specific output budget."""
        return OpenAICompatibleLLM(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            provider=self.provider,
            timeout=self.timeout,
            max_tokens=max_tokens,
            minimax_thinking=self.stage_thinking,
        )

    def complete(self, prompt: str) -> str:
        try:
            self._log_request_start(messages_count=1, tools=False)
            kwargs = self._openai_chat_kwargs(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                response_format=(
                    {"type": "json_object"}
                    if self.provider == "minimax"
                    else None
                ),
            )
            response = self._openai_chat_completion_create(**kwargs)
            self._record_response_metadata(response)
            return response.choices[0].message.content or ""
        except Exception as exc:
            print(
                "[LLM Adapter Error] complete failed: "
                f"{_exception_chain_summary(exc)}"
            )
            raise

    def generate(self, messages, tools=None):
        """Call GLM or MiniMax and normalize optional tool calls."""
        self._log_request_start(
            messages_count=len(messages),
            tools=bool(tools),
        )
        try:
            kwargs = self._openai_chat_kwargs(
                messages=messages,
                temperature=0.1,
                tools=tools,
            )
            response = self._openai_chat_completion_create(**kwargs)
            self._record_response_metadata(response)

            message = response.choices[0].message
            msg_dict = {
                "role": "assistant",
                "content": message.content or "",
            }

            if getattr(message, "tool_calls", None):
                print(
                    "[LLM] Tool calls: "
                    f"{[item.function.name for item in message.tool_calls]}"
                )
                msg_dict["tool_calls"] = [
                    {
                        "id": tool.id,
                        "type": tool.type,
                        "function": {
                            "name": tool.function.name,
                            "arguments": tool.function.arguments,
                        },
                    }
                    for tool in message.tool_calls
                ]

            return msg_dict
        except Exception as exc:
            print(
                "[LLM Adapter Error] Request failed: "
                f"{_exception_chain_summary(exc)}"
            )
            return {
                "role": "assistant",
                "content": f"LLM request failed: {exc}",
            }

    def simple_query(
        self,
        prompt,
        system_message="You are a helpful assistant.",
    ):
        """Small helper for direct JSON-ish LLM calls."""
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
        try:
            self._log_request_start(
                messages_count=len(messages),
                tools=False,
            )
            kwargs = self._openai_chat_kwargs(
                messages=messages,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            response = self._openai_chat_completion_create(**kwargs)
            self._record_response_metadata(response)
            content = response.choices[0].message.content or ""

            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return content
        except Exception as exc:
            print(
                "[LLM Adapter Error] simple_query failed: "
                f"{_exception_chain_summary(exc)}"
            )
            return {"error": str(exc)}

    def _openai_chat_completion_create(self, **kwargs):
        max_retries = _resolve_connection_retries(self.provider)
        retry_delay = _resolve_connection_retry_delay(self.provider)
        jitter_ratio = _resolve_connection_retry_jitter_ratio(self.provider)
        max_delay = _resolve_connection_retry_max_delay(self.provider)

        for attempt in range(max_retries + 1):
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as exc:
                retryable = _is_retryable_openai_connection_error(exc)
                if attempt >= max_retries or not retryable:
                    print(
                        "[LLM] OpenAI-compatible request failed: "
                        f"request_attempt={attempt + 1}/{max_retries + 1} "
                        f"provider={self.provider} "
                        f"retryable={retryable} "
                        f"retry_exhausted={retryable and attempt >= max_retries} "
                        f"error_chain={_exception_chain_summary(exc)}"
                    )
                    raise

                wait_seconds = _connection_retry_wait_seconds(
                    initial_delay_seconds=retry_delay,
                    retry_index=attempt,
                    jitter_ratio=jitter_ratio,
                    max_delay_seconds=max_delay,
                )
                print(
                    "[LLM] Retrying OpenAI-compatible request after "
                    f"{type(exc).__name__}: "
                    f"adapter_retry={attempt + 1}/{max_retries} "
                    f"provider={self.provider} "
                    f"wait_seconds={wait_seconds:.2f}"
                )
                time.sleep(wait_seconds)

        raise RuntimeError("Unreachable retry state")

    def _openai_chat_kwargs(
        self,
        *,
        messages,
        temperature: Optional[float] = None,
        tools=None,
        response_format: Optional[dict] = None,
        extra_body: Optional[dict] = None,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": (
                self.temperature
                if temperature is None
                else temperature
            ),
        }

        if self.max_tokens is not None:
            if self.provider == "minimax":
                kwargs["max_completion_tokens"] = self.max_tokens
            else:
                kwargs["max_tokens"] = self.max_tokens

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        if response_format is not None:
            kwargs["response_format"] = response_format

        merged_extra_body = dict(extra_body or {})

        if self.provider == "minimax":
            # Keep MiniMax reasoning separated from final content when supported.
            merged_extra_body.setdefault("reasoning_split", True)
            thinking_type = str(
                getattr(self, "minimax_thinking", "") or ""
            ).strip().lower()
            if thinking_type and thinking_type != "default":
                merged_extra_body["thinking"] = {"type": thinking_type}

        if self.provider == "glm":
            thinking_type = _resolve_glm_thinking_type(
                override=getattr(
                    self,
                    "glm_thinking_override",
                    "",
                ),
            )
            if thinking_type:
                merged_extra_body["thinking"] = {
                    "type": thinking_type
                }

        if merged_extra_body:
            kwargs["extra_body"] = merged_extra_body

        self.last_request_config = {
            **self.config_summary(),
            "temperature": kwargs.get("temperature"),
            "tools": bool(tools),
            "response_format": response_format,
        }
        return kwargs

    def _record_response_metadata(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        self.last_usage = _usage_to_dict(usage)
        self.last_thinking_metadata = _response_thinking_metadata(
            response,
            normalized_usage=self.last_usage,
        )

        self.last_usage["thinking_present"] = (
            self.last_thinking_metadata["thinking_present"]
        )
        self.last_usage["thinking_chars"] = (
            self.last_thinking_metadata["thinking_chars"]
        )
        self.last_usage["thinking_tokens_estimated"] = (
            self.last_thinking_metadata["thinking_tokens_estimated"]
        )
        self.last_usage["thinking_tokens"] = (
            self.last_thinking_metadata["thinking_tokens"]
        )
        self.last_usage["thinking_tokens_source"] = (
            self.last_thinking_metadata["thinking_tokens_source"]
        )

        if "reasoning_tokens" not in self.last_usage:
            self.last_usage["reasoning_tokens"] = (
                self.last_thinking_metadata["thinking_tokens"]
            )

        self.last_finish_reason = _openai_finish_reason(response)

        if self.last_usage:
            extras = []
            for key in ("reasoning_tokens",):
                value = self.last_usage.get(key)
                if value is not None:
                    extras.append(f"{key}={value}")

            extras.append(
                "thinking_present="
                f"{self.last_thinking_metadata.get('thinking_present')}"
            )

            thinking_chars = self.last_thinking_metadata.get(
                "thinking_chars"
            )
            if thinking_chars is not None:
                extras.append(f"thinking_chars={thinking_chars}")

            estimated_tokens = self.last_thinking_metadata.get(
                "thinking_tokens_estimated"
            )
            if estimated_tokens is not None:
                extras.append(
                    f"thinking_tokens_estimated={estimated_tokens}"
                )

            extra_text = (
                f", {', '.join(extras)}"
                if extras
                else ""
            )
            print(
                "[LLM] Response usage: "
                f"prompt_tokens={self.last_usage.get('prompt_tokens')}, "
                f"completion_tokens={self.last_usage.get('completion_tokens')}, "
                f"total={self.last_usage.get('total_tokens')}"
                f"{extra_text}"
            )

    def config_summary(self) -> Dict[str, Any]:
        return describe_llm_config(
            model=self.model,
            provider=self.provider,
            base_url=self.base_url,
            timeout=self.timeout,
            max_tokens=self.max_tokens,
            minimax_thinking=self.stage_thinking,
        )

    def _log_request_start(
        self,
        *,
        messages_count: int,
        tools: bool,
    ) -> None:
        config = self.config_summary()
        thinking_display: Any = config.get("thinking") or "provider_default"

        print(
            "[LLM] Sending request: "
            f"messages={messages_count} "
            f"tools={'yes' if tools else 'no'} "
            f"provider={config.get('provider')} "
            f"transport={config.get('transport')} "
            f"api_route={config.get('api_route')} "
            f"model={config.get('model')} "
            f"base_url={config.get('base_url')} "
            f"max_tokens={config.get('max_tokens')} "
            f"thinking={thinking_display} "
            f"timeout={config.get('timeout_seconds')} "
            "network_route=direct "
            "application_proxy=disabled"
        )


def _infer_provider(
    model: Optional[str],
    provider: Optional[str] = None,
) -> str:
    configured = (
        provider
        or os.getenv("EVOTX_LLM_PROVIDER")
        or os.getenv("LLM_PROVIDER")
        or ""
    ).strip().lower()

    aliases = {
        "glm": "glm",
        "zhipu": "glm",
        "bigmodel": "glm",
        "minimax": "minimax",
        "minimaxi": "minimax",
    }

    if configured:
        resolved = aliases.get(configured)
        if not resolved:
            raise ValueError(
                f"Unsupported LLM provider '{configured}'. "
                "Only 'glm' and 'minimax' are supported."
            )
        return resolved

    model_lower = str(model or "").strip().lower()
    if "minimax" in model_lower or model_lower.startswith("abab"):
        return "minimax"
    if "glm" in model_lower or "zhipu" in model_lower:
        return "glm"

    raise ValueError(
        "Unable to infer provider. Set provider='glm' or provider='minimax', "
        "or use a model name containing 'glm'/'minimax'."
    )


def describe_llm_config(
    *,
    model: Optional[str],
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: Optional[float] = None,
    max_tokens: Optional[int] = None,
    minimax_thinking: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_provider = _infer_provider(
        model=model,
        provider=provider,
    )
    resolved_model = _resolve_model(
        model,
        resolved_provider,
    )
    resolved_base_url = (
        base_url
        or _resolve_base_url(resolved_provider)
    )
    resolved_timeout = _resolve_timeout(timeout)
    resolved_max_tokens = _resolve_max_tokens(
        resolved_provider,
        max_tokens,
    )

    thinking: Dict[str, Any] = {}

    if resolved_provider == "glm":
        override = _resolve_glm_stage_thinking_override(
            minimax_thinking
        )
        thinking_type = _resolve_glm_thinking_type(
            override=override
        )
        if thinking_type:
            thinking = {
                "transport": "openai",
                "type": thinking_type,
            }

    elif resolved_provider == "minimax":
        minimax_thinking = _resolve_minimax_thinking_type(
            provider="minimax",
            value=minimax_thinking,
        )
        if minimax_thinking and minimax_thinking != "default":
            thinking = {
                "transport": "openai",
                "type": minimax_thinking,
            }

    return {
        "provider": resolved_provider,
        "model": resolved_model,
        "transport": "openai",
        "api_route": "chat_completions",
        "base_url": resolved_base_url,
        "timeout_seconds": resolved_timeout,
        "max_tokens": resolved_max_tokens,
        "network_route": "direct",
        "network_route_scope": "application_http_client",
        "application_proxy": "disabled",
        "http_client_trust_env": False,
        "thinking": thinking,
        "connection_retries": _resolve_connection_retries(
            resolved_provider
        ),
        "connection_retry_delay_seconds": (
            _resolve_connection_retry_delay(resolved_provider)
        ),
    }


def _resolve_model(
    model: Optional[str],
    provider: str,
) -> str:
    model_text = str(model or "").strip()
    if model_text:
        return model_text

    if provider == "glm":
        return (
            os.getenv("ZHIPU_MODEL")
            or os.getenv("GLM_MODEL")
            or "glm-5.1"
        )

    if provider == "minimax":
        return (
            os.getenv("MINIMAX_MODEL")
            or "abab6.5s-chat"
        )

    raise ValueError(
        f"Unsupported LLM provider '{provider}'."
    )


def _resolve_api_key(provider: str) -> Optional[str]:
    if provider == "glm":
        return (
            os.getenv("ZHIPU_API_KEY")
            or os.getenv("GLM_API_KEY")
        )

    if provider == "minimax":
        return (
            os.getenv("MINIMAX_API_KEY")
            or os.getenv("MINIMAX_API_TOKEN")
        )

    return None


def _resolve_base_url(provider: str) -> str:
    if provider == "glm":
        return (
            os.getenv("ZHIPU_BASE_URL")
            or os.getenv("GLM_BASE_URL")
            or ZHIPU_GENERAL_BASE_URL
        )

    if provider == "minimax":
        return (
            os.getenv("MINIMAX_BASE_URL")
            or MINIMAX_GENERAL_BASE_URL
        )

    raise ValueError(
        f"Unsupported LLM provider '{provider}'."
    )


def _resolve_timeout(
    value: Optional[float],
) -> Optional[float]:
    if value is not None:
        return max(1.0, float(value))

    raw = str(
        os.getenv("LLM_TIMEOUT_SECONDS") or ""
    ).strip()
    if raw:
        return max(1.0, float(raw))

    return None


def _resolve_max_tokens(
    provider: str,
    value: Optional[int],
) -> Optional[int]:
    if value is None:
        env_name = (
            "GLM_MAX_TOKENS"
            if provider == "glm"
            else "MINIMAX_MAX_TOKENS"
        )
        raw = str(os.getenv(env_name) or "").strip()
        if not raw:
            return None
        value = int(raw)

    resolved = max(1, int(value))
    if provider == "minimax":
        return min(
            resolved,
            MINIMAX_MAX_COMPLETION_TOKENS,
        )
    return resolved


def _resolve_glm_stage_thinking_override(
    value: Optional[str],
) -> str:
    configured = str(value or "").strip().lower()
    aliases = {
        "enabled": "enabled",
        "on": "enabled",
        "true": "enabled",
        "adaptive": "",
        "default": "default",
        "provider-default": "default",
        "provider_default": "default",
        "disabled": "disabled",
        "off": "disabled",
        "false": "disabled",
    }
    return aliases.get(configured, "")


def _resolve_glm_thinking_type(
    *,
    override: str = "",
) -> str:
    override = str(override or "").strip().lower()

    if override == "default":
        return ""
    if override in {"enabled", "disabled"}:
        return override

    value = str(
        os.getenv("GLM_THINKING") or ""
    ).strip().lower()
    if value in {"enabled", "disabled"}:
        return value

    return ""


def _resolve_minimax_thinking_type(
    *,
    provider: str,
    value: Optional[str],
) -> str:
    if provider != "minimax":
        return ""

    requested = str(value or "").strip().lower()
    if requested in {
        "default",
        "provider-default",
        "provider_default",
    }:
        return "default"

    override = str(
        os.getenv("MINIMAX_THINKING_OVERRIDE") or ""
    ).strip()

    configured = str(
        override
        or (
            value
            if value is not None
            else os.getenv("MINIMAX_THINKING") or ""
        )
    ).strip().lower()

    aliases = {
        "enabled": "adaptive",
        "on": "adaptive",
        "true": "adaptive",
        "off": "disabled",
        "false": "disabled",
    }
    configured = aliases.get(configured, configured)

    if configured not in {
        "adaptive",
        "disabled",
        "default",
    }:
        return ""

    return configured


def _resolve_connection_retries(provider: str) -> int:
    raw = str(
        os.getenv(
            f"{provider.upper()}_CONNECTION_RETRIES"
        )
        or ""
    ).strip()
    return max(0, int(raw)) if raw else 0


def _resolve_connection_retry_delay(
    provider: str,
) -> float:
    raw = str(
        os.getenv(
            f"{provider.upper()}_CONNECTION_RETRY_DELAY_SECONDS"
        )
        or ""
    ).strip()
    return max(0.1, float(raw)) if raw else 1.0


def _resolve_connection_retry_jitter_ratio(
    provider: str,
) -> float:
    raw = str(
        os.getenv(
            f"{provider.upper()}_CONNECTION_RETRY_JITTER_RATIO"
        )
        or ""
    ).strip()
    if not raw:
        return 0.0
    return min(1.0, max(0.0, float(raw)))


def _resolve_connection_retry_max_delay(
    provider: str,
) -> float:
    raw = str(
        os.getenv(
            f"{provider.upper()}_CONNECTION_RETRY_MAX_DELAY_SECONDS"
        )
        or ""
    ).strip()
    return max(0.1, float(raw)) if raw else 60.0


def _connection_retry_wait_seconds(
    *,
    initial_delay_seconds: float,
    retry_index: int,
    jitter_ratio: float,
    max_delay_seconds: float,
) -> float:
    capped_delay = min(
        max(0.1, float(max_delay_seconds)),
        max(0.1, float(initial_delay_seconds))
        * (2 ** max(0, retry_index)),
    )
    ratio = min(
        1.0,
        max(0.0, float(jitter_ratio)),
    )
    if ratio <= 0:
        return capped_delay

    jitter = random.uniform(
        -capped_delay * ratio,
        capped_delay * ratio,
    )
    return max(
        0.1,
        min(
            max_delay_seconds,
            capped_delay + jitter,
        ),
    )


def _exception_chain_summary(
    exc: BaseException,
    max_depth: int = 6,
) -> str:
    parts: List[str] = []
    current: Optional[BaseException] = exc
    depth = 0
    seen = set()

    while current is not None and depth < max_depth:
        obj_id = id(current)
        if obj_id in seen:
            break
        seen.add(obj_id)

        message = str(current).strip()
        parts.append(
            f"{type(current).__name__}"
            + (f": {message}" if message else "")
        )

        current = (
            current.__cause__
            if current.__cause__ is not None
            else current.__context__
        )
        depth += 1

    return " <- ".join(parts)


def _is_retryable_openai_connection_error(
    exc: Exception,
) -> bool:
    try:
        from openai import (
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
        )

        return isinstance(
            exc,
            (
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
                RateLimitError,
            ),
        )
    except ImportError:
        return False


def _openai_finish_reason(response: Any) -> str:
    choices = list(
        getattr(response, "choices", []) or []
    )
    if not choices:
        return ""
    return str(
        getattr(
            choices[0],
            "finish_reason",
            "",
        )
        or ""
    )


def _usage_detail_value(
    details: Any,
    name: str,
) -> Any:
    if details is None:
        return None

    value = getattr(details, name, None)
    if value is None and isinstance(details, dict):
        value = details.get(name)

    return value


def _usage_to_dict(usage) -> dict:
    if not usage:
        return {}

    prompt_tokens = getattr(
        usage,
        "prompt_tokens",
        None,
    )
    completion_tokens = getattr(
        usage,
        "completion_tokens",
        None,
    )
    total_tokens = getattr(
        usage,
        "total_tokens",
        None,
    )

    if (
        total_tokens is None
        and prompt_tokens is not None
        and completion_tokens is not None
    ):
        total_tokens = (
            int(prompt_tokens)
            + int(completion_tokens)
        )

    result = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }

    completion_details = getattr(
        usage,
        "completion_tokens_details",
        None,
    )
    reasoning_tokens = _usage_detail_value(
        completion_details,
        "reasoning_tokens",
    )

    if reasoning_tokens is not None:
        result["reasoning_tokens"] = reasoning_tokens
        result["reasoning_tokens_source"] = (
            "usage.completion_tokens_details.reasoning_tokens"
        )

    return result


def _response_thinking_metadata(
    response: Any,
    *,
    normalized_usage: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return observable reasoning metadata without exposing reasoning text."""
    usage = dict(normalized_usage or {})
    thinking_tokens = usage.get("reasoning_tokens")
    thinking_tokens_source = usage.get(
        "reasoning_tokens_source"
    )

    thinking_present: Optional[bool] = None
    thinking_chars: Optional[int] = None

    choices = list(
        getattr(response, "choices", []) or []
    )
    if choices:
        message = getattr(
            choices[0],
            "message",
            None,
        )
        if message is not None:
            sentinel = object()

            if isinstance(message, dict):
                reasoning_content = message.get(
                    "reasoning_content",
                    sentinel,
                )
                reasoning_details = message.get(
                    "reasoning_details",
                    sentinel,
                )
            else:
                reasoning_content = getattr(
                    message,
                    "reasoning_content",
                    sentinel,
                )
                reasoning_details = getattr(
                    message,
                    "reasoning_details",
                    sentinel,
                )

            exposed = (
                reasoning_content is not sentinel
                or reasoning_details is not sentinel
                or thinking_tokens is not None
            )

            if exposed:
                thinking_present = False
                thinking_chars = 0

            if (
                reasoning_content is not sentinel
                and str(reasoning_content or "").strip()
            ):
                thinking_present = True
                thinking_chars = len(
                    str(reasoning_content or "")
                )

            if (
                reasoning_details is not sentinel
                and reasoning_details
            ):
                thinking_present = True
                if thinking_chars is None:
                    thinking_chars = 0
                thinking_chars += len(
                    json.dumps(
                        reasoning_details,
                        ensure_ascii=False,
                        default=str,
                    )
                )

    if thinking_tokens is not None:
        if int(thinking_tokens) > 0:
            thinking_present = True
        elif thinking_present is None:
            thinking_present = False

    return {
        "thinking_present": thinking_present,
        "thinking_chars": thinking_chars,
        "thinking_tokens_estimated": (
            (thinking_chars + 3) // 4
            if thinking_chars is not None
            and thinking_chars > 0
            else 0
            if thinking_chars == 0
            else None
        ),
        "thinking_tokens": thinking_tokens,
        "thinking_tokens_source": thinking_tokens_source,
    }
