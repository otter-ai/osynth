"""Unified async LLM client supporting Anthropic, OpenAI, and Google providers.

Automatically selects the provider based on model name:
  - ``gpt-*`` models → OpenAI
  - ``gemini-*`` models → Google (requires ``GOOGLE_API_KEY``)
  - ``claude-*`` / everything else → Anthropic
"""
from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
"""Reasoning-effort values available to compatible provider models.

Accepted values depend on the selected model. This client forwards the value
without translation; unsupported values are rejected by the provider.
"""

logger = logging.getLogger(__name__)


_TRUNCATION_STOP_REASONS = {"max_tokens", "length"}


@dataclass
class LLMResponse:
    """Provider-agnostic LLM response."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    model: str = ""
    provider: str = ""
    # Normalised provider stop / finish reason. Anthropic emits values like
    # ``end_turn`` / ``max_tokens`` / ``stop_sequence``; OpenAI emits
    # ``stop`` / ``length`` / ``content_filter``. Empty string when unknown.
    stop_reason: str = ""

    @property
    def truncated(self) -> bool:
        """True when the provider stopped because the output budget was reached."""
        return self.stop_reason in _TRUNCATION_STOP_REASONS


DEFAULT_MAX_OUTPUT_TOKENS = 8192


class Step1Budgets(BaseModel):
    """Output-token budgets for step 1 (workspace) LLM calls."""

    model_config = ConfigDict(extra="forbid")
    company: int = DEFAULT_MAX_OUTPUT_TOKENS
    org_spine: int = DEFAULT_MAX_OUTPUT_TOKENS
    team_people_sketch: int = DEFAULT_MAX_OUTPUT_TOKENS
    person_detail: int = DEFAULT_MAX_OUTPUT_TOKENS
    external_person_detail: int = DEFAULT_MAX_OUTPUT_TOKENS
    project_index: int = DEFAULT_MAX_OUTPUT_TOKENS
    project_cluster_sketch: int = DEFAULT_MAX_OUTPUT_TOKENS
    project_detail: int = DEFAULT_MAX_OUTPUT_TOKENS
    recurring_meetings_company: int = DEFAULT_MAX_OUTPUT_TOKENS
    recurring_meetings_team: int = DEFAULT_MAX_OUTPUT_TOKENS
    recurring_meetings_cross: int = DEFAULT_MAX_OUTPUT_TOKENS


class Step2Budgets(BaseModel):
    """Output-token budgets for step 2 (time period) LLM calls."""

    model_config = ConfigDict(extra="forbid")
    narrative: int = DEFAULT_MAX_OUTPUT_TOKENS
    week_frame: int = DEFAULT_MAX_OUTPUT_TOKENS
    week_meetings: int = DEFAULT_MAX_OUTPUT_TOKENS
    week_emails: int = DEFAULT_MAX_OUTPUT_TOKENS
    week_documents: int = DEFAULT_MAX_OUTPUT_TOKENS
    week_slides: int = DEFAULT_MAX_OUTPUT_TOKENS


class Step4Budgets(BaseModel):
    """Output-token budgets for step 4 (artifacts) LLM calls."""

    model_config = ConfigDict(extra="forbid")
    meeting_overview: int = DEFAULT_MAX_OUTPUT_TOKENS
    meeting_transcript: int = DEFAULT_MAX_OUTPUT_TOKENS
    meeting_event_extraction: int = DEFAULT_MAX_OUTPUT_TOKENS
    email_message: int = DEFAULT_MAX_OUTPUT_TOKENS
    document: int = DEFAULT_MAX_OUTPUT_TOKENS
    slides: int = DEFAULT_MAX_OUTPUT_TOKENS


_STEP_MODELS: dict[str, type[BaseModel]] = {
    "step_1": Step1Budgets,
    "step_2": Step2Budgets,
    "step_4": Step4Budgets,
}


class LLMBudgetConfig(BaseModel):
    """Per-call LLM output-token budgets, organized by pipeline step.

    Configs supply only the per-call overrides they care about; every other
    field falls back to ``default_max_output_tokens`` (which itself defaults
    to ``DEFAULT_MAX_OUTPUT_TOKENS``).
    """

    model_config = ConfigDict(extra="forbid")
    default_max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    step_1: Step1Budgets = Field(default_factory=Step1Budgets)
    step_2: Step2Budgets = Field(default_factory=Step2Budgets)
    step_4: Step4Budgets = Field(default_factory=Step4Budgets)

    @classmethod
    def from_settings(cls, settings: Any) -> "LLMBudgetConfig":
        """Build a budget config from a Dynaconf-style settings object.

        Reads the optional ``generation.llm`` block. Step fields not
        explicitly set in the YAML are filled with ``default_max_output_tokens``
        so call sites can always read a concrete int.
        """
        raw = _read_llm_block(settings)
        default = int(raw.get("default_max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS))

        step_kwargs: dict[str, Any] = {"default_max_output_tokens": default}
        for name, step_cls in _STEP_MODELS.items():
            overrides = raw.get(name) or {}
            merged = {field: default for field in step_cls.model_fields}
            merged.update(overrides)
            step_kwargs[name] = step_cls(**merged)
        return cls(**step_kwargs)


def _read_llm_block(settings: Any) -> dict[str, Any]:
    """Pull the ``generation.llm`` config out of a settings object as a plain dict."""
    if isinstance(settings, BaseModel):
        return _read_llm_block(settings.model_dump(mode="python"))
    if not isinstance(settings, Mapping):
        try:
            settings = settings.to_dict()
        except AttributeError:
            return {}
    generation = settings.get("generation")
    if not isinstance(generation, Mapping):
        return {}
    llm = generation.get("llm")
    if llm is None:
        return {}
    if isinstance(llm, Mapping):
        return dict(llm)
    if isinstance(llm, BaseModel):
        return llm.model_dump(mode="python")
    try:
        return dict(llm.to_dict())
    except AttributeError:
        return {}


def detect_provider(model: str) -> str:
    """Return ``'openai'``, ``'google'``, or ``'anthropic'`` based on model prefix."""
    if model.startswith("gpt-"):
        return "openai"
    if model.startswith("gemini-"):
        return "google"
    return "anthropic"


# OpenAI reasoning model families. These models:
#   * Reject any temperature other than the default (1.0); supplying one
#     yields ``400 BadRequest: 'temperature' does not support X with this
#     model. Only the default (1) value is supported.``
#   * Accept an optional ``reasoning_effort`` request parameter
#     (``minimal`` / ``low`` / ``medium`` / ``high``) that controls how many
#     hidden reasoning tokens the model spends before producing visible
#     output. Hidden reasoning tokens are billed against
#     ``max_completion_tokens``, so leaving the API default (``medium``)
#     can silently exhaust the budget and return empty content.
# Match by prefix so future point releases inherit
# the behavior automatically. Only ``gpt-*`` models route through the
# OpenAI client, so the prefix list is restricted to those.
_OPENAI_REASONING_PREFIXES: tuple[str, ...] = (
    "gpt-5",
    "gpt-6",
)


def model_supports_custom_temperature(model: str) -> bool:
    """Return False for OpenAI models that only accept the default temperature.

    Anthropic and other providers always accept custom temperatures, so this
    helper only inspects OpenAI-routed models.
    """
    if detect_provider(model) != "openai":
        return True
    return not any(model.startswith(prefix) for prefix in _OPENAI_REASONING_PREFIXES)


def model_uses_reasoning(model: str) -> bool:
    """Return True for OpenAI reasoning models that accept ``reasoning_effort``.

    Anthropic and other providers do not expose the OpenAI ``reasoning_effort``
    request parameter, so this helper only inspects OpenAI-routed models.
    """
    if detect_provider(model) != "openai":
        return False
    return any(model.startswith(prefix) for prefix in _OPENAI_REASONING_PREFIXES)


class LLMClient:
    """Async LLM client that dispatches to the correct provider SDK."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.provider = detect_provider(model)
        self._client: Any = None
        self._init_client()

    def _init_client(self) -> None:
        if self.provider == "openai":
            from openai import AsyncOpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            # max_retries enables the SDK's exponential backoff with jitter
            # (honors Retry-After) on 429s / transient errors.
            self._client = AsyncOpenAI(api_key=api_key, max_retries=6)
        elif self.provider == "google":
            from google import genai

            # Use the Vertex AI client, which reads GOOGLE_API_KEY from the
            # environment.
            if not os.getenv("GOOGLE_API_KEY"):
                raise ValueError("GOOGLE_API_KEY environment variable is not set")
            self._client = genai.Client(vertexai=True)
        else:
            import anthropic
            from anthropic import AsyncAnthropic

            # Allow longer backoff when concurrent requests hit rate limits.
            # The SDK still honors the server's Retry-After header first.
            anthropic._base_client.MAX_RETRY_DELAY = 60.0

            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
            # max_retries enables the SDK's exponential backoff with jitter
            # (honors Retry-After) on 429s / transient errors.
            self._client = AsyncAnthropic(api_key=api_key, max_retries=6)

    async def create_message(
        self,
        *,
        system: str = "",
        messages: list[dict[str, str]],
        max_tokens: int = 8192,
        temperature: float = 1.0,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> LLMResponse:
        if self.provider == "openai":
            return await self._call_openai(
                system, messages, max_tokens, temperature, reasoning_effort,
            )
        if self.provider == "google":
            return await self._call_google(system, messages, max_tokens, temperature)
        return await self._call_anthropic(system, messages, max_tokens, temperature)

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------

    # Models where the Anthropic API rejects the temperature parameter.
    _NO_TEMPERATURE_MODELS: frozenset[str] = frozenset({"claude-opus-4-7"})

    @staticmethod
    def _is_retriable_llm_error(exc: BaseException) -> bool:
        """Return True for transient provider errors worth retrying."""
        try:
            from anthropic import (
                APIConnectionError as AnthropicAPIConnectionError,
                APITimeoutError as AnthropicAPITimeoutError,
                InternalServerError as AnthropicInternalServerError,
                RateLimitError as AnthropicRateLimitError,
            )
        except ImportError:
            anthropic_errors: tuple[type[BaseException], ...] = ()
        else:
            anthropic_errors = (
                AnthropicAPIConnectionError,
                AnthropicAPITimeoutError,
                AnthropicInternalServerError,
                AnthropicRateLimitError,
            )

        try:
            from openai import (
                APIConnectionError as OpenAIAPIConnectionError,
                APITimeoutError as OpenAIAPITimeoutError,
                InternalServerError as OpenAIInternalServerError,
                RateLimitError as OpenAIRateLimitError,
            )
        except ImportError:
            openai_errors: tuple[type[BaseException], ...] = ()
        else:
            openai_errors = (
                OpenAIAPIConnectionError,
                OpenAIAPITimeoutError,
                OpenAIInternalServerError,
                OpenAIRateLimitError,
            )

        return isinstance(exc, anthropic_errors + openai_errors)

    async def _anthropic_stream_final_message(self, kwargs: dict[str, Any]) -> Any:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception(self._is_retriable_llm_error),
            stop=stop_after_attempt(4),
            wait=wait_exponential_jitter(initial=2, max=60),
            reraise=True,
        ):
            with attempt:
                if attempt.retry_state.attempt_number > 1:
                    logger.warning(
                        "Retrying Anthropic stream for model=%s (attempt %d): %s",
                        self.model,
                        attempt.retry_state.attempt_number,
                        attempt.retry_state.outcome.exception(),  # type: ignore[union-attr]
                    )
                async with self._client.messages.stream(**kwargs) as stream:
                    return await stream.get_final_message()

    async def _call_anthropic(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens,
            messages=messages,
        )
        if self.model not in self._NO_TEMPERATURE_MODELS:
            kwargs["temperature"] = temperature
        if system:
            kwargs["system"] = system

        response = await self._anthropic_stream_final_message(kwargs)

        usage = response.usage
        return LLMResponse(
            text=response.content[0].text,
            input_tokens=getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
            cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0),
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0),
            model=self.model,
            provider="anthropic",
            stop_reason=response.stop_reason or "",
        )

    async def _call_google(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        from google.genai import types

        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg["content"])],
                )
            )

        config_kwargs: dict[str, Any] = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            config_kwargs["system_instruction"] = system
        config = types.GenerateContentConfig(**config_kwargs)

        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        text = ""
        finish_reason = ""
        if response.candidates:
            candidate = response.candidates[0]
            for part in (candidate.content.parts or []):
                if hasattr(part, "text") and part.text:
                    text += part.text
            fr = candidate.finish_reason
            if fr is not None:
                fr_name = fr.name if hasattr(fr, "name") else str(fr)
                finish_reason = "max_tokens" if fr_name == "MAX_TOKENS" else fr_name.lower()

        usage = response.usage_metadata
        input_tokens = getattr(usage, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0

        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            provider="google",
            stop_reason=finish_reason,
        )

    async def _call_openai(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        reasoning_effort: ReasoningEffort | None,
    ) -> LLMResponse:
        oai_messages: list[dict[str, str]] = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        oai_messages.extend(messages)

        request_kwargs: dict[str, Any] = dict(
            model=self.model,
            max_completion_tokens=max_tokens,
            messages=oai_messages,
        )
        if model_supports_custom_temperature(self.model):
            request_kwargs["temperature"] = temperature
        elif temperature != 1.0 and not getattr(self, "_warned_fixed_temperature", False):
            logger.warning(
                "Model %s only accepts the default temperature (1.0); "
                "ignoring requested %.2f for this and future calls.",
                self.model, temperature,
            )
            self._warned_fixed_temperature = True

        if reasoning_effort is not None:
            if model_uses_reasoning(self.model):
                request_kwargs["reasoning_effort"] = reasoning_effort
            elif not getattr(self, "_warned_unused_reasoning_effort", False):
                logger.warning(
                    "Model %s is not a reasoning model; ignoring requested "
                    "reasoning_effort=%r for this and future calls.",
                    self.model, reasoning_effort,
                )
                self._warned_unused_reasoning_effort = True

        async for attempt in AsyncRetrying(
            retry=retry_if_exception(self._is_retriable_llm_error),
            stop=stop_after_attempt(4),
            wait=wait_exponential_jitter(initial=2, max=60),
            reraise=True,
        ):
            with attempt:
                if attempt.retry_state.attempt_number > 1:
                    logger.warning(
                        "Retrying OpenAI chat completion for model=%s (attempt %d): %s",
                        self.model,
                        attempt.retry_state.attempt_number,
                        attempt.retry_state.outcome.exception(),  # type: ignore[union-attr]
                    )
                response = await self._client.chat.completions.create(**request_kwargs)

        usage = response.usage
        prompt_tokens = 0
        completion_tokens = 0
        cached_tokens = 0
        reasoning_tokens = 0

        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            details = getattr(usage, "prompt_tokens_details", None)
            if details:
                cached_tokens = getattr(details, "cached_tokens", 0) or 0
            completion_details = getattr(usage, "completion_tokens_details", None)
            if completion_details:
                reasoning_tokens = (
                    getattr(completion_details, "reasoning_tokens", 0) or 0
                )

        choice = response.choices[0]
        finish_reason = choice.finish_reason or ""
        content = choice.message.content or ""

        if finish_reason == "length" and not content:
            logger.warning(
                "OpenAI call returned empty content (model=%s, finish_reason=length, "
                "max_completion_tokens=%d, output_tokens=%d, reasoning_tokens=%d). "
                "The response budget was exhausted before any visible output was "
                "produced; raise max_tokens or lower reasoning_effort for this call.",
                self.model, max_tokens, completion_tokens, reasoning_tokens,
            )

        return LLMResponse(
            text=content,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            cache_creation_tokens=0,
            cache_read_tokens=cached_tokens,
            model=self.model,
            provider="openai",
            stop_reason=finish_reason,
        )
