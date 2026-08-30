"""LiteLLM egress layer — the single LLM call path for the whole gateway.

Replaces the previous Portkey coupling.  LiteLLM (BSD-3) is a *library*, not
a hosted service: it needs no account, no API gateway and no running proxy,
and it speaks 100+ providers through one call signature.  It supplies the
retries, timeouts and provider abstraction that Portkey used to, without the
commercial dependency.

Nothing else in the application may call an LLM directly.  Both
``app/router/model_router.py`` and ``app/ingress/streaming_router.py``
route through here.

Degradation ladder, in order:
  1. Requested tier via LiteLLM.
  2. The other tier (SLM <-> FRONTIER) on any dispatch failure.
  3. Contextual mock text — always available, never raises.

Streaming is the one exception to step 3: a failure mid-stream must
propagate so the SSE layer can emit ``[STREAM_ERROR]`` rather than splice
fake tokens into a partially delivered response.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator, Literal

from app.config import settings, _is_real_key

logger = logging.getLogger(__name__)

Tier = Literal["SLM", "FRONTIER"]

# Optional litellm import — the gateway runs on mock responses without it.
try:
    import litellm  # type: ignore[import-untyped]

    # Quieten the library: it otherwise prints provider banners and
    # cost tables to stdout, which corrupts our structured logs.
    litellm.suppress_debug_info = True
    litellm.drop_params = True  # ignore kwargs a given provider rejects
    _acompletion = litellm.acompletion
    _HAS_LITELLM = True
except ImportError:  # pragma: no cover - exercised via monkeypatch
    litellm = None  # type: ignore[assignment]
    _acompletion = None  # type: ignore[assignment]
    _HAS_LITELLM = False
    logger.info(
        "LITELLM_ABSENT — install with `pip install '.[llm]'` for real LLM "
        "dispatch; contextual mock responses active"
    )


# LLM_PROVIDER values -> LiteLLM provider prefixes.  These differ: LiteLLM
# calls Google "gemini" and xAI "xai".
_PROVIDER_PREFIX: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "gemini",
    "grok": "xai",
    "generic": "openai",  # any OpenAI-compatible endpoint via LLM_API_BASE
}

_DEFAULT_SLM = "gpt-4o-mini"


def _model_for_tier(tier: Tier) -> str:
    """Return the fully qualified LiteLLM model string for *tier*.

    Bare names gain a provider prefix; already-qualified names
    (containing "/") are passed through untouched so an operator can mix
    providers across tiers.
    """
    if tier == "FRONTIER":
        model = settings.FRONTIER_MODEL
    else:
        model = settings.SLM_MODEL
        # Back-compat: an existing .env may set only LLM_FALLBACK_MODEL.
        if model == _DEFAULT_SLM and settings.LLM_FALLBACK_MODEL != _DEFAULT_SLM:
            model = settings.LLM_FALLBACK_MODEL

    if "/" in model:
        return model
    prefix = _PROVIDER_PREFIX.get(settings.LLM_PROVIDER, "openai")
    return f"{prefix}/{model}"


def _other(tier: Tier) -> Tier:
    return "FRONTIER" if tier == "SLM" else "SLM"


def is_live() -> bool:
    """True when a real LLM call is possible.

    A real API key is the common case, but keyless local backends (Ollama,
    vLLM, LM Studio) are equally "live" as long as LLM_API_BASE points at
    them — they have no API key concept at all, so gating on the key alone
    would leave them stuck on mock responses forever.
    """
    if not _HAS_LITELLM:
        return False
    return _is_real_key(settings.LLM_API_KEY) or bool(settings.LLM_API_BASE)


def generate_contextual_response(prompt: str) -> str:
    """Deterministic canned answers for local dev, CI and total-failure paths.

    Duplicated here from model_router.py's ``_generate_contextual_response``
    (transitional — see module docstring) so that every mock path —
    buffered, streaming, and post-failure — produces identical text.
    """
    p = prompt.lower()
    if "return" in p or "refund" in p:
        return (
            "Our standard enterprise policy permits hardware returns within 30 "
            "calendar days of delivery, provided items are returned in original "
            "packaging with valid RMA authorization."
        )
    if "balance" in p or "account" in p or "ssn" in p or "email" in p:
        return (
            "Your corporate account identity has been verified. Current balance: "
            "$2,450.00 with active enterprise tier access."
        )
    if "phoenix" in p or "apollo" in p or "checklist" in p:
        return (
            "The deployment verification checklist requires passing all integration "
            "regression suites, validating IAM least-privilege policies, and "
            "ensuring multi-region redundancy."
        )
    if "dan" in p or "bypass" in p:
        return "I am unable to fulfill instructions that request disabling enterprise security guardrails."
    return (
        f"Request processed: {prompt}. Completed safely in accordance with "
        "enterprise safety standards."
    )


def _build_messages(prompt: str, system: str | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def _call_kwargs(model: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "timeout": settings.LLM_TIMEOUT_S,
        "num_retries": settings.LLM_MAX_RETRIES,
        # Same budget concern as model_router: reasoning models (Gemini 3.x,
        # o-series) spend this on internal thinking before emitting visible
        # text, so 512 truncates answers mid-sentence.
        "max_tokens": 2048,
    }
    # Omit api_key entirely when blank rather than passing api_key="" — an
    # empty string is a value litellm would treat as a (bad) real credential
    # instead of "no key configured", which breaks keyless local backends.
    if settings.LLM_API_KEY:
        kwargs["api_key"] = settings.LLM_API_KEY
    if settings.LLM_API_BASE:
        kwargs["api_base"] = settings.LLM_API_BASE
    return kwargs


async def acomplete(
    prompt: str,
    tier: Tier,
    system: str | None = None,
) -> tuple[str, str]:
    """Dispatch *prompt* at *tier* and return ``(text, model_used)``.

    Never raises.  On dispatch failure it retries the opposite tier, then
    falls back to contextual mock text with ``model_used == "mock"``.
    """
    if not is_live():
        return generate_contextual_response(prompt), "mock"

    messages = _build_messages(prompt, system)

    for attempt_tier in (tier, _other(tier)):
        model = _model_for_tier(attempt_tier)
        try:
            response = await _acompletion(**_call_kwargs(model, messages))
            text = response.choices[0].message.content or ""
            logger.info("LLM_DISPATCH_OK model=%s tier=%s", model, attempt_tier)
            return text, model
        except Exception as exc:
            logger.warning(
                "LLM_DISPATCH_FAILED model=%s tier=%s error=%s",
                model,
                attempt_tier,
                exc,
            )

    logger.error("LLM_DISPATCH_EXHAUSTED — falling back to mock response")
    return generate_contextual_response(prompt), "mock"


async def _acompletion_stream(**kwargs: Any) -> AsyncGenerator[Any, None]:
    """Thin seam over litellm streaming so tests can substitute it."""
    stream = await _acompletion(stream=True, **kwargs)
    async for chunk in stream:
        yield chunk


async def astream(
    prompt: str,
    tier: Tier,
    system: str | None = None,
) -> AsyncGenerator[str, None]:
    """Yield content deltas for *prompt*.

    Unlike :func:`acomplete` this **propagates** dispatch errors: the SSE
    endpoint must be able to emit ``[STREAM_ERROR]`` rather than splice mock
    tokens into a half-delivered response.  When no credentials are
    configured it yields a chunked mock stream instead.
    """
    if not is_live():
        for word in generate_contextual_response(prompt).split():
            yield word + " "
            await asyncio.sleep(0)
        return

    model = _model_for_tier(tier)
    messages = _build_messages(prompt, system)

    async for chunk in _acompletion_stream(**_call_kwargs(model, messages)):
        try:
            delta = chunk.choices[0].delta
            token = getattr(delta, "content", None)
        except (AttributeError, IndexError):
            continue
        if token:
            yield token
