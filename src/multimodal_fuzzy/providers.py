"""Provider configuration for OpenAI-compatible multimodal chat APIs."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse


PROVIDER_CHOICES = ("openai", "gemini", "openrouter", "ollama", "ollama-local", "ollama-cloud")

_DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama-local": "http://localhost:11434/v1",
    "ollama-cloud": "https://ollama.com/v1",
}

_PROVIDER_KEY_ENVS = {
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"),
    "openrouter": ("OPENROUTER_API_KEY",),
    "ollama-cloud": ("OLLAMA_API_KEY",),
}


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model: str
    base_url: str
    api_key: str = field(repr=False)


def _legacy_provider() -> str:
    """Infer Gemini for the old OPENAI_BASE_URL Gemini configuration."""
    old_base_url = os.getenv("OPENAI_BASE_URL", "").strip().lower()
    if "generativelanguage.googleapis.com" in old_base_url:
        return "gemini"
    return "openai"


def _normalize_provider(provider: str, mode: str | None) -> str:
    normalized = provider.strip().lower().replace("_", "-")
    if normalized == "ollama":
        selected_mode = (mode or os.getenv("LLM_MODE", "local")).strip().lower()
        if selected_mode not in {"local", "cloud"}:
            raise ValueError("LLM_MODE must be 'local' or 'cloud' when LLM_PROVIDER=ollama.")
        return f"ollama-{selected_mode}"
    if normalized in {"ollama-local", "ollama-cloud"}:
        return normalized
    if normalized not in {"openai", "gemini", "openrouter"}:
        raise ValueError(
            f"Unsupported provider {provider!r}. Choose one of: "
            "openai, gemini, openrouter, ollama (local/cloud)."
        )
    return normalized


def resolve_provider_config(
    *,
    provider: str | None = None,
    mode: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> ProviderConfig:
    """Resolve CLI options and environment variables into a backend config.

    ``LLM_PROVIDER=ollama`` plus ``LLM_MODE=local|cloud`` selects Ollama. The
    older ``OPENAI_MODEL`` and ``OPENAI_BASE_URL`` settings remain supported.
    """
    raw_provider = (provider or os.getenv("LLM_PROVIDER", "").strip() or _legacy_provider())
    raw_normalized = raw_provider.strip().lower().replace("_", "-")
    if mode is not None and raw_normalized != "ollama":
        expected_mode = {"ollama-local": "local", "ollama-cloud": "cloud"}.get(raw_normalized)
        if expected_mode is None or mode != expected_mode:
            raise ValueError("--mode is only valid with --provider ollama (or a matching Ollama provider alias).")
    selected_provider = _normalize_provider(raw_provider, mode)

    model_envs = {
        "openai": ("OPENAI_MODEL",),
        "gemini": ("GEMINI_MODEL", "OPENAI_MODEL"),
        "openrouter": ("OPENROUTER_MODEL",),
        "ollama-local": ("OLLAMA_LOCAL_MODEL", "OLLAMA_MODEL"),
        "ollama-cloud": ("OLLAMA_CLOUD_MODEL", "OLLAMA_MODEL"),
    }
    selected_model = (model or os.getenv("LLM_MODEL", "").strip()).strip()
    if not selected_model:
        for env_name in model_envs[selected_provider]:
            selected_model = os.getenv(env_name, "").strip()
            if selected_model:
                break
    if not selected_model and selected_provider == "ollama-local":
        selected_model = "qwen3-vl:4b-instruct"
    if not selected_model:
        raise ValueError(
            f"No model selected for {selected_provider}. Set LLM_MODEL or pass --model."
        )

    selected_base_url = (base_url or os.getenv("LLM_BASE_URL", "").strip()).strip()
    if not selected_base_url and selected_provider in {"openai", "gemini"}:
        selected_base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if not selected_base_url:
        selected_base_url = _DEFAULT_BASE_URLS[selected_provider]
    parsed = urlparse(selected_base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("LLM_BASE_URL must be a complete http:// or https:// URL.")

    api_key = ""
    for env_name in _PROVIDER_KEY_ENVS.get(selected_provider, ()):
        api_key = os.getenv(env_name, "").strip()
        if api_key:
            break

    # Ollama's local OpenAI-compatible endpoint requires a non-empty key value,
    # but ignores it. A signed-in local server can also proxy :cloud models.
    if selected_provider == "ollama-local":
        api_key = "ollama"
    elif selected_provider == "ollama-cloud" and not api_key:
        host = (parsed.hostname or "").lower()
        if host in {"localhost", "127.0.0.1", "::1"}:
            api_key = "ollama"

    if not api_key:
        key_envs = " or ".join(_PROVIDER_KEY_ENVS[selected_provider])
        raise ValueError(
            f"{selected_provider} requires an API key in {key_envs}. "
            "For Ollama Cloud through a local signed-in server, set "
            "LLM_BASE_URL=http://localhost:11434/v1 and run 'ollama signin' yourself."
        )

    return ProviderConfig(
        provider=selected_provider,
        model=selected_model,
        base_url=selected_base_url.rstrip("/"),
        api_key=api_key,
    )


def create_chat_client(config: ProviderConfig) -> Any:
    """Create a client for the configured OpenAI-compatible endpoint."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is not installed. Run: python -m pip install -r requirements.txt"
        ) from exc
    return OpenAI(api_key=config.api_key, base_url=config.base_url)


def image_content_part(provider: str, data_url: str) -> dict[str, Any]:
    """Return an image part in the shape accepted by the selected endpoint."""
    if provider.startswith("ollama-"):
        # Ollama's OpenAI-compatible vision endpoint documents a base64 data URL
        # as the image_url value itself (rather than an object with a `url`).
        return {"type": "image_url", "image_url": data_url}
    return {
        "type": "image_url",
        "image_url": {"url": data_url, "detail": "high"},
    }
