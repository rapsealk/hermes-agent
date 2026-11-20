"""DeepInfra provider catalogs (sibling of ``hermes_cli.models``).

DeepInfra's generic ``/models`` mixes chat, image, video, speech and embedding
models; the tagged catalog helper here is the only safe source for the chat
picker. The catalog is user-scoped (private fine-tunes), so the cache key
carries the api-key fingerprint and the negative cache keeps an unreachable
catalog from eating the full timeout on every surface helper.
"""

from __future__ import annotations

import re
import time
from typing import Optional

from hermes_cli.version_info import hermes_user_agent

# Legacy id-regex filter for items with no surface tag; unreachable (deletable) once every catalog
# entry carries an explicit ``chat``/``embed``/``image-gen``/``tts``/``stt`` tag.
_DEEPINFRA_EXCLUDE_RE = re.compile(
    r"(?i)(embed|rerank|whisper|stable-diffusion|flux|sdxl|"
    r"tts|bark|speech|image-gen|clip|vit-|dpt-)")

# Surface tags say *what kind of model* this is. Absent all of them, the tags array only carries
# capability tags (``reasoning``, ``vision``, …) and the chat surface falls back to id-regex inference.
_DEEPINFRA_SURFACE_TAGS: frozenset[str] = frozenset({
    "chat", "embed", "image-gen", "tts", "stt", "video-gen"})

_DEEPINFRA_DEFAULT_BASE_URL = "https://api.deepinfra.com/v1/openai"
_DEEPINFRA_MODELS_QUERY = "filter=true&sort_by=hermes"

# Full tagged catalog keyed by base URL; every surface filter reads it so one round-trip serves all.
_deepinfra_catalog_cache: dict[str, list[dict]] = {}

# Negative cache (monotonic time of the last failed fetch per base URL) so an unreachable catalog
# doesn't make every surface helper eat the full timeout in turn. Short TTL so connectivity recovers.
_deepinfra_catalog_neg_cache: dict[str, float] = {}
_DEEPINFRA_CATALOG_NEG_TTL = 60.0  # seconds


def _deepinfra_env(key: str) -> str:
    """Profile-scoped ``.env``/environ read: under a multiplexed turn the launch env is not this profile's."""
    from hermes_cli.config import get_env_value_prefer_dotenv
    return (get_env_value_prefer_dotenv(key) or "").strip()


def _deepinfra_catalog_url() -> tuple[str, str]:
    """Return ``(cache_key, full_url)`` for the DeepInfra catalog endpoint. The key carries the
    api-key fingerprint: the catalog is user-scoped (private fine-tunes), so two profiles with
    different keys must not share an entry."""
    base = (_deepinfra_env("DEEPINFRA_BASE_URL") or _DEEPINFRA_DEFAULT_BASE_URL).rstrip("/")
    from agent.credential_persistence import fingerprint_secret_value
    fp = fingerprint_secret_value(_deepinfra_env("DEEPINFRA_API_KEY")) or "anon"
    return f"{base}#{fp}", f"{base}/models?{_DEEPINFRA_MODELS_QUERY}"


def _fetch_deepinfra_catalog(
    *, timeout: float = 5.0, force_refresh: bool = False) -> Optional[list[dict]]:
    """Raw DeepInfra catalog list (chat, embed, image-gen, TTS, STT in one response), cached per base
    URL. A Bearer token is attached when available so user-scoped catalogs (private fine-tunes) show."""
    from hermes_cli.models import _get_json

    cache_key, url = _deepinfra_catalog_url()
    if not force_refresh:
        if cache_key in _deepinfra_catalog_cache:
            return _deepinfra_catalog_cache[cache_key]
        last_fail = _deepinfra_catalog_neg_cache.get(cache_key)
        if last_fail is not None and (time.monotonic() - last_fail) < _DEEPINFRA_CATALOG_NEG_TTL:
            return None

    headers: dict[str, str] = {"User-Agent": hermes_user_agent()}
    api_key = _deepinfra_env("DEEPINFRA_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        payload = _get_json(url, timeout=timeout, headers=headers)
    except Exception:
        _deepinfra_catalog_neg_cache[cache_key] = time.monotonic()
        return None
    data = payload.get("data")
    if not isinstance(data, list):
        _deepinfra_catalog_neg_cache[cache_key] = time.monotonic()
        return None
    _deepinfra_catalog_cache[cache_key] = data
    _deepinfra_catalog_neg_cache.pop(cache_key, None)
    return data


def _fetch_deepinfra_models_by_tag(
    tag: str, *, timeout: float = 5.0, force_refresh: bool = False) -> Optional[list[dict]]:
    """DeepInfra ``{"id", "metadata"}`` items whose ``metadata.tags`` includes *tag*. Items with no
    surface tag fall through to the legacy id-regex exclusion (chat surface only — embed/image-gen/
    tts/stt cannot be inferred from an id). ``None`` on network failure."""
    data = _fetch_deepinfra_catalog(timeout=timeout, force_refresh=force_refresh)
    if data is None:
        return None
    matched: list[dict] = []
    for item in data:
        mid = item.get("id")
        raw_metadata = item.get("metadata")
        if not mid or raw_metadata is None:  # metadata None = listed-but-not-served stub
            continue
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        raw_tags = metadata.get("tags")
        tags = raw_tags if isinstance(raw_tags, list) else []
        if any(t in _DEEPINFRA_SURFACE_TAGS for t in tags):
            hit = tag in tags
        else:
            hit = tag == "chat" and not _DEEPINFRA_EXCLUDE_RE.search(mid)
        if hit:
            matched.append({"id": mid, "metadata": metadata})
    return matched


def _fetch_deepinfra_models(
    timeout: float = 5.0, *, force_refresh: bool = False) -> Optional[list[str]]:
    """DeepInfra chat-model ids (string-list contract for :func:`provider_model_ids`); ``None`` on
    network failure or when no chat-tagged id exists."""
    items = _fetch_deepinfra_models_by_tag("chat", timeout=timeout, force_refresh=force_refresh)
    return ([item["id"] for item in items] or None) if items is not None else None


def deepinfra_model_ids(tag: str, *, force_refresh: bool = False) -> list[str]:
    """Return DeepInfra model ids carrying surface *tag* (``[]`` on failure)."""
    items = _fetch_deepinfra_models_by_tag(tag, force_refresh=force_refresh)
    return [item["id"] for item in items] if items else []


def deepinfra_base_url(section: Optional[dict] = None) -> str:
    """DeepInfra base URL: config-section ``base_url`` → ``DEEPINFRA_BASE_URL`` env → default; stripped."""
    candidate = section.get("base_url") if isinstance(section, dict) else None
    value = candidate or _deepinfra_env("DEEPINFRA_BASE_URL") or _DEEPINFRA_DEFAULT_BASE_URL
    return str(value).strip().rstrip("/")
