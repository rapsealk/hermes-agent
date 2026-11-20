"""Model metadata disk caches (sibling of ``model_metadata``).

One-shot runs (``hermes -q``, cron) start cold and Nous bypasses the persistent
context cache, so every launch would pay a live probe; these helpers memoize
probed model metadata, remote ``/models`` results and local-endpoint verdicts
under ``get_hermes_home()/cache/`` with TTL pruning.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from utils import atomic_json_write

logger = logging.getLogger(__name__)

_ENDPOINT_MODEL_CACHE_TTL = 300

# Only SUCCESSFUL probes persist (a down server must not pin a negative verdict).
_LOCAL_PROBE_DISK_TTL_SECONDS = 300.0


def _cache_file(name: str) -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "cache" / name


def _load_json_dict(path: Path) -> dict[str, Any]:
    """JSON object at ``path``, or {} when missing/invalid."""
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _ttl_memo_get(path: Path, key: str, ttl: float, *, ts_key: str, value_key: str) -> Optional[Any]:
    """Fresh (< ``ttl`` seconds) ``value_key`` of the ``key`` entry in a TTL memo file, else None."""
    entry = _load_json_dict(path).get(key)
    if not isinstance(entry, dict):
        return None
    try:
        if (time.time() - float(entry[ts_key])) >= ttl:
            return None
        return entry[value_key]
    except Exception:
        return None


def _ttl_memo_put(path: Path, key: str, value: Any, ttl: float, *, ts_key: str, value_key: str, what: str, ts_first: bool = False) -> None:
    """Write ``key`` into a TTL memo file, dropping expired siblings. Best-effort."""
    try:
        now = time.time()
        data = {k: v for k, v in _load_json_dict(path).items() if isinstance(v, dict) and (now - float(v.get(ts_key, 0))) < ttl}
        data[key] = {ts_key: now, value_key: value} if ts_first else {value_key: value, ts_key: now}
        atomic_json_write(path, data, indent=0, separators=(",", ":"))
    except Exception as e:
        logger.debug("Failed to save %s: %s", what, e)


def _local_probe_disk_cache_path() -> Path:
    return _cache_file("local_endpoint_probes.json")


def _local_probe_disk_get(kind: str, key: str) -> Optional[Any]:
    """Return a fresh cached value for ``kind:key``, else None."""
    return _ttl_memo_get(_local_probe_disk_cache_path(), f"{kind}:{key}", _LOCAL_PROBE_DISK_TTL_SECONDS, ts_key="ts", value_key="value")


def _local_probe_disk_put(kind: str, key: str, value: Any) -> None:
    """Persist a successful probe result. Best-effort; prunes stale entries."""
    _ttl_memo_put(_local_probe_disk_cache_path(), f"{kind}:{key}", value, _LOCAL_PROBE_DISK_TTL_SECONDS, ts_key="ts", value_key="value", what="local probe disk cache")


def _get_model_metadata_cache_path() -> Path:
    """Path to the OpenRouter model metadata disk cache."""
    return _cache_file("openrouter_model_metadata.json")


def _model_metadata_disk_cache_age_seconds() -> Optional[float]:
    """Disk-cache age in seconds, or None if freshness is unknown."""
    try:
        age = time.time() - _get_model_metadata_cache_path().stat().st_mtime
        return age if age >= 0 else None
    except Exception:
        return None


def _load_model_metadata_disk_cache() -> dict[str, dict[str, Any]]:
    """Processed OpenRouter metadata cache from disk ({} on any failure)."""
    try:
        with _get_model_metadata_cache_path().open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        return {str(key): value for key, value in data.items() if isinstance(value, dict)} if isinstance(data, dict) else {}
    except Exception as e:
        logger.debug("Failed to load OpenRouter model metadata disk cache: %s", e)
        return {}


def _save_model_metadata_disk_cache(data: dict[str, dict[str, Any]]) -> None:
    try:
        atomic_json_write(_get_model_metadata_cache_path(), data, indent=0, separators=(",", ":"))
    except Exception as e:
        logger.debug("Failed to save OpenRouter model metadata disk cache: %s", e)


def _get_endpoint_metadata_cache_path() -> Path:
    """On-disk memo of remote ``/models`` probes (see ``_endpoint_disk_cache_get``)."""
    return _cache_file("endpoint_model_metadata.json")


def _endpoint_disk_cache_get(normalized: str) -> Optional[dict[str, dict[str, Any]]]:
    """Fresh cross-process memo of a remote ``/models`` probe (same TTL as in-memory): one-shot
    runs (``hermes -q``, cron) start cold and Nous bypasses the persistent context cache, so
    without this every launch paid the live probe. Local endpoints are never memoized."""
    models = _ttl_memo_get(_get_endpoint_metadata_cache_path(), normalized, _ENDPOINT_MODEL_CACHE_TTL, ts_key="at", value_key="models")
    return models if isinstance(models, dict) else None


def _endpoint_disk_cache_put(normalized: str, cache: dict[str, dict[str, Any]]) -> None:
    """Memoize a successful remote ``/models`` probe; expired siblings are dropped."""
    _ttl_memo_put(
        _get_endpoint_metadata_cache_path(), normalized, cache, _ENDPOINT_MODEL_CACHE_TTL,
        ts_key="at", value_key="models", what="endpoint model metadata disk cache", ts_first=True)
