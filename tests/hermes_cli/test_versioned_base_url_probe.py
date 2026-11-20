"""Behaviour-contract tests for the versioned-URL guard in ``probe_api_models``.

A base URL that already carries an API version segment (``/api/v2``, ``/v1beta``,
``/v2023-01-01``) must not get a ``/v1`` appended as a fallback candidate — that
alternate is a nonsense URL, and when the original answers Hermes would report a
``{versioned}/v1`` shape as a "working base". A plain host root still probes
``base`` then ``base/v1``; the salvage builds on the original PR #2 fix
(cherry-picked from ``fix/custom-endpoint-v1-suffix``).
"""

from __future__ import annotations

from unittest.mock import patch

from hermes_cli import models


def _probe_urls_of(base_url: str) -> list[str]:
    """Run ``probe_api_models`` with the outbound request recording and failing."""
    fetched: list[str] = []

    def _fail(url, *, timeout, headers=None, **kw):
        fetched.append(url)
        raise ConnectionError("unreachable in test")

    with patch.object(models, "_get_json", side_effect=_fail):
        models.probe_api_models(None, base_url, timeout=1)
    return fetched


def test_versioned_base_url_skips_v1_alternate():
    # Already versioned: only the exact URL is probed, never {base}/v1.
    assert _probe_urls_of("https://gw.example.com/api/v2") == [
        "https://gw.example.com/api/v2/models",
    ]


def test_plain_base_url_still_probes_v1_alternate():
    # Plumbing still works: an unversioned root gets the /v1 fallback candidate.
    assert _probe_urls_of("https://gw.example.com") == [
        "https://gw.example.com/models",
        "https://gw.example.com/v1/models",
    ]
