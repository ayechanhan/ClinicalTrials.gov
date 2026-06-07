"""Thin async ``httpx`` wrapper around the ClinicalTrials.gov Data API (v2).

Endpoints wrapped
-----------------
* ``GET /studies``             -> :meth:`ClinicalTrialsClient.search_studies`
* ``GET /studies/{nctId}``     -> :meth:`ClinicalTrialsClient.get_study`
* ``GET /stats/field/values``  -> :meth:`ClinicalTrialsClient.get_field_values`

Verified API behaviour (probed against the live API on build)
-------------------------------------------------------------
* ``fields`` accepts short names — ``NCTId``, ``BriefTitle``, ``Phase``,
  ``OverallStatus``, ``StartDate``, ``LeadSponsorName``, ``Condition``,
  ``InterventionName``, ``LocationCountry`` — and the response only contains the
  modules holding those fields.
* Each study is returned as ``{"protocolSection": {<module>: {...}}}``. Useful
  paths: ``identificationModule.nctId`` / ``.briefTitle``,
  ``statusModule.overallStatus`` / ``.startDateStruct.date``,
  ``sponsorCollaboratorsModule.leadSponsor.name``, ``designModule.phases`` (list),
  ``conditionsModule.conditions`` (list),
  ``armsInterventionsModule.interventions[].name``,
  ``contactsLocationsModule.locations[].country``.
* Pagination: pass the previous response's ``nextPageToken`` as ``pageToken``.
* ``countTotal=true`` adds ``totalCount`` to the response.
* Field filters use Essie expressions via ``filter.advanced`` — e.g.
  ``AREA[Phase]PHASE3``. Status filtering uses ``filter.overallStatus``.
* ``/stats/field/values`` is **global only**: it rejects ``query.*`` and
  ``filter.*`` params (HTTP 400) and returns counts but *no* nct_ids. It cannot
  back scoped distributions or per-data-point citations, so the main pipeline
  aggregates fetched studies instead.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Iterable

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://clinicaltrials.gov/api/v2"
DEFAULT_PAGE_SIZE = 100
DEFAULT_MAX_PAGES = 10
MAX_PAGES_CAP = 50  # safety ceiling on the configured page count
API_MAX_PAGE_SIZE = 1000  # hard cap enforced by the API
DEFAULT_TIMEOUT = 30.0


class CTClientError(RuntimeError):
    """Raised when the ClinicalTrials.gov API errors or is unreachable.

    ``status_code`` is the upstream HTTP status when available (``None`` for
    transport-level failures such as timeouts). The pipeline maps this to a 502.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class StudySearchResult:
    """Outcome of a (possibly paginated, capped) ``/studies`` search.

    ``studies`` are raw API study dicts (each a ``{"protocolSection": ...}``).
    ``total_count`` is the API's match count (``None`` if not requested).
    ``truncated`` is True when more studies matched than were fetched (the cap
    was hit) — surfaced in the response ``meta.notes``.
    """

    studies: list[dict[str, Any]]
    total_count: int | None
    pages_fetched: int
    truncated: bool


def _as_csv(value: str | Iterable[str]) -> str:
    """Normalize a string or iterable of strings into a comma-separated string."""
    if isinstance(value, str):
        return value
    return ",".join(str(v) for v in value)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "Env var %s=%r is not an int; using default %d", name, raw, default
        )
        return default


class ClinicalTrialsClient:
    """Async client for the ClinicalTrials.gov v2 Data API.

    Use as an async context manager so the underlying connection pool is closed::

        async with ClinicalTrialsClient() as ct:
            result = await ct.search_studies(query_term="Pembrolizumab")

    Configuration falls back to env vars (``CT_API_BASE``, ``CT_PAGE_SIZE``,
    ``CT_MAX_PAGES``) and then to module defaults. No API key is required — the
    ClinicalTrials.gov API is public.
    """

    def __init__(
        self,
        base_url: str | None = None,
        page_size: int | None = None,
        max_pages: int | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = (
            base_url or os.getenv("CT_API_BASE") or DEFAULT_BASE_URL
        ).rstrip("/")
        size = (
            page_size
            if page_size is not None
            else _env_int("CT_PAGE_SIZE", DEFAULT_PAGE_SIZE)
        )
        self.page_size = max(1, min(size, API_MAX_PAGE_SIZE))
        pages = (
            max_pages
            if max_pages is not None
            else _env_int("CT_MAX_PAGES", DEFAULT_MAX_PAGES)
        )
        self.max_pages = max(1, min(pages, MAX_PAGES_CAP))
        self.timeout = timeout
        self._client = client
        self._owns_client = client is None

    async def __aenter__(self) -> "ClinicalTrialsClient":
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
            self._owns_client = True
        return self

    async def __aexit__(self, *_exc: object) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(
                "ClinicalTrialsClient must be used inside `async with ClinicalTrialsClient() as ct:`"
            )
        return self._client

    async def _get(self, path: str, params: dict[str, Any]) -> Any:
        """GET an absolute URL and return parsed JSON, or raise CTClientError.

        Builds the full URL explicitly (rather than via httpx ``base_url``) to
        avoid RFC-3986 path-join surprises where a leading-slash path would drop
        the ``/api/v2`` prefix.
        """
        url = f"{self.base_url}{path}"
        logger.debug("GET %s params=%s", url, params)
        try:
            resp = await self._http.get(url, params=params)
        except httpx.HTTPError as exc:
            raise CTClientError(f"ClinicalTrials.gov request failed: {exc}") from exc

        if resp.status_code != 200:
            detail = resp.text.strip()
            raise CTClientError(
                f"ClinicalTrials.gov returned HTTP {resp.status_code}: {detail[:300]}",
                status_code=resp.status_code,
            )
        try:
            return resp.json()
        except ValueError as exc:
            raise CTClientError(
                f"ClinicalTrials.gov returned non-JSON response: {resp.text[:300]}"
            ) from exc

    async def search_studies(
        self,
        *,
        query_term: str | None = None,
        condition: str | None = None,
        intervention: str | None = None,
        sponsor: str | None = None,
        status: str | Iterable[str] | None = None,
        advanced_filter: str | None = None,
        fields: str | Iterable[str] | None = None,
        extra_params: dict[str, Any] | None = None,
        max_studies: int | None = None,
        count_total: bool = True,
    ) -> StudySearchResult:
        """Search ``/studies``, following pagination up to the configured cap.

        Friendly args map to API params:
          ``query_term`` -> ``query.term`` (free text)        ``condition`` -> ``query.cond``
          ``intervention`` -> ``query.intr``                  ``sponsor`` -> ``query.spons``
          ``status`` -> ``filter.overallStatus`` (e.g. "RECRUITING")
          ``advanced_filter`` -> ``filter.advanced`` (Essie, e.g. "AREA[Phase]PHASE3")
          ``fields`` -> ``fields`` (short names; list or csv)
        Anything else goes through ``extra_params``.

        The number of studies is capped at ``min(max_studies, page_size * max_pages)``
        to bound cost; ``StudySearchResult.truncated`` reports whether matches were
        left out.
        """
        base_params: dict[str, Any] = {}
        if query_term:
            base_params["query.term"] = query_term
        if condition:
            base_params["query.cond"] = condition
        if intervention:
            base_params["query.intr"] = intervention
        if sponsor:
            base_params["query.spons"] = sponsor
        if status:
            base_params["filter.overallStatus"] = _as_csv(status)
        if advanced_filter:
            base_params["filter.advanced"] = advanced_filter
        if fields:
            base_params["fields"] = _as_csv(fields)
        if extra_params:
            base_params.update(extra_params)

        cap = self.page_size * self.max_pages if max_studies is None else max_studies
        base_params["pageSize"] = max(1, min(self.page_size, cap))

        studies: list[dict[str, Any]] = []
        total_count: int | None = None
        page_token: str | None = None
        pages_fetched = 0

        while pages_fetched < self.max_pages and len(studies) < cap:
            params = dict(base_params)
            if count_total and pages_fetched == 0:
                params["countTotal"] = "true"
            if page_token:
                params["pageToken"] = page_token

            payload = await self._get("/studies", params)
            if total_count is None:
                total_count = payload.get("totalCount")
            studies.extend(payload.get("studies", []))
            pages_fetched += 1

            page_token = payload.get("nextPageToken")
            if not page_token:
                break

        if len(studies) > cap:
            studies = studies[:cap]

        truncated = bool(page_token) or (
            total_count is not None and len(studies) < total_count
        )
        return StudySearchResult(
            studies=studies,
            total_count=total_count,
            pages_fetched=pages_fetched,
            truncated=truncated,
        )

    async def get_study(
        self, nct_id: str, *, fields: str | Iterable[str] | None = None
    ) -> dict[str, Any]:
        """Fetch a single study by NCT id.

        Returns the raw study dict (top-level ``{"protocolSection": ...}``). Used
        to enrich citation excerpts when a richer snippet than the search fields
        is wanted.
        """
        params: dict[str, Any] = {}
        if fields:
            params["fields"] = _as_csv(fields)
        return await self._get(f"/studies/{nct_id}", params)

    async def get_field_values(
        self, fields: str | Iterable[str], *, types: str | Iterable[str] | None = None
    ) -> list[dict[str, Any]]:
        """``GET /stats/field/values`` — GLOBAL value distribution for field(s).

        Returns a list with one entry per field, each like::

            {"field": "protocolSection.designModule.phases", "piece": "Phase",
             "type": "ENUM", "topValues": [{"value": "PHASE2", "studiesCount": 88710}, ...]}

        NOTE: this endpoint ignores query/filter params (counts are global) and
        carries no nct_ids, so it cannot support scoped distributions or
        citations. Prefer :meth:`search_studies` + client-side aggregation for
        anything that needs a filter or citations.
        """
        params: dict[str, Any] = {"fields": _as_csv(fields)}
        if types:
            params["types"] = _as_csv(types)
        return await self._get("/stats/field/values", params)
