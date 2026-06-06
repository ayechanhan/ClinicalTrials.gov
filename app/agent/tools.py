"""Fetcher — turns a ``QueryPlan`` into ClinicalTrials.gov queries and returns the
aggregated buckets the visualization needs.

Accuracy strategy
-----------------
Rather than fetching a capped *sample* of studies and grouping them client-side
(which under-counts large result sets and skews trends — e.g. only 1000 of 2870
Pembrolizumab trials would distort a per-year line), we issue one **count** query
per bucket (``countTotal=true``) for an EXACT count, plus a tiny sample of studies
per bucket for citations. The bucket dimensions handled here are bounded (years,
phases), so the number of concurrent count queries stays small.

Implemented viz types: time_series (by year), bar_chart (distribution by a bounded
enum field), grouped_bar (comparison across phases). network/geographic come in
Step 5.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from app.agent.planner import ExtractedParams, QueryPlan
from app.ct_client.client import ClinicalTrialsClient, StudySearchResult
from app.schemas import VizType

logger = logging.getLogger(__name__)

# Fields requested per bucket sample (kept small — we only need them for citations).
CITATION_FIELDS = ["NCTId", "BriefTitle", "Phase", "StartDate", "OverallStatus", "LeadSponsorName"]
SAMPLE_SIZE = 5            # sample studies per bucket (for citations)
MAX_YEARS = 30            # safety cap on time-series buckets
CONCURRENCY = 8           # polite cap on simultaneous CT.gov requests

# Bounded enum buckets. (api_value, display_label) in canonical display order.
PHASE_BUCKETS: list[tuple[str, str]] = [
    ("EARLY_PHASE1", "Early Phase 1"),
    ("PHASE1", "Phase 1"),
    ("PHASE2", "Phase 2"),
    ("PHASE3", "Phase 3"),
    ("PHASE4", "Phase 4"),
    ("NA", "Not Applicable"),
]
STATUS_BUCKETS: list[tuple[str, str]] = [
    ("RECRUITING", "Recruiting"),
    ("NOT_YET_RECRUITING", "Not Yet Recruiting"),
    ("ACTIVE_NOT_RECRUITING", "Active, Not Recruiting"),
    ("ENROLLING_BY_INVITATION", "Enrolling by Invitation"),
    ("COMPLETED", "Completed"),
    ("SUSPENDED", "Suspended"),
    ("TERMINATED", "Terminated"),
    ("WITHDRAWN", "Withdrawn"),
    ("UNKNOWN", "Unknown"),
]


@dataclass
class Bucket:
    """One aggregated bucket: an exact count plus a small citation sample."""

    key: Any                                  # raw key (year int, phase enum value, ...)
    label: str                                # display label
    count: int                                # EXACT count from countTotal
    sample_studies: list[dict[str, Any]]      # up to SAMPLE_SIZE raw studies (for citations)
    series: str | None = None                 # series name (grouped_bar), else None


@dataclass
class FetchResult:
    """Everything the assembler needs to build a spec, minus presentation."""

    buckets: list[Bucket]
    total_trials: int                         # size of the queried universe
    group_by: str                             # final group-by dimension name
    notes: list[str] = field(default_factory=list)


class FetchError(RuntimeError):
    """Raised when the plan's fetch strategy cannot be satisfied (e.g. unsupported)."""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _base_kwargs(params: ExtractedParams) -> tuple[dict[str, Any], list[str]]:
    """Map extracted params to ``search_studies`` kwargs + advanced Essie clauses."""
    kwargs: dict[str, Any] = {}
    if params.drug_name:
        kwargs["intervention"] = params.drug_name
    if params.condition:
        kwargs["condition"] = params.condition
    if params.sponsor:
        kwargs["sponsor"] = params.sponsor
    if params.status:
        kwargs["status"] = params.status
    clauses: list[str] = []
    if params.phase:
        clauses.append(f"AREA[Phase]{params.phase}")
    return kwargs, clauses


def _combine(clauses: list[str]) -> str | None:
    """Join non-empty Essie clauses with AND, or return None."""
    real = [c for c in clauses if c]
    return " AND ".join(real) if real else None


async def _gather_limited(coros: list[Any], limit: int = CONCURRENCY) -> list[Any]:
    """gather() with a concurrency cap, so we don't hammer the CT.gov API."""
    sem = asyncio.Semaphore(limit)

    async def _run(coro: Any) -> Any:
        async with sem:
            return await coro

    return await asyncio.gather(*(_run(c) for c in coros))


async def _count_bucket(
    client: ClinicalTrialsClient,
    *,
    key: Any,
    label: str,
    base_kwargs: dict[str, Any],
    clauses: list[str],
    series: str | None = None,
) -> Bucket:
    """Run one exact-count query (+ small sample) for a single bucket."""
    res: StudySearchResult = await client.search_studies(
        **base_kwargs,
        advanced_filter=_combine(clauses),
        fields=CITATION_FIELDS,
        max_studies=SAMPLE_SIZE,
        count_total=True,
    )
    return Bucket(key=key, label=label, count=res.total_count or 0,
                  sample_studies=res.studies, series=series)


async def _base_total(client: ClinicalTrialsClient, base_kwargs: dict[str, Any],
                      clauses: list[str] | None = None) -> int:
    """Total matches for the base filters (the size of the visualized universe)."""
    res = await client.search_studies(
        **base_kwargs,
        advanced_filter=_combine(clauses or []),
        fields=["NCTId"],
        max_studies=1,
        count_total=True,
    )
    return res.total_count or 0


# --------------------------------------------------------------------------- #
# Per-viz fetch strategies
# --------------------------------------------------------------------------- #
async def _fetch_time_series(plan: QueryPlan, client: ClinicalTrialsClient) -> FetchResult:
    params = plan.extracted_params
    base_kwargs, base_clauses = _base_kwargs(params)

    start = params.start_year or (date.today().year - 10)
    end = params.end_year or date.today().year
    if end < start:
        start, end = end, start
    years = list(range(start, end + 1))
    notes: list[str] = []
    if len(years) > MAX_YEARS:
        years = years[-MAX_YEARS:]
        notes.append(f"Year range capped to the most recent {MAX_YEARS} years.")

    # Scope the headline total to the same date window the chart covers, so
    # total_trials_fetched matches the sum of the year buckets.
    range_clause = f"AREA[StartDate]RANGE[{years[0]}-01-01,{years[-1]}-12-31]"
    coros = [_base_total(client, base_kwargs, base_clauses + [range_clause])]
    coros += [
        _count_bucket(
            client, key=y, label=str(y), base_kwargs=base_kwargs,
            clauses=base_clauses + [f"AREA[StartDate]RANGE[{y}-01-01,{y}-12-31]"],
        )
        for y in years
    ]
    results = await _gather_limited(coros)
    total, buckets = results[0], list(results[1:])
    return FetchResult(buckets=buckets, total_trials=total, group_by="year", notes=notes)


def _distribution_buckets(group_by: str) -> tuple[list[tuple[str, str]], str, str]:
    """Resolve a group-by field to (bucket defs, Essie area name, output field name)."""
    gb = group_by.lower()
    if gb in ("phase", "phases"):
        return PHASE_BUCKETS, "Phase", "phase"
    if gb in ("status", "overallstatus", "overall_status"):
        return STATUS_BUCKETS, "OverallStatus", "status"
    raise FetchError(
        f"Distribution by '{group_by}' is not supported among the core viz types "
        f"(country/other high-cardinality fields arrive in Step 5)."
    )


async def _fetch_distribution(plan: QueryPlan, client: ClinicalTrialsClient) -> FetchResult:
    params = plan.extracted_params
    bucket_defs, area, out_field = _distribution_buckets(plan.api_strategy.group_by_field or "phase")
    base_kwargs, base_clauses = _base_kwargs(params)

    coros = [_base_total(client, base_kwargs, base_clauses)]
    coros += [
        _count_bucket(client, key=val, label=label, base_kwargs=base_kwargs,
                      clauses=base_clauses + [f"AREA[{area}]{val}"])
        for val, label in bucket_defs
    ]
    results = await _gather_limited(coros)
    total = results[0]
    buckets = [b for b in results[1:] if b.count > 0]
    notes: list[str] = []
    if area == "Phase":
        notes.append(
            "A trial can be registered under multiple phases (or none), so per-phase "
            "counts need not sum to the total trial count."
        )
    return FetchResult(buckets=buckets, total_trials=total, group_by=out_field, notes=notes)


async def _fetch_comparison(plan: QueryPlan, client: ClinicalTrialsClient) -> FetchResult:
    params = plan.extracted_params
    items = params.comparison_items or [i for i in [params.drug_name] if i]
    if len(items) < 2:
        raise FetchError("A comparison needs at least two items (comparison_items).")

    bucket_defs, area, out_field = PHASE_BUCKETS, "Phase", "phase"  # core comparison: across phases
    common_kwargs, base_clauses = _base_kwargs(params)             # condition/status shared across items

    # Base total per item (series); each item overrides the intervention filter.
    item_kwargs = [{**common_kwargs, "intervention": item} for item in items]
    base_coros = [_base_total(client, ik, base_clauses) for ik in item_kwargs]

    # Bucket counts, phase-major then series, for natural grouped-bar ordering.
    bucket_coros = [
        _count_bucket(client, key=val, label=label, base_kwargs=ik,
                      clauses=base_clauses + [f"AREA[{area}]{val}"], series=item)
        for val, label in bucket_defs
        for item, ik in zip(items, item_kwargs)
    ]

    results = await _gather_limited(base_coros + bucket_coros)
    totals = results[: len(items)]
    buckets = [b for b in results[len(items):] if b.count > 0]
    notes = ["Per-item totals may overlap when a trial studies more than one of the compared drugs."]
    return FetchResult(buckets=buckets, total_trials=sum(totals), group_by=out_field, notes=notes)


async def fetch_for_plan(plan: QueryPlan, client: ClinicalTrialsClient) -> FetchResult:
    """Dispatch to the right fetch strategy based on the plan's viz type."""
    viz = plan.viz_type
    if viz == VizType.TIME_SERIES:
        return await _fetch_time_series(plan, client)
    if viz == VizType.GROUPED_BAR:
        return await _fetch_comparison(plan, client)
    if viz == VizType.BAR_CHART:
        return await _fetch_distribution(plan, client)
    raise FetchError(
        f"viz_type '{viz.value}' is not implemented yet (network_graph/geographic/"
        f"histogram/scatter arrive in Step 5)."
    )
