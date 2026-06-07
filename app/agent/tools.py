"""Fetcher — turns a ``QueryPlan`` into ClinicalTrials.gov queries and returns the
aggregated data the visualization needs.

Accuracy strategy
-----------------
For bounded dimensions (years, phases, status) we issue one **count** query per
bucket (``countTotal=true``) for an EXACT count, plus a tiny sample of studies per
bucket for citations — never grouping a capped sample (which would skew counts).

* time_series  : exact count per year (StartDate RANGE filter).
* bar_chart    : exact count per bounded enum value (phase / status).
* grouped_bar  : exact count per (item, phase) cell.
* geographic   : countries are high-cardinality, so we discover candidate countries
                 from a study sample, then take EXACT counts for the top candidates.
* network_graph: fetch a study sample and build sponsor<->drug edges client-side.

network/geographic come with their own truncation/overlap notes.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from app.agent.planner import ExtractedParams, QueryPlan
from app.ct_client.client import ClinicalTrialsClient, StudySearchResult
from app.schemas import IntentClass, VizType

logger = logging.getLogger(__name__)

# Fields requested per bucket sample (kept small — we only need them for citations).
CITATION_FIELDS = ["NCTId", "BriefTitle", "Phase", "StartDate", "OverallStatus", "LeadSponsorName"]
NETWORK_FIELDS = ["NCTId", "BriefTitle", "LeadSponsorName", "InterventionName", "InterventionType"]
SAMPLE_SIZE = 5             # sample studies per bucket (for citations)
EXCERPT_MAX = 200           # max citation excerpt length
MAX_YEARS = 30              # safety cap on time-series buckets
CONCURRENCY = 8             # polite cap on simultaneous CT.gov requests

# Geographic strategy knobs.
GEO_CANDIDATE_SAMPLE = 300  # studies sampled to discover candidate countries
GEO_TOP_CANDIDATES = 25     # candidate countries to exact-count
GEO_TOP_N = 15              # countries shown in the final chart

# Network strategy knobs.
NETWORK_SAMPLE = 200        # studies fetched to build the graph
NETWORK_TOP_EDGES = 40      # densest sponsor<->drug links to keep
DRUG_TYPES = {"DRUG", "BIOLOGICAL"}  # intervention types treated as "drugs"

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

    key: Any                                  # raw key (year int, phase enum value, country, ...)
    label: str                                # display label
    count: int                                # EXACT count from countTotal
    sample_studies: list[dict[str, Any]]      # up to SAMPLE_SIZE raw studies (for citations)
    series: str | None = None                 # series name (grouped_bar), else None


@dataclass
class FetchResult:
    """Everything the assembler needs to build a spec, minus presentation.

    ``buckets`` drives bar/line/grouped charts; ``nodes``/``edges`` drive the
    network graph (buckets is empty in that case).
    """

    buckets: list[Bucket]
    total_trials: int                         # size of the queried universe
    group_by: str                             # final group-by dimension name
    notes: list[str] = field(default_factory=list)
    nodes: list[dict[str, Any]] | None = None
    edges: list[dict[str, Any]] | None = None


class FetchError(RuntimeError):
    """Raised when the plan's fetch strategy cannot be satisfied (e.g. unsupported)."""


# --------------------------------------------------------------------------- #
# Study field extraction helpers
# --------------------------------------------------------------------------- #
def _protocol(study: dict) -> dict:
    return study.get("protocolSection", {})


def _study_countries(study: dict) -> set[str]:
    locations = _protocol(study).get("contactsLocationsModule", {}).get("locations", [])
    return {loc.get("country") for loc in locations if loc.get("country")}


def _study_sponsor(study: dict) -> str | None:
    return _protocol(study).get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name")


def _study_drugs(study: dict) -> list[str]:
    interventions = _protocol(study).get("armsInterventionsModule", {}).get("interventions", [])
    return [i["name"] for i in interventions if i.get("name") and i.get("type") in DRUG_TYPES]


def _study_ref(study: dict) -> dict[str, str] | None:
    ident = _protocol(study).get("identificationModule", {})
    nct = ident.get("nctId")
    if not nct:
        return None
    return {"nct_id": nct, "excerpt": (ident.get("briefTitle") or "").strip()[:EXCERPT_MAX] or nct}


# --------------------------------------------------------------------------- #
# Query-building helpers
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
    if params.country:
        safe_country = params.country.replace('"', "")  # keep the Essie string literal intact
        clauses.append(f'AREA[LocationCountry]"{safe_country}"')
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

    # Scope the headline total to the charted date window so it matches the bucket sum.
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
        f"Distribution by '{group_by}' is not supported (only bounded enum fields "
        f"like phase/status; country is handled by the geographic strategy)."
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

    item_kwargs = [{**common_kwargs, "intervention": item} for item in items]
    base_coros = [_base_total(client, ik, base_clauses) for ik in item_kwargs]
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


async def _fetch_geographic(plan: QueryPlan, client: ClinicalTrialsClient) -> FetchResult:
    params = plan.extracted_params
    base_kwargs, base_clauses = _base_kwargs(params)

    # 1) Sample studies to discover which countries actually appear for these filters.
    sample = await client.search_studies(
        **base_kwargs, advanced_filter=_combine(base_clauses),
        fields=["NCTId", "LocationCountry"], max_studies=GEO_CANDIDATE_SAMPLE, count_total=True,
    )
    total = sample.total_count or 0
    freq: Counter[str] = Counter()
    for study in sample.studies:
        freq.update(_study_countries(study))
    if not freq:
        return FetchResult(buckets=[], total_trials=total, group_by="country",
                           notes=["No location/country data was found for these filters."])

    # 2) Exact counts for the top candidate countries (quote names for multi-word).
    candidates = [country for country, _ in freq.most_common(GEO_TOP_CANDIDATES)]
    coros = [
        _count_bucket(client, key=country, label=country, base_kwargs=base_kwargs,
                      clauses=base_clauses + [f'AREA[LocationCountry]"{country}"'])
        for country in candidates
    ]
    buckets = [b for b in await _gather_limited(coros) if b.count > 0]
    buckets.sort(key=lambda b: b.count, reverse=True)
    buckets = buckets[:GEO_TOP_N]

    notes = [
        "A trial may run in multiple countries, so per-country counts need not sum to the total.",
        f"Showing the top {len(buckets)} countries by exact trial count "
        f"(candidates discovered from a {len(sample.studies)}-study sample).",
    ]
    return FetchResult(buckets=buckets, total_trials=total, group_by="country", notes=notes)


async def _fetch_network(plan: QueryPlan, client: ClinicalTrialsClient) -> FetchResult:
    params = plan.extracted_params
    base_kwargs, base_clauses = _base_kwargs(params)

    res = await client.search_studies(
        **base_kwargs, advanced_filter=_combine(base_clauses),
        fields=NETWORK_FIELDS, max_studies=NETWORK_SAMPLE, count_total=True,
    )
    total = res.total_count or 0

    # Build sponsor<->drug edges: weight = #trials linking that sponsor to that drug.
    weights: dict[tuple[str, str], int] = defaultdict(int)
    cites: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for study in res.studies:
        sponsor = _study_sponsor(study)
        if not sponsor:
            continue
        ref = _study_ref(study)
        for drug in _study_drugs(study):
            edge = (sponsor, drug)
            weights[edge] += 1
            if ref is not None and len(cites[edge]) < SAMPLE_SIZE:
                cites[edge].append(ref)

    top_edges = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:NETWORK_TOP_EDGES]
    edges: list[dict[str, Any]] = []
    used: set[tuple[str, str]] = set()
    for (sponsor, drug), weight in top_edges:
        edges.append({
            "source": f"sponsor:{sponsor}",
            "target": f"drug:{drug}",
            "weight": weight,
            "citations": cites[(sponsor, drug)],
        })
        used.add(("sponsor", sponsor))
        used.add(("drug", drug))
    nodes = [{"id": f"{ntype}:{name}", "label": name, "type": ntype} for ntype, name in sorted(used)]

    notes = [
        f"Network built from a sample of {len(res.studies)} of {total} matching trials.",
        f"Showing the {len(edges)} strongest sponsor-drug links (of {len(weights)} found); "
        f"only DRUG/BIOLOGICAL interventions are treated as drugs.",
    ]
    return FetchResult(buckets=[], total_trials=total, group_by="sponsor_drug",
                       notes=notes, nodes=nodes, edges=edges)


async def fetch_for_plan(plan: QueryPlan, client: ClinicalTrialsClient) -> FetchResult:
    """Dispatch to the right fetch strategy based on the plan."""
    viz = plan.viz_type
    if viz == VizType.TIME_SERIES:
        return await _fetch_time_series(plan, client)
    if viz == VizType.GROUPED_BAR:
        return await _fetch_comparison(plan, client)
    if viz == VizType.NETWORK_GRAPH:
        return await _fetch_network(plan, client)
    if viz == VizType.BAR_CHART:
        group_by = (plan.api_strategy.group_by_field or "").lower()
        is_geo = plan.intent_class == IntentClass.GEOGRAPHIC or group_by in {
            "country", "countries", "location", "locationcountry", "location_country",
        }
        return await (_fetch_geographic if is_geo else _fetch_distribution)(plan, client)
    raise FetchError(
        f"viz_type '{viz.value}' is not implemented (histogram/scatter can be added "
        f"the same way as the existing shapers)."
    )
