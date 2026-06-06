"""Assembler — composes the final ``VisualizationResponse`` from a plan + fetch
result.

Deterministic for the core viz types (no LLM call): it builds a templated title,
maps the fetched buckets to ``DataPoint`` rows via ``spec_builder``, and records
provenance in ``meta``. It never invents trials or values — counts come straight
from the fetcher's exact bucket counts.
"""

from __future__ import annotations

from app.agent.planner import QueryPlan
from app.agent.tools import FetchResult
from app.schemas import (
    ResponseMeta,
    VisualizationResponse,
    VisualizationSpec,
    VizType,
)
from app.viz import spec_builder


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _title(plan: QueryPlan, result: FetchResult) -> str:
    p = plan.extracted_params
    if plan.viz_type == VizType.TIME_SERIES:
        subject = _cap((p.drug_name or p.condition or "Clinical trial").strip())
        years = [b.key for b in result.buckets]
        span = f" ({min(years)}-{max(years)})" if years else ""
        return f"{subject} trials by start year{span}"
    if plan.viz_type == VizType.GROUPED_BAR:
        versus = " vs ".join(p.comparison_items) if p.comparison_items else "comparison"
        context = f" for {p.condition}" if p.condition else ""
        return f"Trials by {result.group_by}: {versus}{context}"
    if plan.viz_type == VizType.NETWORK_GRAPH:
        subject = _cap((p.condition or p.drug_name or "clinical trial").strip())
        return f"Sponsor-drug network for {subject} trials"
    if plan.viz_type == VizType.BAR_CHART:
        subject = _cap((p.condition or p.drug_name or "Clinical trial").strip())
        return f"{subject} trials by {result.group_by}"
    return "ClinicalTrials.gov visualization"


def _interpretation(plan: QueryPlan, result: FetchResult) -> str:
    p = plan.extracted_params
    bits = [f"Interpreted as a {plan.intent_class.value.replace('_', ' ')} query"]
    subject = " vs ".join(p.comparison_items) if p.comparison_items else (p.drug_name or p.condition)
    if subject:
        bits.append(f"about {subject}")
    if p.status:
        bits.append(f"filtered to status {p.status}")
    bits.append(f"grouped by {result.group_by.replace('_', ' ')}")
    return ", ".join(bits) + "."


def _filters(plan: QueryPlan) -> dict:
    p = plan.extracted_params
    candidate = {
        "drug_name": p.drug_name,
        "condition": p.condition,
        "phase": p.phase,
        "status": p.status,
        "sponsor": p.sponsor,
        "country": p.country,
        "start_year": p.start_year,
        "end_year": p.end_year,
        "comparison_items": p.comparison_items or None,
    }
    return {k: v for k, v in candidate.items() if v}


def assemble(plan: QueryPlan, result: FetchResult) -> VisualizationResponse:
    """Build the typed response: spec (type/title/encoding/data) + provenance meta."""
    data, encoding = spec_builder.build_spec_data(result, plan.viz_type)

    spec = VisualizationSpec(
        type=plan.viz_type,
        title=_title(plan, result),
        encoding=encoding,
        data=data,
    )

    notes = list(result.notes)
    if not data:
        notes.append("No matching trials were found for these filters.")
    notes.append(
        "Counts are exact per bucket (ClinicalTrials.gov countTotal); "
        "up to 5 sample studies are cited per data point."
    )

    meta = ResponseMeta(
        filters=_filters(plan),
        query_interpretation=_interpretation(plan, result),
        total_trials_fetched=result.total_trials,
        notes=" ".join(notes),
    )
    return VisualizationResponse(visualization=spec, meta=meta)
