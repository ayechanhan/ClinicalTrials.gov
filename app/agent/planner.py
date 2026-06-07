"""Planner — interprets a natural-language query into a structured ``QueryPlan``.

A single OpenAI call with **structured outputs** (the response is parsed straight
into a Pydantic model) turns the raw query into:
  * ``intent_class``     — one of 5 question categories
  * ``viz_type``         — one of 6 chart types
  * ``extracted_params`` — entities pulled from the query (drug, condition, ...)
  * ``api_strategy``     — which CT.gov endpoint + group-by field to use

Why this step cannot hallucinate data
-------------------------------------
* The planner only ever sees the query text. It never sees API data and never
  emits counts or trial records — only the *plan*. All aggregation happens later,
  deterministically, over real API results.
* Structured outputs constrain every field to the schema (enums for intent/viz),
  so there is no free-form JSON to misparse.
* Explicit request overrides (``QueryRequest`` fields) are merged in
  deterministically *after* the LLM call and take precedence over extraction.
* A viz-type guard (the intent->viz table from CLAUDE.md) coerces any
  intent/viz mismatch back to a sensible default.
"""

from __future__ import annotations

import logging
from datetime import date
from enum import Enum

from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel, Field

from app.config import get_settings
from app.schemas import IntentClass, QueryRequest, VizType

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Planner output schema (internal to the agent — not the public API contract)
# --------------------------------------------------------------------------- #
class ApiEndpoint(str, Enum):
    STUDIES = "/studies"
    STATS_FIELD_VALUES = "/stats/field/values"


class ExtractedParams(BaseModel):
    """Entities pulled from the query.

    Every field is required by the structured-output schema but nullable, so the
    model must explicitly emit ``null`` rather than omit a key (no silent gaps).
    """

    drug_name: str | None = Field(description="Primary intervention/drug, or null.")
    condition: str | None = Field(description="Disease/condition, or null.")
    phase: str | None = Field(
        description="Normalized phase enum: PHASE1|PHASE2|PHASE3|PHASE4|EARLY_PHASE1, or null."
    )
    status: str | None = Field(
        description=(
            "Normalized overall status enum, e.g. RECRUITING, COMPLETED, "
            "ACTIVE_NOT_RECRUITING, NOT_YET_RECRUITING, TERMINATED, WITHDRAWN; or null."
        )
    )
    sponsor: str | None = Field(description="Sponsor / lead organization, or null.")
    country: str | None = Field(description="Country, or null.")
    start_year: int | None = Field(description="Earliest study start year, or null.")
    end_year: int | None = Field(description="Latest study start year, or null.")
    comparison_items: list[str] = Field(
        description=(
            "For comparison queries, the entities being compared "
            '(e.g. ["Pembrolizumab", "Nivolumab"]); otherwise an empty list.'
        )
    )


class ApiStrategy(BaseModel):
    """How the fetcher should hit the CT.gov API for this query."""

    endpoint: ApiEndpoint = Field(description="Which CT.gov endpoint to use.")
    group_by_field: str = Field(
        description="Dimension to aggregate by, e.g. 'year', 'phase', 'country', 'sponsor'."
    )
    notes: str = Field(description="One short sentence on filters / approach.")


class QueryPlan(BaseModel):
    """The full structured interpretation of a query."""

    intent_class: IntentClass
    viz_type: VizType
    extracted_params: ExtractedParams
    api_strategy: ApiStrategy


# --------------------------------------------------------------------------- #
# Intent -> viz mapping (fallback guard; the model picks, this enforces sanity)
# --------------------------------------------------------------------------- #
INTENT_DEFAULT_VIZ: dict[IntentClass, VizType] = {
    IntentClass.TIME_TREND: VizType.TIME_SERIES,
    IntentClass.DISTRIBUTION: VizType.BAR_CHART,
    IntentClass.COMPARISON: VizType.GROUPED_BAR,
    IntentClass.GEOGRAPHIC: VizType.BAR_CHART,
    IntentClass.NETWORK: VizType.NETWORK_GRAPH,
}

# Viz types we actually render, per intent. Anything else — including the
# not-yet-rendered histogram/scatter (kept in the contract for future use) — is
# coerced to the intent's default, so the pipeline never emits a spec it cannot build.
INTENT_COMPATIBLE_VIZ: dict[IntentClass, set[VizType]] = {
    IntentClass.TIME_TREND: {VizType.TIME_SERIES, VizType.BAR_CHART},
    IntentClass.DISTRIBUTION: {VizType.BAR_CHART},
    IntentClass.COMPARISON: {VizType.GROUPED_BAR, VizType.BAR_CHART},
    IntentClass.GEOGRAPHIC: {VizType.BAR_CHART},
    IntentClass.NETWORK: {VizType.NETWORK_GRAPH, VizType.BAR_CHART},
}


SYSTEM_PROMPT = """You are a planning component for a ClinicalTrials.gov data-visualization service.
Given a user's natural-language question about clinical trials, output a STRUCTURED PLAN only.
You never produce data, counts, or trial records — only how to interpret the question.

Do all of the following:

1) Classify intent_class (exactly one):
   - time_trend   : how something changes over time / per year
   - distribution : how trials split across categories of ONE field (phases, status, study type)
   - comparison   : compare TWO or more named entities (e.g. drug A vs drug B)
   - geographic   : by country / location
   - network      : relationships between entities (sponsors<->drugs, drugs<->conditions)

2) Choose viz_type (all of these are supported):
   - time_trend   -> time_series   (or bar_chart)
   - distribution -> bar_chart
   - comparison   -> grouped_bar   (or bar_chart)
   - geographic   -> bar_chart
   - network      -> network_graph

3) Extract entities into extracted_params. Use null for anything not stated. Do NOT invent.
   - Normalize phase to the API enum: "phase 3" / "phase III" -> "PHASE3"; also EARLY_PHASE1.
   - Normalize status to the API enum: "recruiting" -> RECRUITING, "completed" -> COMPLETED,
     "active" -> ACTIVE_NOT_RECRUITING, etc.; null if not mentioned.
   - Parse years: "since 2015" -> start_year=2015, end_year=null; "between 2018 and 2022" -> both.
   - For comparison queries, put the compared entities in comparison_items
     (e.g. "Pembrolizumab vs Nivolumab" -> two items); otherwise an empty list.

4) Pick api_strategy:
   - endpoint: use "/studies" for anything filtered by drug/condition/sponsor/country, or that
     needs citations (the usual case). Use "/stats/field/values" ONLY for an unfiltered global
     distribution of a single field.
   - group_by_field: the dimension to aggregate by — "year" (time_trend), the field for a
     distribution (e.g. "phase"), "country" (geographic), "sponsor" or "drug" (network),
     or the compared dimension (comparison).
   - notes: one short sentence on filters / approach.

Today's date is {today}."""


class PlannerError(RuntimeError):
    """Raised when the planner LLM call fails or returns no usable plan."""


_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    """Lazily build a shared AsyncOpenAI client (reuses one connection pool)."""
    global _client
    if _client is None:
        settings = get_settings()
        if not settings.openai_api_key:
            raise PlannerError("OPENAI_API_KEY is not set; add it to your .env.")
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


def _build_system_prompt() -> str:
    return SYSTEM_PROMPT.format(today=date.today().isoformat())


def _guard_viz_type(intent: IntentClass, viz: VizType) -> VizType:
    """Coerce an intent/viz mismatch back to the intent's default viz type."""
    if viz in INTENT_COMPATIBLE_VIZ[intent]:
        return viz
    default = INTENT_DEFAULT_VIZ[intent]
    logger.info(
        "viz_type '%s' is unusual for intent '%s'; using '%s'.",
        viz.value, intent.value, default.value,
    )
    return default


def _merge_overrides(params: ExtractedParams, request: QueryRequest) -> ExtractedParams:
    """Apply explicit request fields over the LLM's extraction (request wins)."""
    overrides = {
        "drug_name": request.drug_name,
        "condition": request.condition,
        "phase": request.trial_phase,
        "sponsor": request.sponsor,
        "country": request.country,
        "start_year": request.start_year,
        "end_year": request.end_year,
    }
    return params.model_copy(update={k: v for k, v in overrides.items() if v is not None})


PLANNER_ATTEMPTS = 2  # initial attempt + one retry on transient failure


async def plan_query(
    request: QueryRequest,
    *,
    client: AsyncOpenAI | None = None,
    model: str | None = None,
) -> QueryPlan:
    """Interpret ``request.query`` into a structured ``QueryPlan``.

    Retries once on a transient LLM failure (API error or empty parse); a refusal
    is surfaced immediately. Raises ``PlannerError`` if no usable plan is produced
    (the API layer maps that to HTTP 500).
    """
    client = client or _get_client()
    model = model or get_settings().openai_model
    messages = [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "user", "content": request.query},
    ]

    last_error: Exception | None = None
    for attempt in range(1, PLANNER_ATTEMPTS + 1):
        try:
            completion = await client.beta.chat.completions.parse(
                model=model, temperature=0, messages=messages, response_format=QueryPlan,
            )
        except OpenAIError as exc:
            last_error = exc
            logger.warning("Planner API error (attempt %d/%d): %s", attempt, PLANNER_ATTEMPTS, exc)
            continue

        message = completion.choices[0].message
        if getattr(message, "refusal", None):
            raise PlannerError(f"Planner refused the request: {message.refusal}")
        plan = message.parsed
        if plan is None:
            last_error = PlannerError("planner returned no parsed plan")
            logger.warning("Planner empty parse (attempt %d/%d)", attempt, PLANNER_ATTEMPTS)
            continue

        # Deterministic post-processing: viz-type guard + request-override merge.
        plan.viz_type = _guard_viz_type(plan.intent_class, plan.viz_type)
        plan.extracted_params = _merge_overrides(plan.extracted_params, request)
        return plan

    raise PlannerError(f"Planner failed after {PLANNER_ATTEMPTS} attempts: {last_error}")
