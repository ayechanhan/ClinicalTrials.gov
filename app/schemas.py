"""
Pydantic models defining the public contract for the ClinicalTrials.gov
Query-to-Visualization service.

This module is the single source of truth for the request/response shapes. A
frontend should be able to render any response by reading only these models
(plus the OpenAPI schema generated from them) without guessing.

Design notes
------------
* The output is a *visualization specification*, not pre-rendered pixels. It
  names the chart ``type``, the field ``encoding`` (which data key maps to x, y,
  series, etc.), and the ``data`` rows. This keeps the backend renderer-agnostic.
* Every ``DataPoint`` carries its own ``citations`` list so each bar / point /
  edge is traceable back to source studies (nct_id + excerpt). Citations are a
  first-class, always-present field — not bolted on later (see CLAUDE.md
  "Deep Citations").
* ``IntentClass`` and ``VizType`` are closed enums, so the planner's output and
  the emitted spec are self-validating and self-documenting.
"""

import re
from datetime import date
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# --------------------------------------------------------------------------- #
# Controlled vocabularies
# --------------------------------------------------------------------------- #
class IntentClass(str, Enum):
    """High-level question categories the planner classifies a query into."""

    TIME_TREND = "time_trend"
    DISTRIBUTION = "distribution"
    COMPARISON = "comparison"
    GEOGRAPHIC = "geographic"
    NETWORK = "network"


class VizType(str, Enum):
    """Visualization specifications the service can emit."""

    TIME_SERIES = "time_series"
    BAR_CHART = "bar_chart"
    GROUPED_BAR = "grouped_bar"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    NETWORK_GRAPH = "network_graph"


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
# Accepted trial-phase inputs, normalized to the API's enum values.
_PHASE_ALIASES = {
    "PHASE1": "PHASE1", "1": "PHASE1",
    "PHASE2": "PHASE2", "2": "PHASE2",
    "PHASE3": "PHASE3", "3": "PHASE3",
    "PHASE4": "PHASE4", "4": "PHASE4",
    "EARLYPHASE1": "EARLY_PHASE1", "EARLY1": "EARLY_PHASE1",
    "NA": "NA", "NOTAPPLICABLE": "NA",
}
_MIN_YEAR = 1900


class QueryRequest(BaseModel):
    """A natural-language question plus optional structured overrides.

    ``query`` is required. The optional fields let a caller pin an entity
    explicitly (e.g. a UI with dropdowns) instead of relying on the planner to
    extract it from ``query``. When provided, they take precedence over the
    planner's extraction. Strict value validation (empty query, year ranges,
    known phase values) is added in the hardening step.
    """

    query: str = Field(..., description="Natural-language question about clinical trials.")
    drug_name: str | None = Field(default=None, description="Intervention / drug name override.")
    condition: str | None = Field(default=None, description="Disease or condition override.")
    trial_phase: str | None = Field(
        default=None,
        description='Trial phase override, e.g. "PHASE1", "PHASE2", "PHASE3", "PHASE4".',
    )
    sponsor: str | None = Field(default=None, description="Sponsor / lead organization override.")
    country: str | None = Field(default=None, description="Country filter override.")
    start_year: int | None = Field(default=None, description="Earliest study start year (inclusive).")
    end_year: int | None = Field(default=None, description="Latest study start year (inclusive).")

    @field_validator("query")
    @classmethod
    def _query_not_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must not be empty")
        return cleaned

    @field_validator("trial_phase")
    @classmethod
    def _normalize_phase(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = _PHASE_ALIASES.get(re.sub(r"[\s_/]", "", value.upper()))
        if normalized is None:
            raise ValueError(
                f"unknown trial_phase {value!r}; expected one of "
                "PHASE1, PHASE2, PHASE3, PHASE4, EARLY_PHASE1, NA"
            )
        return normalized

    @field_validator("start_year", "end_year")
    @classmethod
    def _year_in_range(cls, value: int | None) -> int | None:
        if value is None:
            return None
        ceiling = date.today().year + 5
        if not (_MIN_YEAR <= value <= ceiling):
            raise ValueError(f"year {value} out of range [{_MIN_YEAR}, {ceiling}]")
        return value

    @model_validator(mode="after")
    def _check_year_order(self) -> "QueryRequest":
        if self.start_year is not None and self.end_year is not None and self.start_year > self.end_year:
            raise ValueError(f"start_year ({self.start_year}) must be <= end_year ({self.end_year})")
        return self

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "How has the number of trials for Pembrolizumab changed per year since 2015?",
                    "start_year": 2015,
                }
            ]
        }
    )


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
class Citation(BaseModel):
    """A single source reference backing a data point.

    ``excerpt`` is an *exact* value or text snippet taken from the API response
    (e.g. a Phase value or BriefTitle) — never paraphrased or model-generated.
    """

    nct_id: str = Field(..., description="ClinicalTrials.gov study id, e.g. 'NCT01234567'.")
    excerpt: str = Field(..., description="Exact field value or text snippet from the study record.")


class DataPoint(BaseModel):
    """One row of visualization data.

    Encoding keys vary by viz type — e.g. ``{"year": 2019, "trial_count": 42}``
    for a time series, or ``{"phase": "PHASE3", "trial_count": 41}`` for a bar
    chart — so arbitrary keys are allowed via ``extra="allow"``. ``citations`` is
    always present and typed, so every point is traceable to its source studies.
    """

    model_config = ConfigDict(extra="allow")

    citations: list[Citation] = Field(
        default_factory=list,
        description="Source studies for this data point (typically capped at 5).",
    )


class Encoding(BaseModel):
    """Maps data keys to visual channels so a renderer needs no guesswork.

    Most charts use ``x``/``y`` (e.g. ``{"field": "year"}``). ``series`` adds a
    grouping dimension (grouped bar). ``nodes``/``edges`` name the data keys that
    carry a network's node and edge collections.
    """

    x: dict | None = Field(default=None, description='e.g. {"field": "year", "type": "temporal"}')
    y: dict | None = Field(default=None, description='e.g. {"field": "trial_count", "type": "quantitative"}')
    series: str | None = Field(default=None, description="Data key used to split series (grouped bar).")
    nodes: str | None = Field(default=None, description="Data key holding node objects (network graph).")
    edges: str | None = Field(default=None, description="Data key holding edge objects (network graph).")


class VisualizationSpec(BaseModel):
    """The renderable chart specification: what to draw, and the data to draw."""

    type: VizType = Field(..., description="Chart type the frontend should render.")
    title: str = Field(..., description="Human-readable chart title.")
    encoding: Encoding = Field(..., description="Field-to-channel mapping.")
    data: list[DataPoint] = Field(default_factory=list, description="Ordered data rows.")


class ResponseMeta(BaseModel):
    """Provenance and interpretation metadata for transparency/debuggability."""

    filters: dict = Field(default_factory=dict, description="Effective filters applied to the CT.gov query.")
    source: str = Field(default="clinicaltrials.gov", description="Upstream data source.")
    query_interpretation: str = Field(..., description="Plain-language restatement of what the planner understood.")
    total_trials_fetched: int = Field(..., ge=0, description="Number of studies fetched to build this response.")
    notes: str | None = Field(default=None, description="Caveats, fallbacks, or warnings (e.g. capped pages).")


class VisualizationResponse(BaseModel):
    """Top-level response: the visualization spec plus its metadata."""

    visualization: VisualizationSpec
    meta: ResponseMeta
