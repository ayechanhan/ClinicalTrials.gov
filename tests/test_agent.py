"""Tests for the scaffold and the deterministic half of the pipeline.

These run fully offline: the planner (LLM) and CT.gov client are *not* invoked.
The assembler + spec_builder are exercised on a synthetic ``FetchResult`` so the
output contract — including per-data-point citations — is pinned without network.
End-to-end tests against the live APIs are run manually (see Step 4 demo).
"""

from fastapi.testclient import TestClient

from app.agent.assembler import assemble
from app.agent.planner import ApiEndpoint, ApiStrategy, ExtractedParams, QueryPlan
from app.agent.tools import Bucket, FetchResult
from app.main import app
from app.schemas import IntentClass, VizType

client = TestClient(app)


def _study(nct: str, title: str) -> dict:
    return {"protocolSection": {"identificationModule": {"nctId": nct, "briefTitle": title}}}


def _params(**overrides) -> ExtractedParams:
    base = dict(
        drug_name=None, condition=None, phase=None, status=None, sponsor=None,
        country=None, start_year=None, end_year=None, comparison_items=[],
    )
    base.update(overrides)
    return ExtractedParams(**base)


def _plan(intent: IntentClass, viz: VizType, group_by: str, **param_overrides) -> QueryPlan:
    return QueryPlan(
        intent_class=intent,
        viz_type=viz,
        extracted_params=_params(**param_overrides),
        api_strategy=ApiStrategy(endpoint=ApiEndpoint.STUDIES, group_by_field=group_by, notes="test"),
    )


# --------------------------------------------------------------------------- #
# Offline contract tests (no network)
# --------------------------------------------------------------------------- #
def test_health() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_query_requires_query_field() -> None:
    # Missing required `query` -> 422 from request validation, before any LLM call.
    assert client.post("/query", json={}).status_code == 422


def test_assemble_time_series() -> None:
    plan = _plan(IntentClass.TIME_TREND, VizType.TIME_SERIES, "year",
                 drug_name="Pembrolizumab", start_year=2015, end_year=2016)
    result = FetchResult(
        buckets=[
            Bucket(key=2016, label="2016", count=5, sample_studies=[_study("NCT2", "B")]),
            Bucket(key=2015, label="2015", count=2, sample_studies=[_study("NCT1", "A")]),
        ],
        total_trials=7, group_by="year",
    )
    body = assemble(plan, result).model_dump(mode="json")

    viz = body["visualization"]
    assert viz["type"] == "time_series"
    assert viz["encoding"]["x"]["field"] == "year"
    # Sorted ascending by year, exact counts preserved.
    assert [(d["year"], d["trial_count"]) for d in viz["data"]] == [(2015, 2), (2016, 5)]
    # Every data point carries citations with nct_id + excerpt.
    for d in viz["data"]:
        assert d["citations"] and all({"nct_id", "excerpt"} <= set(c) for c in d["citations"])
    assert body["meta"]["total_trials_fetched"] == 7


def test_assemble_bar_chart_sorted_desc_drops_empty() -> None:
    plan = _plan(IntentClass.DISTRIBUTION, VizType.BAR_CHART, "phase", condition="lung cancer")
    result = FetchResult(
        buckets=[
            Bucket(key="PHASE2", label="Phase 2", count=10, sample_studies=[_study("NCT2", "B")]),
            Bucket(key="PHASE3", label="Phase 3", count=25, sample_studies=[_study("NCT3", "C")]),
        ],
        total_trials=35, group_by="phase",
    )
    viz = assemble(plan, result).model_dump(mode="json")["visualization"]
    assert viz["type"] == "bar_chart"
    # Sorted by count descending.
    assert [(d["phase"], d["trial_count"]) for d in viz["data"]] == [("Phase 3", 25), ("Phase 2", 10)]


def test_assemble_grouped_bar_has_series() -> None:
    plan = _plan(IntentClass.COMPARISON, VizType.GROUPED_BAR, "phase",
                 comparison_items=["Pembrolizumab", "Nivolumab"])
    result = FetchResult(
        buckets=[
            Bucket(key="PHASE3", label="Phase 3", count=20, series="Pembrolizumab",
                   sample_studies=[_study("NCT1", "A")]),
            Bucket(key="PHASE3", label="Phase 3", count=12, series="Nivolumab",
                   sample_studies=[_study("NCT2", "B")]),
        ],
        total_trials=32, group_by="phase",
    )
    viz = assemble(plan, result).model_dump(mode="json")["visualization"]
    assert viz["type"] == "grouped_bar"
    assert viz["encoding"]["series"] == "series"
    assert {d["series"] for d in viz["data"]} == {"Pembrolizumab", "Nivolumab"}
