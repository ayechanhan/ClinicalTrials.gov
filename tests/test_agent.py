"""Tests for the scaffold and the deterministic half of the pipeline.

These run fully offline: the planner (LLM) and CT.gov client are *not* invoked.
The assembler + spec_builder are exercised on a synthetic ``FetchResult`` so the
output contract — including per-data-point citations — is pinned without network.
End-to-end tests against the live APIs are run manually (see Step 4 demo).
"""

from fastapi.testclient import TestClient

import app.main as main_module
from app.agent.assembler import assemble
from app.agent.planner import ApiEndpoint, ApiStrategy, ExtractedParams, PlannerError, QueryPlan
from app.agent.tools import Bucket, FetchError, FetchResult
from app.ct_client.client import CTClientError
from app.main import app
from app.schemas import IntentClass, QueryRequest, VizType

client = TestClient(app)


def _study(nct: str, title: str, *, date: str | None = None,
           phases: list[str] | None = None, status: str | None = None) -> dict:
    ps: dict = {"identificationModule": {"nctId": nct, "briefTitle": title}}
    if date:
        ps["statusModule"] = {"startDateStruct": {"date": date}}
    if status:
        ps.setdefault("statusModule", {})["overallStatus"] = status
    if phases:
        ps["designModule"] = {"phases": phases}
    return {"protocolSection": ps}


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
            Bucket(key=2016, label="2016", count=5, sample_studies=[_study("NCT2", "B", date="2016-05")]),
            Bucket(key=2015, label="2015", count=2, sample_studies=[_study("NCT1", "A", date="2015-03")]),
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
    # Deep citation: excerpt carries the exact field value tying the study to its bucket.
    assert "Start date: 2015-03" in viz["data"][0]["citations"][0]["excerpt"]
    assert body["meta"]["total_trials_fetched"] == 7


def test_assemble_bar_chart_sorted_desc_drops_empty() -> None:
    plan = _plan(IntentClass.DISTRIBUTION, VizType.BAR_CHART, "phase", condition="lung cancer")
    result = FetchResult(
        buckets=[
            Bucket(key="PHASE2", label="Phase 2", count=10,
                   sample_studies=[_study("NCT2", "B", phases=["PHASE2"])]),
            Bucket(key="PHASE3", label="Phase 3", count=25,
                   sample_studies=[_study("NCT3", "C", phases=["PHASE3"])]),
        ],
        total_trials=35, group_by="phase",
    )
    viz = assemble(plan, result).model_dump(mode="json")["visualization"]
    assert viz["type"] == "bar_chart"
    # Sorted by count descending.
    assert [(d["phase"], d["trial_count"]) for d in viz["data"]] == [("Phase 3", 25), ("Phase 2", 10)]
    # Deep citation carries the exact phase value.
    assert "Phase: PHASE3" in viz["data"][0]["citations"][0]["excerpt"]


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


def test_assemble_network() -> None:
    plan = _plan(IntentClass.NETWORK, VizType.NETWORK_GRAPH, "sponsor_drug", condition="breast cancer")
    result = FetchResult(
        buckets=[], total_trials=100, group_by="sponsor_drug",
        nodes=[
            {"id": "sponsor:Merck", "label": "Merck", "type": "sponsor"},
            {"id": "drug:Pembrolizumab", "label": "Pembrolizumab", "type": "drug"},
        ],
        edges=[{
            "source": "sponsor:Merck", "target": "drug:Pembrolizumab", "weight": 7,
            "citations": [{"nct_id": "NCT1", "excerpt": "A breast cancer trial"}],
        }],
    )
    viz = assemble(plan, result).model_dump(mode="json")["visualization"]
    assert viz["type"] == "network_graph"
    assert viz["encoding"]["nodes"] == "nodes" and viz["encoding"]["edges"] == "edges"
    graph = viz["data"][0]
    assert len(graph["nodes"]) == 2 and len(graph["edges"]) == 1
    # Per-edge citations (the studies that created the link) are preserved...
    assert graph["edges"][0]["citations"][0]["nct_id"] == "NCT1"
    # ...and the container DataPoint still carries citations (the contract).
    assert graph["citations"][0]["nct_id"] == "NCT1"


# --------------------------------------------------------------------------- #
# Input validation + error mapping (offline)
# --------------------------------------------------------------------------- #
def test_empty_query_rejected() -> None:
    assert client.post("/query", json={"query": "   "}).status_code == 422


def test_bad_year_range_rejected() -> None:
    resp = client.post("/query", json={"query": "x", "start_year": 2020, "end_year": 2010})
    assert resp.status_code == 422


def test_out_of_range_year_rejected() -> None:
    assert client.post("/query", json={"query": "x", "start_year": 1500}).status_code == 422


def test_unknown_phase_rejected() -> None:
    assert client.post("/query", json={"query": "x", "trial_phase": "PHASE9"}).status_code == 422


def test_phase_normalized() -> None:
    assert QueryRequest(query="x", trial_phase="phase 3").trial_phase == "PHASE3"
    assert QueryRequest(query="x", trial_phase="early phase 1").trial_phase == "EARLY_PHASE1"


def test_planner_error_maps_to_500(monkeypatch) -> None:
    async def boom(*_a, **_k):
        raise PlannerError("llm unavailable")
    monkeypatch.setattr(main_module, "plan_query", boom)
    assert client.post("/query", json={"query": "x"}).status_code == 500


def test_ct_error_maps_to_502(monkeypatch) -> None:
    async def ok_plan(*_a, **_k):
        return _plan(IntentClass.TIME_TREND, VizType.TIME_SERIES, "year")

    async def ct_down(*_a, **_k):
        raise CTClientError("upstream timeout")

    monkeypatch.setattr(main_module, "plan_query", ok_plan)
    monkeypatch.setattr(main_module, "fetch_for_plan", ct_down)
    assert client.post("/query", json={"query": "x"}).status_code == 502


def test_fetch_error_maps_to_422(monkeypatch) -> None:
    async def ok_plan(*_a, **_k):
        return _plan(IntentClass.TIME_TREND, VizType.TIME_SERIES, "year")

    async def unsupported(*_a, **_k):
        raise FetchError("unsupported viz")

    monkeypatch.setattr(main_module, "plan_query", ok_plan)
    monkeypatch.setattr(main_module, "fetch_for_plan", unsupported)
    assert client.post("/query", json={"query": "x"}).status_code == 422
