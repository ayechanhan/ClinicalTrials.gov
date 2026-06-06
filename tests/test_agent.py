"""Smoke tests for the scaffold.

As later steps add real behavior, this grows to smoke-test each query class
(time_trend, distribution, comparison, geographic, network). For now it pins the
Step 1 contract: the stub endpoint returns a schema-valid response with
per-data-point citations.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_query_stub_returns_valid_visualization() -> None:
    resp = client.post(
        "/query",
        json={"query": "How has the number of trials for Pembrolizumab changed per year since 2015?"},
    )
    assert resp.status_code == 200
    body = resp.json()

    # Top-level contract.
    assert set(body.keys()) == {"visualization", "meta"}

    viz = body["visualization"]
    assert viz["type"] == "time_series"
    assert viz["encoding"]["x"]["field"] == "year"
    assert len(viz["data"]) > 0

    # Bonus contract: every data point carries citations with nct_id + excerpt.
    for point in viz["data"]:
        assert "citations" in point
        for citation in point["citations"]:
            assert {"nct_id", "excerpt"} <= set(citation.keys())

    # Meta contract.
    assert body["meta"]["source"] == "clinicaltrials.gov"
    assert body["meta"]["total_trials_fetched"] >= 0


def test_query_requires_query_field() -> None:
    # Missing required `query` -> 422 from request validation.
    resp = client.post("/query", json={})
    assert resp.status_code == 422
