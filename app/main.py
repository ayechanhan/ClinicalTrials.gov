"""
FastAPI application exposing the ClinicalTrials.gov Query-to-Visualization API.

Step 1 scaffold: a single ``POST /query`` endpoint that returns a hardcoded
example ``VisualizationResponse``. The real planner -> fetcher -> assembler
pipeline is wired up in later steps; the response shape here is the contract
those steps will fill with live data.
"""

from fastapi import FastAPI

from app.schemas import (
    Citation,
    DataPoint,
    Encoding,
    QueryRequest,
    ResponseMeta,
    VisualizationResponse,
    VisualizationSpec,
    VizType,
)

app = FastAPI(
    title="ClinicalTrials.gov Query-to-Visualization Agent",
    description=(
        "Turn a natural-language question about clinical trials into a structured "
        "visualization specification backed by real ClinicalTrials.gov data, with "
        "per-data-point citations."
    ),
    version="0.1.0",
)


@app.get("/", tags=["meta"])
def root() -> dict:
    """Service banner with a pointer to the interactive docs."""
    return {
        "service": "ClinicalTrials.gov Query-to-Visualization Agent",
        "version": "0.1.0",
        "docs": "/docs",
        "query_endpoint": "POST /query",
    }


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


def _stub_response() -> VisualizationResponse:
    """Hardcoded example response (Step 1 placeholder).

    Demonstrates the full output contract — including per-data-point citations —
    using the canonical time-trend query. Replaced by the live pipeline in Step 4.
    """
    data = [
        DataPoint(
            year=2015,
            trial_count=2,
            citations=[
                Citation(nct_id="NCT02345678", excerpt="Start Date: 2015-03"),
                Citation(nct_id="NCT02360001", excerpt="Start Date: 2015-09"),
            ],
        ),
        DataPoint(
            year=2016,
            trial_count=5,
            citations=[Citation(nct_id="NCT02712983", excerpt="Start Date: 2016-01")],
        ),
        DataPoint(
            year=2017,
            trial_count=9,
            citations=[Citation(nct_id="NCT03012345", excerpt="Start Date: 2017-06")],
        ),
        DataPoint(
            year=2018,
            trial_count=14,
            citations=[Citation(nct_id="NCT03456789", excerpt="Start Date: 2018-02")],
        ),
    ]
    spec = VisualizationSpec(
        type=VizType.TIME_SERIES,
        title="Pembrolizumab trials started per year (2015-2018)",
        encoding=Encoding(
            x={"field": "year", "type": "temporal"},
            y={"field": "trial_count", "type": "quantitative"},
        ),
        data=data,
    )
    meta = ResponseMeta(
        filters={"drug_name": "Pembrolizumab", "start_year": 2015},
        query_interpretation=(
            "Count Pembrolizumab trials by start year since 2015 and show the trend over time."
        ),
        total_trials_fetched=30,
        notes="STUB RESPONSE — hardcoded example data. The live CT.gov pipeline is wired in Step 4.",
    )
    return VisualizationResponse(visualization=spec, meta=meta)


@app.post("/query", response_model=VisualizationResponse, tags=["query"])
def query(request: QueryRequest) -> VisualizationResponse:
    """Accept a natural-language query and return a visualization specification.

    Step 1: returns a hardcoded example regardless of input. The request body is
    still parsed and validated against ``QueryRequest``, so the input half of the
    contract is exercised end-to-end.
    """
    return _stub_response()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
