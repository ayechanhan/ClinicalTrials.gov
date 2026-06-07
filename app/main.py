"""FastAPI application: natural-language query -> visualization specification.

Pipeline (POST /query)::

    QueryRequest
      -> planner.plan_query    (LLM, structured output: intent / viz / params / strategy)
      -> tools.fetch_for_plan  (deterministic CT.gov count-per-bucket fetch)
      -> assembler.assemble    (deterministic: buckets -> typed VisualizationResponse)

The planner only sees the query text; all numbers come from the real API and are
aggregated deterministically — nothing in the data path is model-generated.
"""

from fastapi import FastAPI, HTTPException

from app.agent.assembler import assemble
from app.agent.planner import PlannerError, plan_query
from app.agent.tools import FetchError, fetch_for_plan
from app.ct_client.client import ClinicalTrialsClient, CTClientError
from app.schemas import QueryRequest, VisualizationResponse

app = FastAPI(
    title="ClinicalTrials.gov Query-to-Visualization Agent",
    description=(
        "Turn a natural-language question about clinical trials into a structured "
        "visualization specification backed by real ClinicalTrials.gov data, with "
        "per-data-point citations."
    ),
    version="0.2.0",
)


@app.get("/", tags=["meta"])
def root() -> dict:
    """Service banner with a pointer to the interactive docs."""
    return {
        "service": "ClinicalTrials.gov Query-to-Visualization Agent",
        "version": "0.2.0",
        "docs": "/docs",
        "query_endpoint": "POST /query",
    }


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/query", response_model=VisualizationResponse, tags=["query"])
async def query(request: QueryRequest) -> VisualizationResponse:
    """Plan -> fetch -> assemble a visualization spec for a natural-language query.

    Error handling here is intentionally light; the hardening step adds retries,
    input validation, and richer error bodies.
    """
    try:
        plan = await plan_query(request)
    except PlannerError as exc:
        # Planner already retried once; a persistent failure is a server-side error.
        raise HTTPException(status_code=500, detail=f"Query planning failed: {exc}") from exc

    try:
        async with ClinicalTrialsClient() as ct:
            result = await fetch_for_plan(plan, ct)
    except FetchError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except CTClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return assemble(plan, result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
