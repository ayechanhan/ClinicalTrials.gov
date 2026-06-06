"""Deterministic shaping of fetched buckets into typed ``DataPoint`` rows + an
``Encoding``.

No network, no LLM, no invented numbers — pure transformation of the fetcher's
EXACT counts into visualization data, with per-DataPoint citations. Adding a new
viz type means adding one shaper here and one dispatch branch.
"""

from __future__ import annotations

from app.agent.tools import Bucket, FetchResult
from app.schemas import Citation, DataPoint, Encoding, VizType

MAX_CITATIONS = 5      # cap per data point (kept tight in Step 4; refined in Step 6)
EXCERPT_MAX = 200      # max excerpt length


def _citation(study: dict) -> Citation | None:
    """Build a citation from a raw study dict; excerpt is the exact BriefTitle."""
    ident = study.get("protocolSection", {}).get("identificationModule", {})
    nct = ident.get("nctId")
    if not nct:
        return None
    excerpt = (ident.get("briefTitle") or "").strip()[:EXCERPT_MAX] or nct
    return Citation(nct_id=nct, excerpt=excerpt)


def _citations(bucket: Bucket) -> list[Citation]:
    out: list[Citation] = []
    for study in bucket.sample_studies[:MAX_CITATIONS]:
        cite = _citation(study)
        if cite is not None:
            out.append(cite)
    return out


def _time_series(result: FetchResult) -> tuple[list[DataPoint], Encoding]:
    field = result.group_by  # "year"
    points = [
        DataPoint(**{field: b.key, "trial_count": b.count, "citations": _citations(b)})
        for b in sorted(result.buckets, key=lambda b: b.key)
    ]
    encoding = Encoding(
        x={"field": field, "type": "temporal"},
        y={"field": "trial_count", "type": "quantitative"},
    )
    return points, encoding


def _bar_chart(result: FetchResult) -> tuple[list[DataPoint], Encoding]:
    field = result.group_by  # e.g. "phase"
    points = [
        DataPoint(**{field: b.label, "trial_count": b.count, "citations": _citations(b)})
        for b in sorted(result.buckets, key=lambda b: b.count, reverse=True)
    ]
    encoding = Encoding(
        x={"field": field, "type": "nominal"},
        y={"field": "trial_count", "type": "quantitative"},
    )
    return points, encoding


def _grouped_bar(result: FetchResult) -> tuple[list[DataPoint], Encoding]:
    field = result.group_by  # e.g. "phase"
    points = [
        DataPoint(**{field: b.label, "series": b.series, "trial_count": b.count,
                     "citations": _citations(b)})
        for b in result.buckets  # already phase-major, series-minor from the fetcher
    ]
    encoding = Encoding(
        x={"field": field, "type": "nominal"},
        y={"field": "trial_count", "type": "quantitative"},
        series="series",
    )
    return points, encoding


def _network(result: FetchResult) -> tuple[list[DataPoint], Encoding]:
    """Shape a sponsor<->drug graph into a single container DataPoint.

    The whole graph is one data row exposing ``nodes`` and ``edges`` arrays (named
    by the encoding); each edge carries its own citations (the studies that created
    that link). The container's typed ``citations`` hold a deduped representative
    sample so the per-DataPoint citation contract still holds.
    """
    nodes = result.nodes or []
    edges = result.edges or []

    seen: set[str] = set()
    representative: list[Citation] = []
    for edge in edges:
        for cite in edge.get("citations", []):
            if cite["nct_id"] not in seen:
                seen.add(cite["nct_id"])
                representative.append(Citation(nct_id=cite["nct_id"], excerpt=cite["excerpt"]))
            if len(representative) >= MAX_CITATIONS:
                break
        if len(representative) >= MAX_CITATIONS:
            break

    container = DataPoint(
        nodes=nodes, edges=edges,
        node_count=len(nodes), edge_count=len(edges),
        citations=representative,
    )
    encoding = Encoding(nodes="nodes", edges="edges")
    return [container], encoding


_SHAPERS = {
    VizType.TIME_SERIES: _time_series,
    VizType.BAR_CHART: _bar_chart,
    VizType.GROUPED_BAR: _grouped_bar,
    VizType.NETWORK_GRAPH: _network,
}


def build_spec_data(result: FetchResult, viz_type: VizType) -> tuple[list[DataPoint], Encoding]:
    """Shape a FetchResult into (data, encoding) for the given viz type."""
    shaper = _SHAPERS.get(viz_type)
    if shaper is None:
        raise ValueError(f"spec_builder has no shaper for viz_type '{viz_type.value}'.")
    return shaper(result)
