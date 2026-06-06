"""Deterministic shaping of fetched buckets into typed ``DataPoint`` rows + an
``Encoding``.

No network, no LLM, no invented numbers — pure transformation of the fetcher's
EXACT counts into visualization data, with per-DataPoint citations. Adding a new
viz type means adding one shaper here and one dispatch branch.
"""

from __future__ import annotations

from app.agent.tools import Bucket, FetchResult
from app.schemas import Citation, DataPoint, Encoding, VizType

MAX_CITATIONS = 5      # cap per data point
EXCERPT_MAX = 220      # max excerpt length
TITLE_MAX = 160        # title portion kept before appending the field detail


def _dimension_detail(study: dict, dimension: str, bucket: Bucket) -> str | None:
    """The exact field value tying this study to its bucket (the 'deep' part).

    Each value is read verbatim from the API response — never paraphrased — so a
    reader can trace exactly why this study landed in this data point.
    """
    ps = study.get("protocolSection", {})
    if dimension == "year":
        date = ps.get("statusModule", {}).get("startDateStruct", {}).get("date")
        return f"Start date: {date}" if date else None
    if dimension == "phase":
        phases = ps.get("designModule", {}).get("phases") or []
        return f"Phase: {', '.join(phases)}" if phases else None
    if dimension == "status":
        status = ps.get("statusModule", {}).get("overallStatus")
        return f"Status: {status}" if status else None
    if dimension == "country":
        # Every study in this bucket matched AREA[LocationCountry]"<bucket label>".
        return f"Country: {bucket.label}"
    return None


def _citation(study: dict, dimension: str, bucket: Bucket) -> Citation | None:
    """Build one deep citation: trial title + the field value justifying the bucket."""
    ident = study.get("protocolSection", {}).get("identificationModule", {})
    nct = ident.get("nctId")
    if not nct:
        return None
    title = (ident.get("briefTitle") or "").strip()
    detail = _dimension_detail(study, dimension, bucket)
    if title and detail:
        excerpt = f"{title[:TITLE_MAX]} — {detail}"
    else:
        excerpt = title or detail or nct
    return Citation(nct_id=nct, excerpt=excerpt[:EXCERPT_MAX])


def _citations(bucket: Bucket, dimension: str) -> list[Citation]:
    """Up to MAX_CITATIONS deep citations for a bucket's data point."""
    out: list[Citation] = []
    for study in bucket.sample_studies[:MAX_CITATIONS]:
        cite = _citation(study, dimension, bucket)
        if cite is not None:
            out.append(cite)
    return out


def _time_series(result: FetchResult) -> tuple[list[DataPoint], Encoding]:
    field = result.group_by  # "year"
    points = [
        DataPoint(**{field: b.key, "trial_count": b.count, "citations": _citations(b, field)})
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
        DataPoint(**{field: b.label, "trial_count": b.count, "citations": _citations(b, field)})
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
                     "citations": _citations(b, field)})
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
