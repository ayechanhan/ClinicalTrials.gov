# ClinicalTrials.gov Query-to-Visualization Agent

A backend service that turns a **natural-language question** about clinical trials into a
**structured visualization specification** backed by real
[ClinicalTrials.gov](https://clinicaltrials.gov/data-api/api) data — with a
**citation back to source trials on every single data point**.

```
POST /query  {"query": "How has the number of trials for Pembrolizumab changed per year since 2015?"}
   → { "visualization": { "type": "time_series", "encoding": {...},
                          "data": [ {"year": 2015, "trial_count": 120, "citations": [...]}, ... ] },
       "meta": { "total_trials_fetched": 2804, "query_interpretation": "...", "filters": {...} } }
```

The output is a renderer-agnostic *spec* (chart `type` + field `encoding` + `data` rows), not
pre-rendered pixels, so any frontend can draw it without guessing.

---

## How it works

```
QueryRequest
   │
   ▼
[1] PLANNER  (LLM, structured output)        app/agent/planner.py
   │   query text → intent_class, viz_type, extracted_params, api_strategy
   │   • only sees the query; never sees data; never produces numbers
   ▼
[2] FETCHER  (deterministic)                  app/agent/tools.py
   │   plan → ClinicalTrials.gov queries; returns EXACT counts + citation samples
   ▼
[3] SPEC BUILDER (deterministic)             app/viz/spec_builder.py
   │   buckets → typed DataPoints + Encoding, with deep citations
   ▼
[4] ASSEMBLER (deterministic)                app/agent/assembler.py
   │   → VisualizationResponse (spec + provenance meta)
   ▼
VisualizationResponse
```

**Only step 1 uses the LLM**, and only to *interpret* the question. Every number comes from
the real API and is aggregated in Python. Nothing in the data path is model-generated — so
there is nothing for the model to hallucinate.

---

## Setup

Requires **Python 3.10+** (developed and tested on 3.12).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then edit .env and add your OpenAI key
```

`.env`:

```
OPENAI_API_KEY=sk-...       # required (the ClinicalTrials.gov API itself needs no key)
OPENAI_MODEL=gpt-4.1        # any model that supports OpenAI Structured Outputs
CT_API_BASE=https://clinicaltrials.gov/api/v2
CT_PAGE_SIZE=100            # studies per page (API max 1000)
CT_MAX_PAGES=10             # pagination safety cap (ceiling 50)
LOG_LEVEL=INFO
```

> **Model note:** the spec suggested `gpt-4o`, but the provided key's project only grants
> access to `gpt-4.1`, `gpt-4.1-mini`, and `gpt-4o-mini`. The default is `gpt-4.1`; any of the
> three works (all support Structured Outputs). Change it in `.env`.

---

## Running

```bash
uvicorn app.main:app --reload          # http://127.0.0.1:8000
```

Open **http://127.0.0.1:8000/docs** for interactive Swagger docs, or:

```bash
curl -s http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "How are lung cancer trials distributed across phases?"}' | python -m json.tool
```

Five ready-made request/response pairs live in [`examples/`](examples/) (one per query class).

### Demo UI

With the server running, open **http://127.0.0.1:8000/demo** for a single-page UI that calls
`POST /query` and renders every chart type (Chart.js for line/bar/grouped, vis-network for the
graph) with clickable citations — a quick way to see the spec rendered without building a frontend.

---

## API

### `POST /query`

**Request** — `query` is required; the optional fields let a caller pin an entity instead of
relying on extraction (they take precedence over the planner).

| field | type | notes |
|---|---|---|
| `query` | string (required) | the natural-language question |
| `drug_name`, `condition`, `sponsor`, `country` | string? | filter overrides |
| `trial_phase` | string? | `PHASE1`–`PHASE4`, `EARLY_PHASE1`, `NA` (accepts `"phase 3"` etc.) |
| `start_year`, `end_year` | int? | `1900 ≤ year ≤ current+5`, `start ≤ end` |

**Response** — `VisualizationResponse`:

```jsonc
{
  "visualization": {
    "type": "time_series | bar_chart | grouped_bar | network_graph | histogram | scatter",
    "title": "Pembrolizumab trials by start year (2015-2026)",
    "encoding": {                       // which data key maps to which visual channel
      "x": {"field": "year", "type": "temporal"},
      "y": {"field": "trial_count", "type": "quantitative"},
      "series": null,                   // grouped_bar: the splitting key
      "nodes": null, "edges": null      // network_graph: data keys holding nodes/edges
    },
    "data": [
      { "year": 2015, "trial_count": 120,
        "citations": [ {"nct_id": "NCT02406781", "excerpt": "…Advanced Sarcomas — Start date: 2015-06"} ] }
    ]
  },
  "meta": {
    "filters": {"drug_name": "Pembrolizumab", "start_year": 2015},
    "source": "clinicaltrials.gov",
    "query_interpretation": "Interpreted as a time trend query, about Pembrolizumab, grouped by year.",
    "total_trials_fetched": 2804,
    "notes": "Counts are exact per bucket …"
  }
}
```

`DataPoint` keys vary by viz type (e.g. `year`/`phase`/`country` + `trial_count`), but **every
data point always carries a typed `citations: [{nct_id, excerpt}]` list**.

### Errors

| status | when |
|---|---|
| **422** | invalid input (empty query, bad year range, unknown phase) — with a clear message; or an unsupported fetch strategy |
| **500** | planner/LLM failure (after one automatic retry) |
| **502** | ClinicalTrials.gov upstream error (surfaced, never swallowed) |

---

## Query & visualization coverage

| Intent | Viz type | Example query |
|---|---|---|
| time_trend | `time_series` | *How has the number of trials for Pembrolizumab changed per year since 2015?* |
| distribution | `bar_chart` | *How are lung cancer trials distributed across phases?* |
| comparison | `grouped_bar` | *Compare phases for trials involving Pembrolizumab vs Nivolumab.* |
| geographic | `bar_chart` (by country) | *Which countries have the most recruiting trials for diabetes?* |
| network | `network_graph` | *Show a network of sponsors and drugs for breast cancer trials.* |

The planner picks the viz type; an intent→viz **guard** coerces any mismatch back to a sensible
default. `histogram`/`scatter` are defined in the contract and slot in the same way.

---

## Design decisions

- **One LLM call, not two.** Only the planner uses the model — to interpret the question into a
  typed plan. The assembler is fully deterministic, so even *it* cannot invent a trial or a
  count. (The brief floated an LLM assembler; making it deterministic is a strictly stronger
  anti-hallucination guarantee.)

- **Structured Outputs for the planner.** The plan is parsed straight into a Pydantic model with
  enum-constrained fields, so there is no free-form JSON to misparse.

- **Exact counts via count-per-bucket — not capped-sample grouping.** Fetching ~1000 studies and
  grouping them client-side would *skew* a large query (Pembrolizumab has ~2,870 trials). Instead,
  for bounded dimensions (years, phases, status) we issue one `countTotal` query **per bucket**
  for an exact count, plus a 5-study sample per bucket for citations. Queries run concurrently
  (capped). Every count in the examples matches the live API exactly.

- **Geographic = sample-then-exact-count.** Countries are high-cardinality, so we discover
  candidate countries from a study sample, then take **exact** `AREA[LocationCountry]` counts for
  the top candidates.

- **Deterministic aggregation.** All grouping/counting/sorting happens in Python — LLMs are bad at
  arithmetic.

- **Deep citations, per data point.** Each `DataPoint` carries up to 5 citations, and each excerpt
  appends the **exact API field value that placed the study in that bucket** (start date, phase,
  country…), read verbatim — so a reader can trace *why* each study backs each point. Network
  citations attach to the **edge** (the studies that created that sponsor↔drug link).

- **The stats endpoint is global-only.** `GET /stats/field/values` rejects query/filter params and
  returns no NCT IDs, so it can't support scoped distributions *or* citations — which is exactly
  why the pipeline aggregates fetched studies instead.

- **Extensibility.** A new viz type = one fetch strategy + one `spec_builder` shaper + one dispatch
  branch; the planner prompt doesn't change.

---

## Limitations

- **Drug-name normalization.** Network drug nodes come from CT.gov's free-text intervention field,
  so a few are messy (e.g. `"AZD5363 when combined with weekly…"`). Names are used as-is rather
  than fuzzily merged; relationships are still accurate and traceable.
- **Network is sampled.** The graph is built from a 200-study sample of the matches and shows the
  40 strongest links (both documented in `meta.notes`).
- **Overlapping dimensions.** A trial can have multiple phases / run in multiple countries, so
  per-bucket counts need not sum to the total — flagged in `meta.notes`.
- **Condition matching** uses CT.gov's text relevance, which can include loosely related trials.

---

## Testing

```bash
python -m pytest -q          # 14 tests, fully offline (no network / no API key needed)
```

The suite pins the deterministic core — assembler/spec_builder output for every viz type
(including deep citations), input validation, and error→status mapping — by feeding synthetic
data and monkeypatching the pipeline. End-to-end runs against the live APIs are captured in
[`examples/`](examples/).

---

## Project structure

```
app/
├── main.py              FastAPI app + POST /query (plan → fetch → assemble)
├── schemas.py           Pydantic I/O contract + input validation
├── config.py            .env-backed settings
├── agent/
│   ├── planner.py       LLM call: query → structured plan (Structured Outputs)
│   ├── tools.py         deterministic fetcher (count-per-bucket, geographic, network)
│   └── assembler.py     deterministic: plan + data → VisualizationResponse
├── ct_client/client.py  async httpx wrapper for the CT.gov v2 API
└── viz/spec_builder.py  deterministic bucket → DataPoint/Encoding shaping + citations
examples/                5 saved request/response pairs (one per query class)
tests/test_agent.py      offline test suite
```
