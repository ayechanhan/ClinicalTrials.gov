"""Deterministic post-processing: raw studies -> normalized, sorted, paginated
``DataPoint`` rows per viz type. All grouping, counting and sorting happens here
in Python (LLMs are bad at counting), and citations are attached per bucket.

Adding a new viz type means adding a branch here plus an entry in the
intent->viz mapping; the planner prompt does not need to change.

Implemented in Step 4.
"""
