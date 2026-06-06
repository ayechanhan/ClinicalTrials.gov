"""Assembler — turns aggregated CT.gov data into a final ``VisualizationSpec``
with per-data-point citations.

Aggregation/counting is done deterministically in ``viz.spec_builder``; the
assembler composes the typed response. It never invents trials or values.

Implemented in Step 4.
"""
