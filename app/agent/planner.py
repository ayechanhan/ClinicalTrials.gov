"""Planner — interprets a natural-language query into a structured plan
(intent_class, viz_type, extracted_params, api_strategy) via a single LLM call
with structured output.

The planner never sees API data; it only produces the plan. This is what keeps
data values out of the model's hands (no hallucinated counts).

Implemented in Step 3.
"""
