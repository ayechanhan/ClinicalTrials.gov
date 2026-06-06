"""Thin async ``httpx`` wrapper around the ClinicalTrials.gov Data API (v2).

Will expose: ``search_studies()``, ``get_field_values()``, ``get_study()``.
Handles pagination with a documented page-size / max-pages cap.

Base URL: https://clinicaltrials.gov/api/v2

Implemented in Step 2.
"""
