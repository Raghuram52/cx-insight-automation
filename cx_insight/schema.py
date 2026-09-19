"""The contract for a classification.

The LLM is an unreliable component: it will occasionally return prose instead of
JSON, invent a category that isn't in our taxonomy, or drop a field. Rather than
trust it and clean up downstream, every response is parsed into this model. If it
doesn't fit, it doesn't pass — the caller decides whether to retry or quarantine.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from .config import CATEGORIES, PRIORITIES, SENTIMENTS


class Classification(BaseModel):
    category: str
    priority: str
    sentiment: str
    summary: str = Field(min_length=1, max_length=240)
    business_impact: str = Field(default="", max_length=240)

    @field_validator("category")
    @classmethod
    def _known_category(cls, v: str) -> str:
        if v not in CATEGORIES:
            raise ValueError(f"unknown category {v!r}")
        return v

    @field_validator("priority")
    @classmethod
    def _known_priority(cls, v: str) -> str:
        if v not in PRIORITIES:
            raise ValueError(f"unknown priority {v!r}")
        return v

    @field_validator("sentiment")
    @classmethod
    def _known_sentiment(cls, v: str) -> str:
        if v not in SENTIMENTS:
            raise ValueError(f"unknown sentiment {v!r}")
        return v
