"""Classifiers.

Two implementations sit behind one small interface:

  HeuristicClassifier  - keyword rules. Free, instant, deterministic. It exists to
                         be a baseline: if the LLM can't beat this, the LLM isn't
                         worth its cost or latency. It also lets the whole pipeline,
                         the tests, and CI run with no API key.

  ClaudeClassifier     - the real thing. Calls the model, validates the response
                         against the schema, retries on malformed output, and
                         reports token/cost/latency for every call.

Everything downstream (pipeline, store, eval) depends only on `classify()` returning
a Result, so swapping or comparing the two is a one-line change.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Optional

from pydantic import ValidationError

from .config import CATEGORIES, DEFAULT_MODEL, PRICE_PER_MTOK
from .schema import Classification


@dataclass
class Result:
    """A classification plus everything we want to observe about the call."""
    ticket_id: str
    classification: Optional[Classification]  # None == the model never gave us valid output
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    attempts: int = 1
    error: str = ""

    @property
    def valid(self) -> bool:
        return self.classification is not None

    @property
    def cost_usd(self) -> float:
        rate = PRICE_PER_MTOK.get(self.model)
        if not rate:
            return 0.0
        return (self.input_tokens / 1e6) * rate["input"] + (self.output_tokens / 1e6) * rate["output"]


# ── heuristic baseline ────────────────────────────────────────────────────────

# Ordered most-specific first; first hit wins. Deliberately simple — this is a
# floor to beat, not a model.
_CATEGORY_RULES = [
    ("Security & Compliance", ["breach", "unauthorized", "compliance", "gdpr", "soc2", "security", "vulnerab"]),
    ("Billing & Payments",    ["invoice", "charged", "refund", "billing", "payment", "subscription", "overcharge"]),
    ("Integration & API",     ["api", "integration", "webhook", "salesforce", "connector", "sync", "oauth"]),
    ("Access & Authentication", ["locked out", "cannot access", "can't log", "login", "password", "sso", "2fa", "mfa"]),
    ("Data & Reporting",      ["export", "report", "dashboard", "csv", "data", "analytics", "chart"]),
    ("Performance & Stability", ["slow", "timeout", "crash", "outage", "down", "latency", "error 5", "unresponsive"]),
    ("User Management",       ["add user", "add a user", "team member", "seat", "invite", "permission", "role"]),
    ("Feature Request",       ["would love", "feature request", "can you add", "it would be great", "wish"]),
    ("Positive Feedback",     ["thank you", "thanks", "great work", "love the", "fantastic", "amazing", "really happy"]),
    ("Product Education",     ["how do i", "how to", "where do i", "not sure how", "documentation", "guide"]),
]

_URGENT_HINTS = ["urgent", "asap", "immediately", "breach", "outage", "locked out", "data loss", "down"]
_NEGATIVE_HINTS = ["frustrat", "unacceptable", "disappoint", "angry", "still not", "again", "broken"]
_POSITIVE_HINTS = ["thank", "great", "love", "fantastic", "amazing", "happy", "appreciate"]


class HeuristicClassifier:
    name = "heuristic"

    def __init__(self, model: str = "heuristic"):
        self.model = model

    def classify(self, ticket: dict) -> Result:
        t0 = time.perf_counter()
        text = f"{ticket.get('subject','')} {ticket.get('description','')}".lower()

        category = "Product Education"  # weakest-signal fallback; most "how do I" tickets land here
        for cat, kws in _CATEGORY_RULES:
            if any(k in text for k in kws):
                category = cat
                break

        priority = self._priority(text, ticket, category)
        sentiment = self._sentiment(text, category)
        summary = (ticket.get("subject") or "").strip()[:200] or "Support ticket"

        c = Classification(
            category=category,
            priority=priority,
            sentiment=sentiment,
            summary=summary,
            business_impact="",
        )
        return Result(
            ticket_id=ticket["ticket_id"],
            classification=c,
            model=self.model,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    @staticmethod
    def _priority(text: str, ticket: dict, category: str) -> str:
        enterprise = ticket.get("customer_segment") == "Enterprise"
        if any(h in text for h in ["breach", "data loss", "outage", "locked out"]):
            return "Critical" if enterprise else "High"
        if category in ("Security & Compliance",):
            return "Critical" if enterprise else "High"
        if any(h in text for h in ["broken", "failing", "not working", "error", "double"]):
            return "High" if enterprise else "Medium"
        if category in ("Positive Feedback", "Feature Request", "Product Education"):
            return "Low"
        return "Medium"

    @staticmethod
    def _sentiment(text: str, category: str) -> str:
        if any(h in text for h in _URGENT_HINTS):
            return "Urgent"
        if category == "Positive Feedback" or any(h in text for h in _POSITIVE_HINTS):
            return "Positive"
        if any(h in text for h in _NEGATIVE_HINTS):
            return "Negative"
        return "Neutral"


# ── Claude ──────────────────────────────────────────────────────────────────

_PROMPT = """You are a customer experience analyst. Classify this support ticket and return a JSON object only.

Ticket Subject: {subject}
Ticket Description: {description}
Customer Segment: {customer_segment}

Return ONLY a valid JSON object with these exact keys:
{{
  "category": one of {categories},
  "priority": one of ["Critical", "High", "Medium", "Low"],
  "sentiment": one of ["Positive", "Neutral", "Negative", "Urgent"],
  "summary": one sentence on the core issue, max 20 words,
  "business_impact": one sentence on the risk if unresolved, max 15 words
}}

Priority guidance:
- Critical: security breach, data loss, or full outage for an enterprise customer
- High: broken feature affecting many users, billing error, compliance issue
- Medium: single-user issue, non-urgent bug, configuration problem
- Low: feature request, positive feedback, general how-to question"""


def _extract_json(text: str) -> dict:
    """Pull the JSON object out of a model response.

    Models wrap JSON in ```json fences, add a sentence before it, etc. We grab the
    outermost {...} and parse that rather than trusting the whole string.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = re.sub(r"^json", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in response")
    return json.loads(text[start : end + 1])


class ClaudeClassifier:
    name = "claude"

    def __init__(self, model: str = DEFAULT_MODEL, max_attempts: int = 2, client=None):
        self.model = model
        self.max_attempts = max_attempts
        # Injected in tests; created lazily otherwise so importing this module never
        # requires an API key.
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def classify(self, ticket: dict) -> Result:
        prompt = _PROMPT.format(
            subject=ticket.get("subject", ""),
            description=ticket.get("description", ""),
            customer_segment=ticket.get("customer_segment", "Unknown"),
            categories=json.dumps(CATEGORIES),
        )

        t0 = time.perf_counter()
        in_tok = out_tok = 0
        last_err = ""

        for attempt in range(1, self.max_attempts + 1):
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            usage = getattr(resp, "usage", None)
            if usage:
                in_tok += usage.input_tokens
                out_tok += usage.output_tokens

            raw = resp.content[0].text
            try:
                data = _extract_json(raw)
                c = Classification(**data)
                return Result(
                    ticket_id=ticket["ticket_id"], classification=c, model=self.model,
                    input_tokens=in_tok, output_tokens=out_tok,
                    latency_ms=(time.perf_counter() - t0) * 1000, attempts=attempt,
                )
            except (ValueError, ValidationError, json.JSONDecodeError) as e:
                last_err = str(e)
                # tighten the ask and try once more
                prompt += "\n\nYour previous reply was not valid. Return the JSON object only, nothing else."

        # every attempt failed — quarantine rather than invent a label
        return Result(
            ticket_id=ticket["ticket_id"], classification=None, model=self.model,
            input_tokens=in_tok, output_tokens=out_tok,
            latency_ms=(time.perf_counter() - t0) * 1000,
            attempts=self.max_attempts, error=last_err,
        )


def get_classifier(name: str):
    if name == "claude":
        return ClaudeClassifier()
    if name == "heuristic":
        return HeuristicClassifier()
    raise ValueError(f"unknown classifier {name!r} (use 'heuristic' or 'claude')")
