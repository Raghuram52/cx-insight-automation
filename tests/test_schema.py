import pytest
from pydantic import ValidationError

from cx_insight.schema import Classification


def test_valid_classification():
    c = Classification(
        category="Billing & Payments", priority="High", sentiment="Negative",
        summary="Customer charged twice for one period.",
    )
    assert c.category == "Billing & Payments"
    assert c.business_impact == ""  # optional


def test_rejects_unknown_category():
    with pytest.raises(ValidationError):
        Classification(category="Refunds", priority="High", sentiment="Negative", summary="x")


def test_rejects_unknown_priority():
    with pytest.raises(ValidationError):
        Classification(category="Billing & Payments", priority="Blocker", sentiment="Negative", summary="x")


def test_rejects_empty_summary():
    with pytest.raises(ValidationError):
        Classification(category="Billing & Payments", priority="High", sentiment="Negative", summary="")
