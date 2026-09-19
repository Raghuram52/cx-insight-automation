"""Central config: paths, the label vocabularies, and the model pricing table.

Kept in one place so the label sets used for prompting, validation, and evaluation
can never drift apart.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TICKETS_CSV = ROOT / "data" / "support_tickets.csv"
LABELED_CSV = ROOT / "data" / "labeled_tickets.csv"
DB_PATH = ROOT / "cx_insight.duckdb"
REPORT_XLSX = ROOT / "output" / "cx_insight_report.xlsx"
EVAL_REPORT = ROOT / "eval" / "eval_report.md"

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

CATEGORIES = [
    "Access & Authentication",
    "Billing & Payments",
    "Data & Reporting",
    "Integration & API",
    "Performance & Stability",
    "Feature Request",
    "Positive Feedback",
    "Security & Compliance",
    "User Management",
    "Product Education",
]

PRIORITIES = ["Critical", "High", "Medium", "Low"]
SENTIMENTS = ["Positive", "Neutral", "Negative", "Urgent"]

# Per-million-token rates used to estimate spend. These are config, not gospel —
# update them to match current published pricing for whatever model you point at.
# (Estimates only; the point is a cost signal in the call log, not billing-grade
# accounting.)
PRICE_PER_MTOK = {
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
}

# Below this classification accuracy the eval gate fails. Set deliberately low as a
# regression floor, not a target — a green CI means "the classifier didn't fall off
# a cliff", not "the classifier is good". The keyword baseline sits around 0.55, so
# 0.50 gives CI headroom; raise it once you trust a model to clear a higher bar.
ACCURACY_FLOOR = 0.50
