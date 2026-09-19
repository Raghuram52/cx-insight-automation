"""KPI computation and a deterministic fallback narrative.

The KPIs are plain pandas over the classified frame. The narrative here is templated,
not model-written — it's what the report uses when you run the free heuristic path or
in CI. The Claude path swaps in a model-written briefing (see pipeline.py).
"""
from __future__ import annotations

import pandas as pd


def compute_kpis(df: pd.DataFrame) -> dict:
    total = len(df)
    if total == 0:
        return {}
    neg = df["ai_sentiment"].isin(["Negative", "Urgent"]).sum()
    pos = (df["ai_sentiment"] == "Positive").sum()
    ent = (df["customer_segment"] == "Enterprise").sum()
    return {
        "Total Tickets": total,
        "Critical Issues": int((df["ai_priority"] == "Critical").sum()),
        "High Priority Issues": int((df["ai_priority"] == "High").sum()),
        "Negative/Urgent Sentiment %": f"{round(neg / total * 100, 1)}%",
        "Positive Sentiment %": f"{round(pos / total * 100, 1)}%",
        "Enterprise Ticket %": f"{round(ent / total * 100, 1)}%",
        "Top Issue Category": df["ai_category"].value_counts().idxmax(),
        "Top Inbound Channel": df["channel"].value_counts().idxmax(),
    }


def templated_summary(df: pd.DataFrame, kpis: dict) -> str:
    """A no-LLM executive summary. Boring on purpose — it states the numbers plainly."""
    if not kpis:
        return "No classified tickets in this run."
    cats = df["ai_category"].value_counts()
    top_two = ", ".join(cats.head(2).index)
    crit = kpis["Critical Issues"]
    return (
        f"Across {kpis['Total Tickets']} tickets, {top_two} accounted for the largest share of volume. "
        f"{crit} ticket(s) were flagged Critical and {kpis['High Priority Issues']} High. "
        f"Negative or urgent sentiment ran at {kpis['Negative/Urgent Sentiment %']}, "
        f"with {kpis['Enterprise Ticket %']} of volume coming from Enterprise accounts. "
        f"Most inbound arrived via {kpis['Top Inbound Channel']}."
    )
