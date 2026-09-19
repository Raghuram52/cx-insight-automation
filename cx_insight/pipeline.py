"""End-to-end pipeline and CLI.

    ingest tickets -> classify only what's new -> persist -> aggregate -> export

Default classifier is the free heuristic so this runs with no API key. Point it at
Claude with --classifier claude once you've set ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from . import config, kpis, report, store
from .classifiers import get_classifier


def run(classifier_name: str = "heuristic", write_report: bool = True, rebuild: bool = False) -> dict:
    if rebuild and config.DB_PATH.exists():
        config.DB_PATH.unlink()

    con = store.connect(config.DB_PATH)
    clf = get_classifier(classifier_name)

    tickets = pd.read_csv(config.TICKETS_CSV)
    tickets["date"] = pd.to_datetime(tickets["date"])
    store.land_tickets(con, tickets)

    todo = store.pending(con)
    print(f"{len(tickets)} tickets landed, {len(todo)} need classification "
          f"({len(tickets) - len(todo)} served from cache)")

    for _, row in todo.iterrows():
        result = clf.classify(row.to_dict())
        store.save(con, result, row.to_dict())
        flag = "" if result.valid else "  [quarantined]"
        label = result.classification.priority if result.valid else "—"
        print(f"  {row['ticket_id']}: {label}{flag}")

    df = store.classified_frame(con)
    metrics = kpis.compute_kpis(df)

    if classifier_name == "claude":
        summary = _llm_summary(df, metrics)
    else:
        summary = kpis.templated_summary(df, metrics)

    if write_report and not df.empty:
        report.write_report(df, metrics, summary, config.REPORT_XLSX)
        print(f"report → {config.REPORT_XLSX}")

    stats = store.run_stats(con)
    con.close()
    print("run:", json.dumps(stats))
    return stats


def _llm_summary(df: pd.DataFrame, kpis_: dict) -> str:
    """A VP-facing briefing written by the model. Only reached on the claude path."""
    import anthropic

    prompt = (
        "You are a senior CX analyst writing a weekly briefing for the VP of Customer Success. "
        "Write 4-5 sentences, paragraph form, no bullet points, on the state of support this period.\n\n"
        f"Tickets: {len(df)}\n"
        f"Category counts: {json.dumps(df['ai_category'].value_counts().to_dict())}\n"
        f"Priority counts: {json.dumps(df['ai_priority'].value_counts().to_dict())}\n"
        f"Critical summaries: {df[df['ai_priority'] == 'Critical']['ai_summary'].tolist()}\n"
    )
    resp = anthropic.Anthropic().messages.create(
        model=config.DEFAULT_MODEL, max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the CX insight pipeline.")
    ap.add_argument("--classifier", default="heuristic", choices=["heuristic", "claude"])
    ap.add_argument("--rebuild", action="store_true", help="drop the warehouse and reclassify everything")
    ap.add_argument("--no-report", action="store_true", help="skip the Excel export")
    args = ap.parse_args(argv)
    run(args.classifier, write_report=not args.no_report, rebuild=args.rebuild)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
