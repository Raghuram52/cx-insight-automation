"""Evaluation harness.

Runs a classifier over a hand-labelled ticket set and measures how well it agrees
with the gold labels: overall accuracy on each field, per-category precision/recall/F1,
and a confusion matrix for the category field (where the interesting mistakes live).

Run it two ways:
  python -m cx_insight.evaluate --classifier heuristic   # free, deterministic, the baseline
  python -m cx_insight.evaluate --classifier claude       # the model, costs a few cents

The exit code is the CI gate: category accuracy below config.ACCURACY_FLOOR fails the
build. That turns "did my prompt change quietly make classification worse" into a red X
instead of a silent regression.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

from . import config
from .classifiers import get_classifier


@dataclass
class Metrics:
    field_accuracy: dict          # field -> accuracy
    per_category: dict            # category -> {precision, recall, f1, support}
    confusion: dict               # gold -> {predicted -> count}
    n: int
    quarantined: int = 0
    predictions: list = field(default_factory=list)  # (ticket_id, gold, pred) for category


def _prf(confusion: dict, labels) -> dict:
    """Precision / recall / F1 / support per label, from the category confusion matrix."""
    out = {}
    for lab in labels:
        tp = confusion.get(lab, {}).get(lab, 0)
        fp = sum(confusion.get(g, {}).get(lab, 0) for g in labels if g != lab)
        support = sum(confusion.get(lab, {}).values())
        fn = support - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        if support:  # don't report labels that never appear in the gold set
            out[lab] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return out


def evaluate(classifier, tickets: pd.DataFrame, labels: pd.DataFrame) -> Metrics:
    gold = labels.set_index("ticket_id")
    hits = defaultdict(int)
    graded = 0
    quarantined = 0
    confusion: dict = defaultdict(lambda: defaultdict(int))
    preds = []

    for _, ticket in tickets.iterrows():
        tid = ticket["ticket_id"]
        if tid not in gold.index:
            continue
        result = classifier.classify(ticket.to_dict())
        if not result.valid:
            quarantined += 1
            continue
        graded += 1
        c = result.classification
        g = gold.loc[tid]
        for fieldname, predicted, actual in [
            ("category", c.category, g["gold_category"]),
            ("priority", c.priority, g["gold_priority"]),
            ("sentiment", c.sentiment, g["gold_sentiment"]),
        ]:
            if predicted == actual:
                hits[fieldname] += 1
        confusion[g["gold_category"]][c.category] += 1
        preds.append((tid, g["gold_category"], c.category))

    accuracy = {f: (hits[f] / graded if graded else 0.0) for f in ("category", "priority", "sentiment")}
    confusion = {k: dict(v) for k, v in confusion.items()}
    return Metrics(
        field_accuracy=accuracy,
        per_category=_prf(confusion, config.CATEGORIES),
        confusion=confusion,
        n=graded,
        quarantined=quarantined,
        predictions=preds,
    )


def format_report(m: Metrics, classifier_name: str) -> str:
    lines = [
        f"# Classification evaluation — `{classifier_name}`",
        "",
        f"Graded **{m.n}** tickets against gold labels"
        + (f" ({m.quarantined} quarantined as unparseable)." if m.quarantined else "."),
        "",
        "## Accuracy by field",
        "",
        "| Field | Accuracy |",
        "| --- | --- |",
    ]
    for f in ("category", "priority", "sentiment"):
        lines.append(f"| {f.capitalize()} | {m.field_accuracy[f]:.1%} |")

    lines += ["", "## Category — precision / recall / F1", "",
              "| Category | Precision | Recall | F1 | Support |", "| --- | --- | --- | --- | --- |"]
    for cat, s in sorted(m.per_category.items(), key=lambda kv: -kv[1]["support"]):
        lines.append(f"| {cat} | {s['precision']:.2f} | {s['recall']:.2f} | {s['f1']:.2f} | {s['support']} |")

    # Confusion matrix, gold rows × predicted columns, only over categories that appear.
    present = [c for c in config.CATEGORIES if c in m.confusion or any(c in row for row in m.confusion.values())]
    short = {c: c.split(" & ")[0][:6] for c in present}  # keep the header narrow
    lines += ["", "## Category confusion (gold ↓ / predicted →)", "",
              "| gold \\ pred | " + " | ".join(short[c] for c in present) + " |",
              "| --- |" + " --- |" * len(present)]
    for g in present:
        row = m.confusion.get(g, {})
        cells = " | ".join(str(row.get(p, 0)) if row.get(p, 0) else "·" for p in present)
        lines.append(f"| **{short[g]}** | {cells} |")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate a ticket classifier against gold labels.")
    ap.add_argument("--classifier", default="heuristic", choices=["heuristic", "claude"])
    ap.add_argument("--floor", type=float, default=config.ACCURACY_FLOOR,
                    help="fail if category accuracy is below this")
    args = ap.parse_args(argv)

    tickets = pd.read_csv(config.TICKETS_CSV)
    labels = pd.read_csv(config.LABELED_CSV)
    clf = get_classifier(args.classifier)

    m = evaluate(clf, tickets, labels)
    report = format_report(m, args.classifier)
    config.EVAL_REPORT.parent.mkdir(parents=True, exist_ok=True)
    config.EVAL_REPORT.write_text(report)

    print(report)
    cat_acc = m.field_accuracy["category"]
    print(f"\ncategory accuracy {cat_acc:.1%} (floor {args.floor:.0%})")
    if cat_acc < args.floor:
        print("FAIL: below accuracy floor", file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
