import pandas as pd

from cx_insight import config
from cx_insight.classifiers import HeuristicClassifier
from cx_insight.evaluate import evaluate


def test_evaluate_on_labeled_set():
    tickets = pd.read_csv(config.TICKETS_CSV)
    labels = pd.read_csv(config.LABELED_CSV)
    m = evaluate(HeuristicClassifier(), tickets, labels)

    assert m.n == len(labels)                 # every labelled ticket got graded
    assert 0.0 <= m.field_accuracy["category"] <= 1.0
    # confusion matrix totals must equal the number graded
    assert sum(sum(row.values()) for row in m.confusion.values()) == m.n


def test_per_category_only_covers_present_labels():
    tickets = pd.read_csv(config.TICKETS_CSV)
    labels = pd.read_csv(config.LABELED_CSV)
    m = evaluate(HeuristicClassifier(), tickets, labels)
    for cat, stats in m.per_category.items():
        assert stats["support"] > 0
        assert 0.0 <= stats["precision"] <= 1.0
        assert 0.0 <= stats["recall"] <= 1.0
