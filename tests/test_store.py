import pandas as pd

from cx_insight import store
from cx_insight.classifiers import HeuristicClassifier

TICKETS = pd.DataFrame([
    {"ticket_id": "T1", "date": "2024-01-01", "customer_segment": "SMB",
     "channel": "Email", "subject": "Billing charged twice", "description": "duplicate invoice charge"},
    {"ticket_id": "T2", "date": "2024-01-02", "customer_segment": "Enterprise",
     "channel": "Chat", "subject": "Cannot access dashboard", "description": "locked out of login"},
])


def _con():
    return store.connect(":memory:")


def test_land_and_pending():
    con = _con()
    store.land_tickets(con, TICKETS)
    assert len(store.pending(con)) == 2  # nothing classified yet


def test_cache_skips_already_classified():
    con = _con()
    store.land_tickets(con, TICKETS)
    clf = HeuristicClassifier()
    for _, row in store.pending(con).iterrows():
        store.save(con, clf.classify(row.to_dict()), row.to_dict())
    # second pass: everything is cached, nothing pending
    assert store.pending(con).empty


def test_changed_content_reclassifies():
    con = _con()
    store.land_tickets(con, TICKETS)
    clf = HeuristicClassifier()
    for _, row in store.pending(con).iterrows():
        store.save(con, clf.classify(row.to_dict()), row.to_dict())

    changed = TICKETS.copy()
    changed.loc[changed.ticket_id == "T1", "description"] = "totally different problem now"
    store.land_tickets(con, changed)
    pend = store.pending(con)
    assert list(pend.ticket_id) == ["T1"]  # only the edited ticket comes back


def test_quarantined_result_is_not_stored_as_label():
    con = _con()
    store.land_tickets(con, TICKETS.head(1))
    row = store.pending(con).iloc[0].to_dict()
    r = HeuristicClassifier().classify(row)
    r.classification = None  # simulate a quarantine
    r.error = "unparseable"
    store.save(con, r, row)
    assert store.pending(con).shape[0] == 1              # still pending, will retry
    assert store.run_stats(con)["quarantined"] == 1      # but the failed call was logged
