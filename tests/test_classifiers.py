"""Classifier tests.

The Claude tests use a fake client so they run offline and deterministically — the
point is to exercise the parse / validate / retry / quarantine logic, not the model.
"""
from cx_insight.classifiers import ClaudeClassifier, HeuristicClassifier

TICKET = {
    "ticket_id": "T001",
    "subject": "Cannot access dashboard after update",
    "description": "Our entire team is locked out and it is blocking our reporting.",
    "customer_segment": "Enterprise",
}


class _Resp:
    def __init__(self, text):
        self.content = [type("blk", (), {"text": text})()]
        self.usage = type("u", (), {"input_tokens": 120, "output_tokens": 40})()


class _FakeClient:
    """Returns queued responses in order, so a test can script 'junk then valid'."""
    def __init__(self, texts):
        self._texts = list(texts)
        self.calls = 0
        self.messages = self

    def create(self, **_):
        self.calls += 1
        return _Resp(self._texts.pop(0))


GOOD = '{"category": "Access & Authentication", "priority": "Critical", "sentiment": "Urgent", "summary": "Team locked out of dashboard.", "business_impact": "Reporting blocked."}'


def test_heuristic_labels_access_ticket():
    r = HeuristicClassifier().classify(TICKET)
    assert r.valid
    assert r.classification.category == "Access & Authentication"
    assert r.cost_usd == 0.0


def test_claude_parses_valid_json():
    clf = ClaudeClassifier(client=_FakeClient([GOOD]))
    r = clf.classify(TICKET)
    assert r.valid
    assert r.attempts == 1
    assert r.input_tokens == 120
    assert r.cost_usd > 0


def test_claude_strips_code_fences():
    fenced = "Here is the classification:\n```json\n" + GOOD + "\n```"
    r = ClaudeClassifier(client=_FakeClient([fenced])).classify(TICKET)
    assert r.valid


def test_claude_retries_then_succeeds():
    clf = ClaudeClassifier(client=_FakeClient(["not json at all", GOOD]), max_attempts=2)
    r = clf.classify(TICKET)
    assert r.valid
    assert r.attempts == 2


def test_claude_quarantines_after_all_attempts_fail():
    clf = ClaudeClassifier(client=_FakeClient(["nope", "still nope"]), max_attempts=2)
    r = clf.classify(TICKET)
    assert not r.valid
    assert r.classification is None
    assert r.error  # carries the last parse error


def test_claude_rejects_out_of_taxonomy_category():
    bad = '{"category": "Refunds", "priority": "High", "sentiment": "Negative", "summary": "x"}'
    clf = ClaudeClassifier(client=_FakeClient([bad, bad]), max_attempts=2)
    r = clf.classify(TICKET)
    assert not r.valid  # invented category never passes the contract
