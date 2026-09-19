# CX Insight Pipeline

Classifies customer support tickets with an LLM, but treats the LLM as an unreliable
component: every response is validated against a schema, the classifier is measured
against a labelled test set before it's trusted, and results are cached so re-runs
don't re-pay for tickets already seen. Output is a DuckDB warehouse plus a three-sheet
Excel report for stakeholders.

The pipeline runs with no API key using a keyword baseline, which also serves as the
number the LLM has to beat.

```
ingest tickets ─> classify only what's new ─> validate ─> persist (DuckDB) ─> report
                        │                                         │
                        └── heuristic  or  Claude                 └── cached by content hash
```

## Design

Putting an LLM in the middle of a data pipeline raises three questions a support team
would actually ask, and each one drove a decision here:

1. **How often is the classifier right?** A labelled gold set (49 tickets) gives
   per-field accuracy and a per-category confusion matrix, so classifier quality is a
   number, not an assumption.
2. **What happens when the model returns garbage?** Every response is validated against
   a pydantic contract. A response that fails twice is quarantined and retried on the
   next run rather than getting a default label that quietly pollutes the data.
3. **Is the LLM worth its cost?** A rule-based classifier runs behind the same interface
   as a baseline. If the model can't clear it by a wide margin, it isn't earning its
   cost or latency — and the eval is how you check.

## Baseline vs model

The keyword baseline is deliberately simple. On the 49-ticket gold set:

| Field | Heuristic baseline | Claude (`claude-haiku-4-5`) |
| --- | --- | --- |
| Category | 55.1% | _run `make eval-claude` to fill in_ |
| Priority | 49.0% | _" "_ |
| Sentiment | 34.7% | _" "_ |

The baseline runs in CI on every push and gates the build: if category accuracy drops
below the floor in `config.py`, CI fails. The Claude column is left for you to fill in
from a live run, because it costs a few cents of API spend and shouldn't run in CI.

## Running it

No API key needed for the default path:

```
make install
make run        # ingest -> classify (heuristic) -> DuckDB -> Excel report
make eval       # baseline accuracy + confusion matrix, writes eval/eval_report.md
make test       # 16 tests, all offline
```

To use the model, set a key and switch the classifier:

```
export ANTHROPIC_API_KEY=sk-ant-...
make run-claude
make eval-claude    # this is the number that matters — model vs baseline
```

Re-running `make run` after adding tickets only classifies the new ones; the rest are
served from the cache. Editing a ticket's text changes its content hash, so that one
ticket is reclassified and the others are not.

## Layout

```
cx_insight/
  schema.py        pydantic contract for a classification
  classifiers.py   heuristic baseline + Claude, behind one classify() call
  store.py         DuckDB: raw tickets, labels, and a per-call log (tokens/cost/latency)
  evaluate.py      metrics, confusion matrix, and the CI accuracy gate
  kpis.py          KPI computation + a no-LLM fallback summary
  report.py        three-sheet Excel export
  pipeline.py      orchestration + CLI
data/
  support_tickets.csv    50 tickets
  labeled_tickets.csv    49 gold labels (category / priority / sentiment)
tests/                   schema, classifiers, cache, evaluation
```

## What's persisted

The DuckDB file holds three tables. `classifications` is the labelled output;
`call_log` records every classify() call with its token counts, estimated cost,
latency, attempts, and whether it passed validation, so cost and failure rate are
queryable rather than lost to stdout:

```sql
select model, count(*), round(sum(cost_usd), 4) as spend,
       round(avg(latency_ms), 1) as avg_ms,
       sum(case when not valid then 1 else 0 end) as quarantined
from call_log group by model;
```

## Notes

- Pricing in `config.py` is an estimate for the cost signal, not billing-grade
  accounting. Update the rates to match current published pricing.
- The gold labels are one analyst's judgement. A handful of tickets are genuinely
  ambiguous (an account-tier request with no clean category is left out of the gold
  set), which is part of why no classifier reaches 100%.
