.PHONY: install run eval eval-claude test clean

install:
	pip install -r requirements.txt

# Full pipeline on the free heuristic classifier (no API key needed).
run:
	python -m cx_insight.pipeline --rebuild

# Classify with Claude instead. Needs ANTHROPIC_API_KEY; costs a few cents.
run-claude:
	python -m cx_insight.pipeline --classifier claude --rebuild

# Baseline evaluation + the CI accuracy gate.
eval:
	python -m cx_insight.evaluate --classifier heuristic

# The real comparison: how much accuracy the model buys over the baseline.
eval-claude:
	python -m cx_insight.evaluate --classifier claude

test:
	pytest -q

clean:
	rm -f cx_insight.duckdb cx_insight.duckdb.wal
	rm -rf output __pycache__ */__pycache__ .pytest_cache eval/eval_report.md
