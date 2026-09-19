"""Persistence and the incremental cache.

Three tables:
  tickets_raw     - landed input, one row per ticket, with a content hash
  classifications - one row per (ticket_id, content_hash) we've successfully labelled
  call_log        - one row per classify() call ever made: tokens, cost, latency, outcome

The content hash is what makes re-runs cheap. A ticket is only sent to the model if
we've never seen that exact (id, content) before — so classifying 500 tickets, adding
5, and re-running costs 5 calls, not 505. This is the same idea as an incremental load
in a warehouse, applied to an expensive LLM step instead of a SQL transform.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import duckdb
import pandas as pd

from .classifiers import Result

SCHEMA = """
create table if not exists tickets_raw (
    ticket_id        varchar primary key,
    date             date,
    customer_segment varchar,
    channel          varchar,
    subject          varchar,
    description      varchar,
    content_hash     varchar
);
create table if not exists classifications (
    ticket_id        varchar,
    content_hash     varchar,
    category         varchar,
    priority         varchar,
    sentiment        varchar,
    summary          varchar,
    business_impact  varchar,
    model            varchar,
    classified_at    timestamp,
    primary key (ticket_id, content_hash)
);
create table if not exists call_log (
    ticket_id     varchar,
    model         varchar,
    input_tokens  integer,
    output_tokens integer,
    latency_ms    double,
    attempts      integer,
    cost_usd      double,
    valid         boolean,
    error         varchar,
    logged_at     timestamp
);
"""


def connect(db_path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(db_path))
    con.execute(SCHEMA)
    return con


def content_hash(subject: str, description: str) -> str:
    h = hashlib.sha256()
    h.update((str(subject) + "\x00" + str(description)).encode("utf-8"))
    return h.hexdigest()[:16]


def land_tickets(con, df: pd.DataFrame) -> int:
    """Upsert raw tickets. Returns the row count landed."""
    df = df.copy()
    df["content_hash"] = [content_hash(s, d) for s, d in zip(df["subject"], df["description"])]
    con.register("incoming", df)
    con.execute("delete from tickets_raw where ticket_id in (select ticket_id from incoming)")
    con.execute(
        """insert into tickets_raw
           select ticket_id, cast(date as date), customer_segment, channel,
                  subject, description, content_hash
           from incoming"""
    )
    con.unregister("incoming")
    return len(df)


def pending(con) -> pd.DataFrame:
    """Tickets that have no valid classification for their current content."""
    return con.execute(
        """select t.*
           from tickets_raw t
           left join classifications c
             on t.ticket_id = c.ticket_id and t.content_hash = c.content_hash
           where c.ticket_id is null"""
    ).df()


def save(con, result: Result, ticket: dict) -> None:
    """Record one classification outcome: always log the call, store the label if valid."""
    now = datetime.now(timezone.utc)
    con.execute(
        "insert into call_log values (?,?,?,?,?,?,?,?,?,?)",
        [result.ticket_id, result.model, result.input_tokens, result.output_tokens,
         result.latency_ms, result.attempts, round(result.cost_usd, 6),
         result.valid, result.error, now],
    )
    if not result.valid:
        return  # quarantined — no row in classifications, so a later run will retry it
    c = result.classification
    ch = content_hash(ticket.get("subject", ""), ticket.get("description", ""))
    con.execute("delete from classifications where ticket_id = ? and content_hash = ?", [result.ticket_id, ch])
    con.execute(
        "insert into classifications values (?,?,?,?,?,?,?,?,?)",
        [result.ticket_id, ch, c.category, c.priority, c.sentiment,
         c.summary, c.business_impact, result.model, datetime.now(timezone.utc)],
    )


def classified_frame(con) -> pd.DataFrame:
    """Tickets joined to their labels — the input to KPIs and the report."""
    return con.execute(
        """select t.ticket_id, t.date, t.customer_segment, t.channel, t.subject, t.description,
                  c.category   as ai_category,
                  c.priority   as ai_priority,
                  c.sentiment  as ai_sentiment,
                  c.summary    as ai_summary,
                  c.business_impact as ai_business_impact
           from tickets_raw t
           join classifications c
             on t.ticket_id = c.ticket_id and t.content_hash = c.content_hash
           order by t.date, t.ticket_id"""
    ).df()


def run_stats(con) -> dict:
    """A quick read on the most recent activity, for the run summary line."""
    total = con.execute("select count(*) from tickets_raw").fetchone()[0]
    labelled = con.execute("select count(*) from classifications").fetchone()[0]
    spend, calls, quarantined = con.execute(
        "select coalesce(sum(cost_usd),0), count(*), coalesce(sum(case when not valid then 1 else 0 end),0) from call_log"
    ).fetchone()
    return {
        "tickets": total,
        "labelled": labelled,
        "calls_made": calls,
        "quarantined": quarantined,
        "total_cost_usd": round(spend, 4),
    }
