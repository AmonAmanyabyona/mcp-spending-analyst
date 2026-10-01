"""
MCP server exposing local bank statement analysis.

The data never leaves this process. Tools return computed aggregates --
totals, counts, deltas -- not raw transaction rows, with one deliberate
exception (search_transactions) that is capped and redacted.

That design is doing double duty: aggregates are what keeps the model from
doing arithmetic it will get wrong, and they are also what keeps the raw
statement off the network.

Never print to stdout from this process. On stdio transport stdout IS the
protocol channel, and one stray print corrupts the JSON-RPC stream and
drops the connection. Diagnostics go to stderr.

Run directly for stdio:
    python server.py

Inspect interactively:
    npx @modelcontextprotocol/inspector python server.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
from mcp.server.mcpserver import MCPServer

from analyst.categorise import add_categories, coverage
from analyst.parse import ParseError, load_folder

mcp = MCPServer("spending-analyst")

DATA_DIR = Path(os.environ.get("ANALYST_DATA_DIR", "data"))
FALLBACK_DIR = Path("sample_data")
MAX_SEARCH_RESULTS = 20
MONTH_PATTERN = re.compile(r"^\d{4}-\d{2}$")

# A subscription bills about once a month. A supermarket bills eight times,
# and among that many charges some will land near each other by chance.
# This ratio is what separates the two.
MAX_CHARGES_PER_MONTH = 1.5

# Recurring but not optional -- reported separately so the headline number
# isn't dominated by rent.
ESSENTIAL_CATEGORIES = {"housing", "utilities"}

_cache: dict[str, object] = {}


# --- data loading -----------------------------------------------------


def _source_dir() -> Path:
    """Use real data if present, otherwise the committed sample.

    This is what makes the repo runnable straight after cloning.
    """
    if DATA_DIR.exists() and any(DATA_DIR.glob("*.csv")):
        return DATA_DIR
    return FALLBACK_DIR


def _fingerprint(folder: Path) -> tuple:
    """Cheap change detection: file names plus modification times."""
    return tuple(sorted((f.name, f.stat().st_mtime) for f in folder.glob("*.csv")))


def _load() -> pd.DataFrame:
    """Parsed, categorised transactions. Cached until the files change."""
    folder = _source_dir()
    stamp = _fingerprint(folder)

    if _cache.get("stamp") != stamp:
        frame = add_categories(load_folder(folder))
        _cache["stamp"] = stamp
        _cache["frame"] = frame
        _cache["folder"] = folder

    return _cache["frame"]  # type: ignore[return-value]


def _spending(frame: pd.DataFrame) -> pd.DataFrame:
    """Outgoing transactions only, with amounts as positive numbers."""
    out = frame[frame["amount"] < 0].copy()
    out["amount"] = out["amount"].abs()
    return out


def _check_month(month: str) -> str:
    if not MONTH_PATTERN.match(month):
        raise ValueError(
            f"Month must be formatted as YYYY-MM, got {month!r}. "
            f"Call list_months to see which months have data."
        )
    return month


def _redact(text: str) -> str:
    """Strip identifiers from a payment reference before it leaves here."""
    text = re.sub(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b", "[IBAN]", str(text))
    text = re.sub(r"\b\d{8,}\b", "[REF]", text)
    return text[:60].strip()


# --- tools ------------------------------------------------------------


@mcp.tool()
def list_months() -> dict:
    """List the months that have transaction data, and summarise what is loaded.

    Call this FIRST, before any question that mentions a time period. The
    data covers a fixed historical range that will not usually include the
    current month, so never assume which months exist -- check here, then
    use one of the returned values for the `month` argument of other tools.
    """
    try:
        frame = _load()
    except ParseError as exc:
        raise ValueError(str(exc)) from exc

    months = sorted(frame["month"].unique())
    spending = _spending(frame)
    missed = coverage(frame)

    return {
        "months": months,
        "earliest_date": frame["date"].min().strftime("%Y-%m-%d"),
        "latest_date": frame["date"].max().strftime("%Y-%m-%d"),
        "transaction_count": int(len(frame)),
        "currency": frame["currency"].mode().iat[0] if len(frame) else "EUR",
        "total_spending": round(float(spending["amount"].sum()), 2),
        "total_income": round(float(frame[frame["amount"] > 0]["amount"].sum()), 2),
        "categories": sorted(spending["category"].unique()),
        "uncategorised_merchants": int(len(missed)),
        "source": str(_cache.get("folder", "")),
        "note": (
            "Using bundled sample data, not real statements."
            if _cache.get("folder") == FALLBACK_DIR
            else "Using real statement data."
        ),
    }


@mcp.tool()
def spending_by_category(month: str | None = None) -> dict:
    """Total spending per category, largest first.

    Use this for any question about where money went, what was spent on a
    category, or how much went out in a period. The totals are computed
    here -- report them as given and do not re-add or estimate them.

    Args:
        month: A month as YYYY-MM, e.g. "2026-03". Omit for all months
            combined. Use list_months to see which months have data.
    """
    frame = _spending(_load())

    if month is not None:
        _check_month(month)
        frame = frame[frame["month"] == month]
        if frame.empty:
            raise ValueError(
                f"No transactions in {month}. Call list_months for available months."
            )

    grouped = (
        frame.groupby("category")
        .agg(total=("amount", "sum"), transactions=("amount", "size"))
        .sort_values("total", ascending=False)
        .reset_index()
    )

    total = float(frame["amount"].sum())

    return {
        "month": month or "all months",
        "total_spending": round(total, 2),
        "categories": [
            {
                "category": row.category,
                "total": round(float(row.total), 2),
                "transactions": int(row.transactions),
                "share_percent": round(100 * float(row.total) / total, 1),
            }
            for row in grouped.itertuples()
        ],
    }


@mcp.tool()
def find_recurring_charges(min_months: int = 3, tolerance_percent: float = 2.0) -> dict:
    """Find subscriptions and other recurring charges.

    Detects merchants that bill roughly once a month at a near-identical
    amount -- streaming services, gym memberships, rent, phone contracts.
    Use this for questions about subscriptions, recurring payments, or
    money going out on things the user may have forgotten about.

    Shops visited often are deliberately excluded, even when the amounts
    look similar, because a frequent merchant is not a subscription.

    Args:
        min_months: How many distinct months a charge must appear in to
            count as recurring. Default 3.
        tolerance_percent: How much the amount may vary between months and
            still count as the same charge. Keep this small; a real
            subscription charges the same amount every time. Default 2%.
    """
    frame = _spending(_load())
    total_months = frame["month"].nunique()
    found = []

    for merchant, rows in frame.groupby("merchant"):
        months_present = rows["month"].nunique()
        if months_present < min_months:
            continue

        # Frequency filter: a subscription bills about once per month.
        if len(rows) > months_present * MAX_CHARGES_PER_MONTH:
            continue

        # Find the amount that repeats across the most distinct months,
        # rather than assuming the median is the recurring one.
        best_months, best_cluster = 0, None
        for candidate in rows["amount"].unique():
            margin = max(candidate * tolerance_percent / 100, 0.01)
            cluster = rows[(rows["amount"] - candidate).abs() <= margin]
            months_hit = cluster["month"].nunique()
            if months_hit > best_months:
                best_months, best_cluster = months_hit, cluster

        if best_months < min_months or best_cluster is None:
            continue

        typical = float(best_cluster["amount"].median())
        spread = float(best_cluster["amount"].max() - best_cluster["amount"].min())

        found.append(
            {
                "merchant": merchant,
                "category": best_cluster["category"].mode().iat[0],
                "typical_amount": round(typical, 2),
                "months_charged": best_months,
                "months_in_data": total_months,
                "first_seen": best_cluster["date"].min().strftime("%Y-%m-%d"),
                "last_seen": best_cluster["date"].max().strftime("%Y-%m-%d"),
                "total_paid": round(float(best_cluster["amount"].sum()), 2),
                "every_month": best_months == total_months,
                "amount_varies_by": round(spread, 2),
            }
        )

    found.sort(key=lambda item: item["typical_amount"], reverse=True)

    monthly = sum(item["typical_amount"] for item in found)
    discretionary = sum(
        item["typical_amount"]
        for item in found
        if item["category"] not in ESSENTIAL_CATEGORIES
    )

    return {
        "recurring_charges": found,
        "count": len(found),
        "monthly_total": round(monthly, 2),
        "discretionary_monthly_total": round(discretionary, 2),
        "annual_estimate": round(monthly * 12, 2),
        "months_analysed": total_months,
        "note": (
            "discretionary_monthly_total excludes rent and utilities, "
            "which are recurring but not optional."
        ),
    }


@mcp.tool()
def compare_months(month_a: str, month_b: str) -> dict:
    """Compare spending between two months, by category.

    Use this for any question about change over time -- whether spending
    went up or down, what got more expensive, how one month compares to
    another. The differences are computed here; report them as given rather
    than subtracting the numbers yourself.

    Args:
        month_a: The earlier month, as YYYY-MM.
        month_b: The later month, as YYYY-MM.
    """
    _check_month(month_a)
    _check_month(month_b)

    frame = _spending(_load())
    available = set(frame["month"].unique())

    for month in (month_a, month_b):
        if month not in available:
            raise ValueError(
                f"No data for {month}. Available months: {sorted(available)}"
            )

    a = frame[frame["month"] == month_a].groupby("category")["amount"].sum()
    b = frame[frame["month"] == month_b].groupby("category")["amount"].sum()

    categories = sorted(set(a.index) | set(b.index))
    rows = []

    for category in categories:
        before = float(a.get(category, 0.0))
        after = float(b.get(category, 0.0))
        change = after - before
        rows.append(
            {
                "category": category,
                month_a: round(before, 2),
                month_b: round(after, 2),
                "change": round(change, 2),
                "percent_change": (
                    round(100 * change / before, 1) if before else None
                ),
            }
        )

    rows.sort(key=lambda row: abs(row["change"]), reverse=True)

    total_a = round(float(a.sum()), 2)
    total_b = round(float(b.sum()), 2)

    return {
        "month_a": month_a,
        "month_b": month_b,
        "total_a": total_a,
        "total_b": total_b,
        "total_change": round(total_b - total_a, 2),
        "by_category": rows,
    }


@mcp.tool()
def search_transactions(
    merchant: str | None = None,
    category: str | None = None,
    month: str | None = None,
    min_amount: float | None = None,
    max_results: int = MAX_SEARCH_RESULTS,
) -> dict:
    """Find individual transactions matching some filters.

    Use this only when the question is about specific transactions -- when
    something was last paid, whether a particular shop appears, what the
    largest single purchase was. For totals and summaries use
    spending_by_category or compare_months instead, which are both more
    accurate and return less data.

    Results are capped at 20 rows and payment references are redacted.

    Args:
        merchant: Merchant name, matched loosely. e.g. "spotify", "rewe".
        category: One of the categories returned by list_months.
        month: A month as YYYY-MM.
        min_amount: Only transactions at or above this amount, as a
            positive number.
        max_results: Up to 20.
    """
    frame = _spending(_load())

    if merchant:
        mask = frame["merchant"].str.contains(merchant, case=False, na=False)
        mask |= frame["description"].str.contains(merchant, case=False, na=False)
        frame = frame[mask]

    if category:
        frame = frame[frame["category"].str.lower() == category.lower()]

    if month:
        _check_month(month)
        frame = frame[frame["month"] == month]

    if min_amount is not None:
        frame = frame[frame["amount"] >= abs(min_amount)]

    matched = len(frame)
    total_matched = round(float(frame["amount"].sum()), 2) if matched else 0.0

    limit = max(1, min(int(max_results), MAX_SEARCH_RESULTS))
    frame = frame.sort_values("date", ascending=False).head(limit)

    return {
        "matched": matched,
        "returned": len(frame),
        "truncated": matched > len(frame),
        "total_of_all_matches": total_matched,
        "transactions": [
            {
                "date": row.date.strftime("%Y-%m-%d"),
                "merchant": row.merchant,
                "amount": round(float(row.amount), 2),
                "category": row.category,
                "reference": _redact(row.reference),
            }
            for row in frame.itertuples()
        ],
    }


if __name__ == "__main__":
    mcp.run()