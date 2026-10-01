"""Tests for analyst.parse.

The sample generator is seeded, so these assert on exact values. If
generate_sample.py changes, these need updating -- that is intentional.
"""

from pathlib import Path

import pandas as pd
import pytest

from analyst.parse import ParseError, load_file, normalise_header, parse_amount

SAMPLE = Path("sample_data/transactions.csv")


# --- amount parsing ---------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1.234,50", 1234.50),   # German
        ("-10,99", -10.99),
        ("10,99-", -10.99),      # trailing sign
        ("1,234.50", 1234.50),   # English
        ("10.99", 10.99),
        ("1.234", 1234.0),       # lone dot, 3 digits = thousands
        ("3.200,00", 3200.00),
        ("0,00", 0.0),
        ("1.234.567,89", 1234567.89),
        ("€ 45,20", 45.20),      # currency symbol
        (" 45,20 ", 45.20),      # whitespace
        (12.5, 12.5),            # already numeric
    ],
)
def test_parse_amount(raw, expected):
    assert parse_amount(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["", "   ", "abc", "-"])
def test_parse_amount_rejects_junk(raw):
    with pytest.raises(ValueError):
        parse_amount(raw)


# --- header normalising -----------------------------------------------


def test_normalise_header_strips_accents_and_punctuation():
    assert normalise_header("Auftraggeber/Empfänger") == "auftraggeber empfanger"
    assert normalise_header("Betrag (EUR)") == "betrag eur"
    assert normalise_header("  Währung ") == "wahrung"


# --- loading the sample file ------------------------------------------


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    if not SAMPLE.exists():
        pytest.skip("run `python generate_sample.py` first")
    return load_file(SAMPLE)


def test_skips_junk_preamble(frame):
    # Four preamble lines sit above the header. If they leaked through we
    # would see rows with a null date.
    assert frame["date"].notna().all()
    assert len(frame) == 288


def test_canonical_columns_present(frame):
    assert {"date", "description", "reference", "amount", "currency", "month"} <= set(
        frame.columns
    )


def test_dates_parsed_as_datetimes(frame):
    assert pd.api.types.is_datetime64_any_dtype(frame["date"])
    assert frame["date"].min().strftime("%Y-%m-%d") == "2026-03-01"
    assert frame["date"].max().strftime("%Y-%m-%d") == "2026-08-28"


def test_six_months_present(frame):
    assert sorted(frame["month"].unique()) == [
        "2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08",
    ]


def test_sign_convention(frame):
    """Spending is negative, salary is positive."""
    salary = frame[frame["description"].str.contains("GEHALT|LOHN", regex=True)]
    assert len(salary) == 6
    assert (salary["amount"] > 0).all()

    rent = frame[frame["description"].str.contains("MIETE|HAUSVERWALTUNG", regex=True)]
    assert (rent["amount"] < 0).all()


def test_sorted_by_date(frame):
    assert frame["date"].is_monotonic_increasing


def test_subscriptions_are_findable(frame):
    """The planted Spotify charges exist under three different merchant
    strings -- this is what the categoriser has to collapse."""
    spotify = frame[frame["description"].str.contains("SPOTIFY")]
    assert len(spotify) == 6
    assert spotify["amount"].unique().tolist() == [-10.99]
    assert spotify["description"].nunique() > 1


def test_missing_file_raises():
    with pytest.raises(ParseError):
        load_file("sample_data/does_not_exist.csv")