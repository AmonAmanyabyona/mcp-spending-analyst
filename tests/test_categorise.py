"""Tests for analyst.categorise.

The important assertion is that merchant variants collapse to one key.
Everything downstream groups on that key, so a failure here shows up as
missing subscriptions rather than as an exception.
"""

from pathlib import Path

import pytest

from analyst.categorise import (
    UNCATEGORISED,
    add_categories,
    categorise,
    clean,
    coverage,
    normalise_merchant,
)
from analyst.parse import load_file

SAMPLE = Path("sample_data/transactions.csv")


# --- cleanup ----------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("PAYPAL *SPOTIFY", "SPOTIFY"),
        ("REWE SAGT DANKE", "REWE"),
        ("REWE MARKT GMBH//BERLIN", "REWE MARKT"),
        ("SPOTIFY AB STOCKHOLM", "SPOTIFY"),
        ("NETFLIX INTERNATIONAL B.V.", "NETFLIX"),
        ("FITX BERLIN MITTE", "FITX"),
        ("SHELL 1234 BERLIN", "SHELL"),
    ],
)
def test_clean_strips_noise(raw, expected):
    assert clean(raw) == expected


# --- merchant collapsing ---------------------------------------------


@pytest.mark.parametrize(
    "variants,expected",
    [
        (["SPOTIFY AB", "PAYPAL *SPOTIFY", "SPOTIFY AB STOCKHOLM"], "Spotify"),
        (["NETFLIX.COM", "NETFLIX INTERNATIONAL B.V."], "Netflix"),
        (["APPLE.COM/BILL", "APPLE COM BILL ITUNES"], "Apple"),
        (["FITX BERLIN MITTE", "FITX GMBH", "FITX BERLIN"], "FitX"),
        (["REWE SAGT DANKE", "REWE MARKT GMBH//BERLIN", "REWE 4712 BERLIN"], "REWE"),
        (["AMZN MKTP DE", "AMAZON.DE*MK2YT", "AMAZON PAYMENTS"], "Amazon"),
        (["BVG FAHRAUSWEIS", "BVG APP TICKET", "BVG BERLIN"], "BVG"),
        (["DEUTSCHE BAHN AG", "DB VERTRIEB GMBH"], "Deutsche Bahn"),
    ],
)
def test_variants_collapse_to_one_merchant(variants, expected):
    """This is the test that matters. If merchant variants don't collapse,
    recurring charge detection silently finds nothing."""
    assert {normalise_merchant(v) for v in variants} == {expected}


def test_unknown_merchant_keeps_first_two_words():
    assert normalise_merchant("ZUM SCHWARZEN CAFE") == "Zum Schwarzen"


def test_empty_description():
    assert normalise_merchant("") == "Unknown"


# --- categories -------------------------------------------------------


@pytest.mark.parametrize(
    "description,expected",
    [
        ("REWE SAGT DANKE", "groceries"),
        ("PAYPAL *SPOTIFY", "entertainment"),
        ("BVG FAHRAUSWEIS", "transport"),
        ("FITX GMBH", "fitness"),
        ("MIETE WOHNUNG", "housing"),
        ("GEHALT NOVADATA GMBH", "income"),
        ("VATTENFALL EUROPE", "utilities"),
        ("SHELL 1234 BERLIN", "fuel"),
        ("BONANZA COFFEE", "dining"),          # via keyword fallback
        ("APOTHEKE AM MARKT", "health"),       # via keyword fallback
    ],
)
def test_categorise(description, expected):
    assert categorise(description) == expected


def test_unknown_falls_through():
    assert categorise("QWERTYUIOP XYZ") == UNCATEGORISED


# --- against the sample file -----------------------------------------


@pytest.fixture(scope="module")
def frame():
    if not SAMPLE.exists():
        pytest.skip("run `python generate_sample.py` first")
    return add_categories(load_file(SAMPLE))


def test_adds_columns(frame):
    assert "merchant" in frame.columns
    assert "category" in frame.columns


def test_spotify_collapses_in_real_data(frame):
    spotify = frame[frame["merchant"] == "Spotify"]
    assert len(spotify) == 6
    assert spotify["category"].unique().tolist() == ["entertainment"]


def test_coverage_is_high(frame):
    """Uncategorised spending should be a small share of the total.
    If this fails, print `coverage(frame)` and add rules."""
    spending = frame[frame["amount"] < 0]
    missed = spending[spending["category"] == UNCATEGORISED]
    share = missed["amount"].sum() / spending["amount"].sum()
    assert share < 0.15, f"{share:.1%} of spending uncategorised:\n{coverage(frame)}"


def test_coverage_report_shape(frame):
    report = coverage(frame)
    assert set(report.columns) == {"merchant", "transactions", "total"}