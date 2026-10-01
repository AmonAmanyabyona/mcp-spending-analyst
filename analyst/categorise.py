"""
Merchant strings -> stable merchant keys and spending categories.

Banks write the same merchant a dozen ways:
    SPOTIFY AB
    PAYPAL *SPOTIFY
    SPOTIFY AB STOCKHOLM

All three are one subscription. Collapsing them is what makes recurring
charge detection possible, so this module runs before any grouping.

Two stages, deliberately separate:
  1. generic cleanup  -- strip processor prefixes, store numbers, cities,
                         legal suffixes. Works on merchants we've never seen.
  2. known aliases    -- a lookup table for the cases cleanup can't solve,
                         e.g. AMZN MKTP DE and AMAZON.DE are the same shop
                         but no amount of regex will tell you that.

Categories are assigned from the normalised merchant. Note that
"subscription" is NOT a category -- it is a *pattern over time*, detected
separately in the tools layer. Spotify's category is entertainment whether
or not you pay monthly.
"""

from __future__ import annotations

import re

import pandas as pd

UNCATEGORISED = "uncategorised"

# --- stage 1: generic cleanup ----------------------------------------

PROCESSOR_PREFIXES = re.compile(
    r"^(PAYPAL|SUMUP|SQ|IZ|ELV|STRIPE|KLARNA|SHOPIFY)\s*[\*\-/]?\s*",
)

NOISE = [
    re.compile(r"//.*$"),                       # REWE MARKT GMBH//BERLIN
    re.compile(r"\bSAGT\s+DANKE\b"),
    re.compile(r"\bVIELEN\s+DANK\b"),
    re.compile(r"\bKARTENZAHLUNG\b"),
    re.compile(r"\b(GMBH|MBH|AG|KG|SE|AB|BV|NV|LTD|INC|PLC|EK|CO)\b"),
    re.compile(r"\b(BERLIN|HAMBURG|MUENCHEN|MUNCHEN|KOELN|FRANKFURT|STOCKHOLM"
               r"|AMSTERDAM|DUBLIN|LONDON|MITTE|KREUZBERG|NEUKOELLN)\b"),
    re.compile(r"\b(DEUTSCHLAND|GERMANY|INTERNATIONAL|EUROPE|DE|NL|SE|COM)\b"),
    re.compile(r"\b[A-Z0-9]*\d[A-Z0-9]*\b"),    # anything containing a digit
]

# --- stage 2: known merchants ----------------------------------------
# Checked against the cleaned string. First match wins, so order matters
# where patterns could overlap.

KNOWN: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(AMZN|AMAZON)\b"), "Amazon"),
    (re.compile(r"\bREWE\b"), "REWE"),
    (re.compile(r"\bEDEKA\b"), "Edeka"),
    (re.compile(r"\bLIDL\b"), "Lidl"),
    (re.compile(r"\bALDI\b"), "Aldi"),
    (re.compile(r"\bKAUFLAND\b"), "Kaufland"),
    (re.compile(r"\bPENNY\b"), "Penny"),
    (re.compile(r"\bNETTO\b"), "Netto"),
    (re.compile(r"\bDM\b|DROGERIE"), "dm"),
    (re.compile(r"\bROSSMANN\b"), "Rossmann"),
    (re.compile(r"\bSPOTIFY\b"), "Spotify"),
    (re.compile(r"\bNETFLIX\b"), "Netflix"),
    (re.compile(r"\bDISNEY\b"), "Disney+"),
    (re.compile(r"\b(APPLE|ITUNES)\b"), "Apple"),
    (re.compile(r"\bFITX\b"), "FitX"),
    (re.compile(r"\b(MCFIT|URBAN SPORTS)\b"), "Gym"),
    (re.compile(r"\bBVG\b"), "BVG"),
    (re.compile(r"\b(DEUTSCHE BAHN|DB VERTRIEB|BAHN)\b"), "Deutsche Bahn"),
    (re.compile(r"\b(LIEFERANDO|TAKEAWAY)\b"), "Lieferando"),
    (re.compile(r"\bWOLT\b"), "Wolt"),
    (re.compile(r"\b(UBER|BOLT|FREENOW)\b"), "Rideshare"),
    (re.compile(r"\bSHELL\b"), "Shell"),
    (re.compile(r"\b(ARAL|ESSO|TOTAL)\b"), "Petrol station"),
    (re.compile(r"\bVATTENFALL\b"), "Vattenfall"),
    (re.compile(r"\b(EON|E ON)\b"), "E.ON"),
    (re.compile(r"\b(TELEKOM|VODAFONE|O2|OTELO)\b"), "Mobile/internet"),
    (re.compile(r"\bMEDIAMARKT|SATURN\b"), "MediaMarkt"),
    (re.compile(r"\bIKEA\b"), "IKEA"),
    (re.compile(r"\bH M HENNES|HENNES\b"), "H&M"),
    (re.compile(r"\bZARA\b"), "Zara"),
    (re.compile(r"\bUNIQLO\b"), "Uniqlo"),
    (re.compile(r"\b(MIETE|HAUSVERWALTUNG)\b"), "Rent"),
    (re.compile(r"\b(GEHALT|LOHN|SALARY)\b"), "Salary"),
]

# --- categories -------------------------------------------------------

CATEGORIES: dict[str, tuple[str, ...]] = {
    "groceries": ("REWE", "Edeka", "Lidl", "Aldi", "Kaufland", "Penny", "Netto"),
    "household": ("dm", "Rossmann", "IKEA"),
    "dining": ("Lieferando", "Wolt"),
    "transport": ("BVG", "Deutsche Bahn", "Rideshare"),
    "fuel": ("Shell", "Petrol station"),
    "shopping": ("Amazon", "MediaMarkt", "H&M", "Zara", "Uniqlo"),
    "entertainment": ("Spotify", "Netflix", "Disney+", "Apple"),
    "fitness": ("FitX", "Gym"),
    "utilities": ("Vattenfall", "E.ON", "Mobile/internet"),
    "housing": ("Rent",),
    "income": ("Salary",),
}

# Fallback keywords, checked against the cleaned string when the merchant
# isn't in KNOWN. Catches the long tail of cafes, pharmacies and the like.
#
# ORDER MATTERS -- first match wins. Specific categories go before generic
# ones, and genuinely ambiguous keywords are left out entirely. A bare
# "MARKT" would steal "APOTHEKE AM MARKT" for groceries and quietly inflate
# the food total, so only unambiguous compounds are listed.
KEYWORD_CATEGORIES: dict[str, tuple[str, ...]] = {
    "health": ("APOTHEKE", "PRAXIS", "ZAHNARZT", "ARZT", "KLINIK"),
    "dining": ("CAFE", "KAFFEE", "COFFEE", "RESTAURANT", "PIZZA", "BURGER",
               "BAR ", "IMBISS", "BACKEREI", "BAECKEREI", "KANTINE"),
    "transport": ("TAXI", "TICKET", "FAHRKARTE", "FLIXBUS"),
    "groceries": ("SUPERMARKT", "MARKTHALLE", "BIO COMPANY"),
}

MERCHANT_TO_CATEGORY = {
    merchant: category
    for category, merchants in CATEGORIES.items()
    for merchant in merchants
}


def clean(raw: str) -> str:
    """Generic cleanup. Returns an uppercase, noise-free string."""
    text = str(raw).upper().strip()
    text = PROCESSOR_PREFIXES.sub("", text)
    # Join dotted acronyms before stripping punctuation, so "B.V." becomes
    # "BV" -- one token the legal-suffix rule can match -- not "B V".
    text = re.sub(r"\b([A-Z])\.", r"\1", text)
    text = re.sub(r"[^A-Z0-9/]+", " ", text)
    for pattern in NOISE:
        text = pattern.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalise_merchant(raw: str) -> str:
    """Collapse a raw bank description into a stable merchant name.

    >>> normalise_merchant("PAYPAL *SPOTIFY")
    'Spotify'
    >>> normalise_merchant("REWE 4712 BERLIN")
    'REWE'
    """
    cleaned = clean(raw)

    for pattern, canonical in KNOWN:
        if pattern.search(cleaned):
            return canonical

    if not cleaned:
        return "Unknown"

    # Unknown merchant: keep the first two words as the key. Short enough to
    # group reliably, long enough to stay readable.
    return " ".join(cleaned.split()[:2]).title()


def categorise(raw: str, reference: str = "") -> str:
    """Assign a spending category to a raw bank description."""
    merchant = normalise_merchant(raw)
    if merchant in MERCHANT_TO_CATEGORY:
        return MERCHANT_TO_CATEGORY[merchant]

    haystack = f"{clean(raw)} {clean(reference)}"
    for category, keywords in KEYWORD_CATEGORIES.items():
        if any(keyword in haystack for keyword in keywords):
            return category

    return UNCATEGORISED


def add_categories(frame: pd.DataFrame) -> pd.DataFrame:
    """Add `merchant` and `category` columns to a parsed DataFrame."""
    frame = frame.copy()
    frame["merchant"] = frame["description"].map(normalise_merchant)
    frame["category"] = [
        categorise(description, reference)
        for description, reference in zip(frame["description"], frame["reference"])
    ]
    return frame


def coverage(frame: pd.DataFrame) -> pd.DataFrame:
    """What fell through the rules, worst first, by money involved.

    Run this after every rule change. Uncategorised spending is the one
    number that tells you whether the category totals can be trusted.
    """
    if "category" not in frame.columns:
        frame = add_categories(frame)

    missed = frame[frame["category"] == UNCATEGORISED]
    if missed.empty:
        return pd.DataFrame(columns=["merchant", "transactions", "total"])

    report = (
        missed.groupby("merchant")
        .agg(transactions=("amount", "size"), total=("amount", "sum"))
        .sort_values("total")
        .reset_index()
    )
    return report