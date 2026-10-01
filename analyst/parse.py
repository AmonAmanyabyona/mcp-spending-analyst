"""
Bank CSV -> clean DataFrame.

Knows nothing about MCP. Pure pandas, fully testable on its own.

Real exports are hostile in predictable ways, and this module handles each:
  - junk preamble lines above the actual header
  - semicolon delimiters
  - German number format (1.234,50)
  - DD.MM.YYYY dates
  - cp1252 encoding instead of UTF-8
  - different column names per bank

Canonical output schema:
    date        datetime64
    description str      raw merchant string, as the bank wrote it
    reference   str      payment reference / Verwendungszweck
    amount      float    negative = money out
    currency    str
    month       str      "YYYY-MM"
"""

from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path

import pandas as pd

ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
DELIMITERS = (";", ",", "\t")

# Column aliases, normalised (lowercase, no accents, no punctuation).
# Add a bank by adding its header names here -- nothing else changes.
ALIASES: dict[str, tuple[str, ...]] = {
    "date": (
        "buchungstag", "buchungsdatum", "datum", "valutadatum", "wertstellung",
        "date", "booking date", "transaction date",
    ),
    "description": (
        "beschreibung", "auftraggeber empfanger", "beguenstigter zahlungspflichtiger",
        "begunstigter zahlungspflichtiger", "name", "empfanger", "zahlungsempfanger",
        "description", "payee", "merchant", "counterparty",
    ),
    "reference": (
        "verwendungszweck", "buchungstext", "vorgang verwendungszweck",
        "reference", "purpose", "memo", "details",
    ),
    "amount": (
        "betrag", "umsatz", "betrag eur", "soll haben betrag",
        "amount", "value",
    ),
    "currency": ("waehrung", "wahrung", "currency", "wrg"),
}


class ParseError(Exception):
    """Raised when a file cannot be understood as a bank export."""


# --- small helpers ----------------------------------------------------


def normalise_header(name: str) -> str:
    """'Auftraggeber/Empfänger' -> 'auftraggeber empfanger'."""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().replace("ß", "ss")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def parse_amount(raw: str | float | int) -> float:
    """Parse a money string in either German or English format.

    '1.234,50' -> 1234.5     (dot thousands, comma decimal)
    '-10,99'   -> -10.99
    '1,234.50' -> 1234.5     (comma thousands, dot decimal)
    '10.99'    -> 10.99
    '1.234'    -> 1234.0     (3 digits after a lone dot = thousands)
    """
    if isinstance(raw, (int, float)):
        return float(raw)

    text = str(raw).strip()
    if not text:
        raise ValueError("empty amount")

    # Trailing sign: '10,99-' is a real format some banks emit.
    negative = text.startswith("-") or text.endswith("-")
    text = text.strip("+-").strip()
    text = re.sub(r"[^\d.,]", "", text)  # drop currency symbols, spaces

    if not text:
        raise ValueError(f"no digits in amount: {raw!r}")

    has_dot, has_comma = "." in text, "," in text

    if has_dot and has_comma:
        # Whichever appears last is the decimal separator.
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif has_comma:
        text = text.replace(",", ".")
    elif has_dot:
        # A lone dot followed by exactly 3 digits is a thousands separator
        # ('1.234'); anything else is a decimal point ('10.99').
        if re.fullmatch(r"\d{1,3}(\.\d{3})+", text):
            text = text.replace(".", "")

    value = float(text)
    return -value if negative else value


def sniff(path: Path) -> tuple[str, str, int]:
    """Work out the encoding, delimiter, and which line holds the header.

    Returns (encoding, delimiter, header_line_index).
    """
    for encoding in ENCODINGS:
        try:
            lines = path.read_text(encoding=encoding).splitlines()
        except (UnicodeDecodeError, LookupError):
            continue

        for index, line in enumerate(lines[:30]):
            for delimiter in DELIMITERS:
                fields = [normalise_header(f) for f in next(
                    csv.reader([line], delimiter=delimiter)
                )]
                # A header row is one where we recognise both a date column
                # and an amount column. Junk preamble never satisfies both.
                has_date = any(f in ALIASES["date"] for f in fields)
                has_amount = any(f in ALIASES["amount"] for f in fields)
                if has_date and has_amount:
                    return encoding, delimiter, index

        # Decoded cleanly but found no header; no point trying other encodings
        # unless decoding itself failed.
        break

    raise ParseError(
        f"Could not find a header row in {path.name}. "
        f"Expected a line containing a date column and an amount column. "
        f"Add this bank's column names to ALIASES in analyst/parse.py."
    )


def map_columns(columns: list[str]) -> dict[str, str]:
    """Map the file's actual headers onto canonical names."""
    mapping: dict[str, str] = {}
    for column in columns:
        key = normalise_header(column)
        for canonical, options in ALIASES.items():
            if key in options and canonical not in mapping.values():
                mapping[column] = canonical
                break
    return mapping


# --- the public entry point -------------------------------------------


def load_file(path: str | Path) -> pd.DataFrame:
    """Load one bank CSV into the canonical schema."""
    path = Path(path)
    if not path.exists():
        raise ParseError(f"No such file: {path}")

    encoding, delimiter, header_row = sniff(path)

    frame = pd.read_csv(
        path,
        encoding=encoding,
        delimiter=delimiter,
        skiprows=header_row,
        dtype=str,
        keep_default_na=False,
    )

    mapping = map_columns(list(frame.columns))
    missing = {"date", "amount"} - set(mapping.values())
    if missing:
        raise ParseError(
            f"{path.name}: could not find column(s) for {', '.join(sorted(missing))}. "
            f"Headers found: {list(frame.columns)}"
        )

    frame = frame.rename(columns=mapping)
    frame = frame[[c for c in frame.columns if c in ALIASES]]

    # Optional columns -- fill in blanks rather than branching everywhere later.
    for optional in ("description", "reference", "currency"):
        if optional not in frame.columns:
            frame[optional] = ""

    frame["date"] = pd.to_datetime(
        frame["date"], format="%d.%m.%Y", errors="coerce"
    )
    # Fall back to pandas' own guesser for non-German date formats.
    if frame["date"].isna().all():
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")

    frame["amount"] = frame["amount"].map(_safe_amount)

    before = len(frame)
    frame = frame.dropna(subset=["date", "amount"])
    dropped = before - len(frame)
    if dropped:
        print(f"  warning: dropped {dropped} unparseable row(s) from {path.name}")

    if frame.empty:
        raise ParseError(f"{path.name}: no usable rows after parsing.")

    for text_column in ("description", "reference", "currency"):
        frame[text_column] = frame[text_column].astype(str).str.strip()

    frame["month"] = frame["date"].dt.strftime("%Y-%m")

    return frame.sort_values("date").reset_index(drop=True)


def _safe_amount(value: str) -> float | None:
    try:
        return parse_amount(value)
    except (ValueError, TypeError):
        return None


def load_folder(folder: str | Path = "data") -> pd.DataFrame:
    """Load every CSV in a folder and concatenate.

    Duplicate rows across overlapping exports are dropped, which matters
    when you download statements month by month and the ranges overlap.
    """
    folder = Path(folder)
    if not folder.exists():
        raise ParseError(f"No such folder: {folder}")

    files = sorted(folder.glob("*.csv"))
    if not files:
        raise ParseError(
            f"No CSV files in {folder}/. "
            f"Export a statement from your bank and drop it in there."
        )

    frames = [load_file(f) for f in files]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["date", "description", "amount", "reference"]
    )
    return combined.sort_values("date").reset_index(drop=True)