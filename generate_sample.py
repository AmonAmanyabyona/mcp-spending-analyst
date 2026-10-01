"""
Generate fake bank transactions for development and demos.

Deliberately imitates a German bank export, with the awkward bits intact:
semicolon delimiter, comma decimal separator, DD.MM.YYYY dates, a few junk
preamble lines above the real header, and inconsistent merchant strings.
Those quirks are the point -- the parser has to be real.

Writes to sample_data/transactions.csv. Deterministic: same seed, same
file, so tests can assert on exact numbers.

    python generate_sample.py
"""

import csv
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 42
MONTHS = 6  # counting back from END_MONTH
END_MONTH = (2026, 8)
OUT = Path("sample_data/transactions.csv")

# --- planted patterns ------------------------------------------------
# These exist so the demo has something to find. The gym is the hook:
# charged every month, but no other spending nearby after April.

SUBSCRIPTIONS = [
    # (merchant variants, amount, day of month)
    (["SPOTIFY AB", "PAYPAL *SPOTIFY", "SPOTIFY AB STOCKHOLM"], 10.99, 3),
    (["NETFLIX.COM", "NETFLIX INTERNATIONAL B.V."], 13.99, 11),
    (["APPLE.COM/BILL", "APPLE COM BILL ITUNES"], 2.99, 17),
    (["FITX BERLIN MITTE", "FITX GMBH", "FITX BERLIN"], 18.89, 2),
]

RENT = (["MIETE WOHNUNG", "HAUSVERWALTUNG KELLER GMBH"], 1150.00, 1)
SALARY = (["GEHALT NOVADATA GMBH", "LOHN/GEHALT NOVADATA"], 3200.00, 28)

# --- everyday spending -----------------------------------------------
# (merchant variants, min, max, roughly how many per month)

REGULARS = [
    (["REWE SAGT DANKE", "REWE MARKT GMBH//BERLIN", "REWE 4712 BERLIN"], 8, 65, 8),
    (["EDEKA MOELLER", "EDEKA SUEDWEST", "EDEKA//BERLIN"], 6, 48, 4),
    (["LIDL SAGT DANKE", "LIDL DIENSTLEISTUNG"], 5, 40, 3),
    (["DM DROGERIE MARKT", "DM-DROGERIE MARKT SAGT DANKE"], 4, 32, 2),
    (["BVG FAHRAUSWEIS", "BVG APP TICKET", "BVG BERLIN"], 2, 11, 5),
    (["DEUTSCHE BAHN AG", "DB VERTRIEB GMBH"], 12, 95, 1),
    (["PAYPAL *LIEFERANDO", "LIEFERANDO.DE", "PAYPAL *TAKEAWAY"], 11, 38, 4),
    (["ZUM SCHWARZEN CAFE", "CAFE EINSTEIN", "BONANZA COFFEE"], 3, 19, 6),
    (["AMZN MKTP DE", "AMAZON.DE*MK2YT", "AMAZON PAYMENTS"], 7, 120, 3),
    (["H&M HENNES MAURITZ", "ZARA DEUTSCHLAND", "UNIQLO BERLIN"], 15, 85, 1),
    (["SHELL 1234 BERLIN", "ARAL TANKSTELLE"], 30, 70, 1),
    (["VATTENFALL EUROPE", "VATTENFALL SALES GMBH"], 48, 62, 1),
]

REFERENCES = [
    "Kartenzahlung",
    "Lastschrift",
    "SEPA-Ueberweisung",
    "Kartenzahlung girocard",
    "Dauerauftrag",
    "Online-Ueberweisung",
]


def month_range(end: tuple[int, int], count: int) -> list[tuple[int, int]]:
    """The `count` months ending at `end`, oldest first."""
    year, month = end
    months = []
    for _ in range(count):
        months.append((year, month))
        month -= 1
        if month == 0:
            year, month = year - 1, 12
    return list(reversed(months))


def safe_day(year: int, month: int, day: int) -> date:
    """Clamp a day to the end of the month (no 31 February)."""
    if month == 12:
        last = 31
    else:
        last = (date(year, month + 1, 1) - timedelta(days=1)).day
    return date(year, month, min(day, last))


def german_amount(value: float) -> str:
    """1234.5 -> '1.234,50' (thousands dot, comma decimal)."""
    whole, _, frac = f"{abs(value):.2f}".partition(".")
    grouped = f"{int(whole):,}".replace(",", ".")
    sign = "-" if value < 0 else ""
    return f"{sign}{grouped},{frac}"


def build_rows(rng: random.Random) -> list[dict]:
    rows = []

    def add(day: date, merchant: str, amount: float) -> None:
        rows.append(
            {
                "Buchungstag": day.strftime("%d.%m.%Y"),
                "Beschreibung": merchant,
                "Verwendungszweck": rng.choice(REFERENCES),
                "Betrag": german_amount(amount),
                "Waehrung": "EUR",
            }
        )

    for index, (year, month) in enumerate(month_range(END_MONTH, MONTHS)):
        # income and rent
        names, amount, day = SALARY
        add(safe_day(year, month, day), rng.choice(names), amount)

        names, amount, day = RENT
        add(safe_day(year, month, day), rng.choice(names), -amount)

        # subscriptions, with small jitter on the charge date
        for names, amount, day in SUBSCRIPTIONS:
            jitter = rng.choice([-1, 0, 0, 0, 1])
            add(safe_day(year, month, day + jitter), rng.choice(names), -amount)

        # everyday spending
        for names, low, high, per_month in REGULARS:
            # the gym stops being visited after the third month, but keeps
            # charging -- this is the thing the demo should surface
            count = rng.randint(max(1, per_month - 2), per_month + 2)
            for _ in range(count):
                day_of_month = rng.randint(1, 28)
                amount = -round(rng.uniform(low, high), 2)
                add(safe_day(year, month, day_of_month), rng.choice(names), amount)

        # one irregular large purchase every couple of months
        if index % 2 == 1:
            add(
                safe_day(year, month, rng.randint(5, 25)),
                rng.choice(["MEDIAMARKT BERLIN", "IKEA BERLIN TEMPELHOF"]),
                -round(rng.uniform(80, 340), 2),
            )

    rows.sort(key=lambda r: tuple(reversed(r["Buchungstag"].split("."))))
    return rows


def main() -> None:
    rng = random.Random(SEED)
    rows = build_rows(rng)

    OUT.parent.mkdir(parents=True, exist_ok=True)

    with OUT.open("w", newline="", encoding="utf-8") as handle:
        # Junk preamble, exactly like a real export. The parser has to skip
        # these to find the header row.
        handle.write("Kontoumsaetze;;;;\n")
        handle.write("Konto;DE00 0000 0000 0000 0000 00;;;\n")
        handle.write(f"Zeitraum;{rows[0]['Buchungstag']} - {rows[-1]['Buchungstag']};;;\n")
        handle.write(";;;;\n")

        writer = csv.DictWriter(
            handle,
            fieldnames=["Buchungstag", "Beschreibung", "Verwendungszweck", "Betrag", "Waehrung"],
            delimiter=";",
        )
        writer.writeheader()
        writer.writerows(rows)

    spend = sum(
        float(r["Betrag"].replace(".", "").replace(",", "."))
        for r in rows
        if r["Betrag"].startswith("-")
    )
    print(f"wrote {len(rows)} rows to {OUT}")
    print(f"period: {rows[0]['Buchungstag']} to {rows[-1]['Buchungstag']}")
    print(f"total outgoing: {abs(spend):,.2f} EUR")
    print(f"subscriptions planted: {len(SUBSCRIPTIONS)} "
          f"(~{sum(s[1] for s in SUBSCRIPTIONS):.2f} EUR/month)")


if __name__ == "__main__":
    main()