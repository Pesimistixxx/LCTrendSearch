"""Comparable money: constant US dollars of one base year.

10 USD of 2010 and 10 USD of 2025 are different amounts, and so are 10
USD and 10 EUR. A model trained on nominal amounts learns the inflation
and the exchange rates of its training years instead of the technology,
and a later snapshot drifts out of its range. Every amount therefore also
gets a real value: converted to US dollars at the average rate of its own
year, then deflated to dollars of ``base_year`` (money.json).

The conversion is deterministic reference data, not a model's guess; the
nominal amount and currency stay stored next to it, and a year outside the
tables is flagged instead of silently treated as exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

from .config import load_catalog

EXACT = "exact"
FX_NEAREST_YEAR = "fx_nearest_year"
CPI_EXTRAPOLATED = "cpi_extrapolated"


@dataclass(frozen=True)
class RealAmount:
    value: float
    base_year: int
    # exact, or the approximations used, joined by "+".
    status: str


@lru_cache(maxsize=1)
def _tables() -> Tuple[int, Dict[int, float], Dict[str, Dict[int, float]]]:
    catalog = load_catalog("money")
    deflator = {
        int(year): float(value) for year, value in catalog["deflator"].items()
    }
    fx = {
        currency: {int(year): float(value) for year, value in years.items()}
        for currency, years in catalog["fx"].items()
    }
    return int(catalog["base_year"]), deflator, fx


def _nearest(table: Dict[int, float], year: int) -> Tuple[float, bool]:
    if year in table:
        return table[year], True
    nearest = min(table, key=lambda known: (abs(known - year), -known))
    return table[nearest], False


def to_real_usd(
    amount: Optional[float], currency: Optional[str], year: Optional[int]
) -> Optional[RealAmount]:
    """``amount`` of ``currency`` in ``year`` as dollars of the base year.

    None when the amount, the year or the currency is unknown, including
    an ambiguous symbol ($, ¥): it is never guessed.
    """
    if amount is None or year is None or not currency:
        return None
    if not isinstance(amount, (int, float)) or not math.isfinite(amount):
        return None
    base_year, deflator, fx = _tables()
    currency = currency.strip().upper()
    status = []
    if currency == "USD":
        dollars = float(amount)
    elif currency in fx and fx[currency]:
        rate, exact = _nearest(fx[currency], year)
        if not exact:
            status.append(FX_NEAREST_YEAR)
        if rate <= 0:
            return None
        dollars = float(amount) / rate
    else:
        return None
    index, exact = _nearest(deflator, year)
    if not exact:
        status.append(CPI_EXTRAPOLATED)
    base_index = deflator.get(base_year) or _nearest(deflator, base_year)[0]
    return RealAmount(
        value=dollars * base_index / index,
        base_year=base_year,
        status="+".join(status) or EXACT,
    )


def real_fields(
    amount: Optional[float], currency: Optional[str], year: Optional[int]
) -> Dict[str, Any]:
    """The real-value fields stored next to a nominal amount."""
    real = to_real_usd(amount, currency, year)
    if real is None:
        return {
            "amount_usd_real": None,
            "real_base_year": None,
            "real_status": None,
        }
    return {
        "amount_usd_real": real.value,
        "real_base_year": real.base_year,
        "real_status": real.status,
    }
