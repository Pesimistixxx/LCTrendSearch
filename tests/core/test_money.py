"""Money of different years and currencies is comparable (feedback: "10$
in 2010 is not 10$ in 2060")."""

import pytest

from lctrend.core.money import (
    CPI_EXTRAPOLATED,
    EXACT,
    FX_NEAREST_YEAR,
    real_fields,
    to_real_usd,
)


def test_dollars_of_an_earlier_year_are_worth_more_today():
    real = to_real_usd(10, "USD", 2010)
    assert real.base_year == 2025
    assert real.status == EXACT
    # CPI-U 2025 / 2010 = 321.962 / 218.076.
    assert real.value == pytest.approx(14.76, abs=0.01)
    assert to_real_usd(10, "USD", 2025).value == pytest.approx(10)


def test_other_currencies_use_the_rate_of_their_own_year():
    # 1 USD = 92.5524 RUB in 2024; 1 000 000 RUB of 2024 in 2025 dollars.
    real = to_real_usd(1_000_000, "RUB", 2024)
    assert real.value == pytest.approx(
        1_000_000 / 92.5524 * 321.962 / 313.698, rel=1e-6
    )
    assert real.status == EXACT


def test_years_outside_the_tables_are_flagged_not_guessed():
    future = to_real_usd(100, "RUB", 2060)
    assert FX_NEAREST_YEAR in future.status
    assert CPI_EXTRAPOLATED in future.status
    assert to_real_usd(100, "USD", 2060).status == CPI_EXTRAPOLATED


@pytest.mark.parametrize(
    "amount,currency,year",
    [
        (10, "$", 2020),
        (10, "XYZ", 2020),
        (None, "USD", 2020),
        (10, "USD", None),
    ],
)
def test_unknown_or_ambiguous_money_has_no_real_value(amount, currency, year):
    assert to_real_usd(amount, currency, year) is None
    assert real_fields(amount, currency, year)["amount_usd_real"] is None
