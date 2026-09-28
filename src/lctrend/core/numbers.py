"""One reading of numbers written in source text.

"1,000,000", "1 000 000", "1,5" and "12 345,67" must mean the same thing to
the claim validator and to the economics parser; two readings let a wrong
value pass validation while the right one was rejected.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Optional


def parse_number(text: str) -> Optional[Decimal]:
    """The number a digit group denotes, or None if it is ambiguous.

    Comma or dot groups of exactly three digits are thousands; a single
    comma otherwise is a decimal comma; spaces group thousands.
    """
    digits = re.sub(r"[\s  ]", "", text).rstrip(".,")
    sign = ""
    if digits[:1] in "+-" and digits[:1]:
        sign, digits = digits[0], digits[1:]
    mantissa, exponent = digits, ""
    match = re.fullmatch(r"(.*?)([eE][+-]?\d+)", digits)
    if match:
        mantissa, exponent = match.groups()
    if not mantissa or not re.fullmatch(r"[\d.,]+", mantissa):
        return None
    if "," in mantissa and "." in mantissa:
        mantissa = mantissa.replace(",", "")
    elif "," in mantissa:
        mantissa = (
            mantissa.replace(",", "")
            if re.fullmatch(r"\d{1,3}(?:,\d{3})+", mantissa)
            else mantissa.replace(",", ".")
        )
    if mantissa.count(".") > 1:
        if not re.fullmatch(r"\d{1,3}(?:\.\d{3})+", mantissa):
            return None
        mantissa = mantissa.replace(".", "")
    try:
        return Decimal(sign + mantissa + exponent)
    except InvalidOperation:
        return None
