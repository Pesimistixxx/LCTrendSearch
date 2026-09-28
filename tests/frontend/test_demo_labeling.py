"""The demo search UI must never pass synthetic data off as analysis."""

import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "frontend" / "src"
BANNER = "ДЕМО: синтетические данные"


def component(source, name):
    start = source.index(f"function {name}(")
    end = re.search(r"\n(?:function |const [A-Z])", source[start + 1 :])
    return source[start : start + 1 + end.start()] if end else source[start:]


@pytest.mark.parametrize("screen", ["Results", "Insight"])
def test_result_and_report_screens_carry_the_demo_banner(screen):
    app = (SRC / "App.jsx").read_text(encoding="utf-8")
    assert BANNER in app
    assert "<DemoBanner" in component(app, screen)


def test_demo_banner_is_printed():
    css = (SRC / "styles.css").read_text(encoding="utf-8")
    start = css.index("@media print")
    body = css[start : css.index("\n}\n", start)]
    hidden = re.search(r"([^{}]*)\{\s*display:\s*none", body).group(1)
    assert "demo-banner" not in hidden
    assert re.search(r"\.demo-banner\s*\{[^}]*display:\s*(?!none)", body)


@pytest.mark.parametrize(
    "brand",
    [
        "Gartner", "OECD", "arXiv", "Роспатент", "WIPO", "Nature",
        "МФТИ", "Банк России", "IEEE", "CB Insights", "TAdviser", "Хабр",
        "Telegram", "GitHub", "ЕБС", "ЦБ",
    ],
)
def test_mock_data_has_no_real_attributions_or_brands(brand):
    mock = (SRC / "mock.js").read_text(encoding="utf-8")
    assert brand not in mock


def test_mock_links_point_to_reserved_example_domains():
    mock = (SRC / "mock.js").read_text(encoding="utf-8")
    hosts = set(re.findall(r"https?://([^/'\"]+)", mock))
    assert hosts and hosts <= {"example.org", "example.com"}
