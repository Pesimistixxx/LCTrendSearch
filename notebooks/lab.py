"""Shared setup of the modeling notebooks: paths, loaders, chart style.

Every notebook starts with ``from lab import *``. Paths come from the
training config (src/lctrend/resources/training.json), so notebooks, the
scripts and the trainer image read the same files.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)  # config paths are relative to the repository

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from lctrend.modeling.config import load_config  # noqa: E402
from lctrend.modeling.storage import RunLayout  # noqa: E402

CONFIG = load_config()
# The same run the scripts, the pipeline and the trainer image write to.
LAYOUT = RunLayout.at(CONFIG["run"])
RUN_DIR = LAYOUT.root
DATASET_DIR = LAYOUT.dataset
MODELS_DIR = LAYOUT.models
LABELING_DIR = LAYOUT.labeling

# Categorical slots 1-3 of the validated reference palette (blue, orange,
# aqua): distinct for colour-blind readers when three or fewer series are
# on one chart. More series fold into "other" or separate charts.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = [BLUE, ORANGE, AQUA]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update(
    {
        "figure.figsize": (8, 3.6),
        "figure.dpi": 110,
        "axes.prop_cycle": plt.cycler(color=SERIES),
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "patch.linewidth": 0,
    }
)
# Notebook output is for people: no pandas performance chatter.
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
pd.set_option("display.max_colwidth", 80)
pd.set_option("display.width", 160)


def num(value) -> str:
    """12 345 — thousands split by a space, as in Russian text."""
    return f"{int(value):,}".replace(",", " ")


def short(text, limit=60) -> str:
    """A long name cut at a word, with an ellipsis."""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def read(path, **options) -> pd.DataFrame:
    """A CSV of the run as a frame; empty cells stay NaN."""
    return pd.read_csv(path, low_memory=False, **options)


def read_jsonl(path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as stream:
        return pd.DataFrame(
            json.loads(line) for line in stream if line.strip()
        )


def bar(ax, labels, values, color=BLUE, horizontal=False):
    """Thin bars with a surface gap, values as muted text at the ends."""
    positions = range(len(values))
    if horizontal:
        ax.barh(positions, values, color=color, height=0.7)
        ax.set_yticks(list(positions), labels)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        for position, value in zip(positions, values):
            ax.text(
                value,
                position,
                f" {value:,.0f}",
                va="center",
                color=MUTED,
                fontsize=8,
            )
    else:
        ax.bar(positions, values, color=color, width=0.7)
        ax.set_xticks(list(positions), labels)
        ax.grid(axis="x", visible=False)
    return ax


__all__ = [
    "ROOT",
    "CONFIG",
    "RUN_DIR",
    "DATASET_DIR",
    "MODELS_DIR",
    "LABELING_DIR",
    "BLUE",
    "ORANGE",
    "AQUA",
    "SERIES",
    "MUTED",
    "plt",
    "pd",
    "json",
    "Path",
    "read",
    "num",
    "short",
    "read_jsonl",
    "bar",
]
