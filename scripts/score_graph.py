"""Score every technology of the graph (level 3): python scripts/score_graph.py

Merges duplicates in Neo4j, computes each technology's features and
subgraph as of the latest data, applies the trained models of a run
(CatBoost, HGT, stacked; the winner of report.json decides) and writes
signal_probability, signal_flag, signal_model and signal_snapshot onto the
Technology nodes. Scores go to artifacts/modeling/<run>/labeling/graph-<date>/.

    python scripts/score_graph.py                      # the default run
    python scripts/score_graph.py --run 2026-09-29-dedup --no-llm-judge
    python scripts/score_graph.py --dry-run            # plan merges, no writes
    python scripts/score_graph.py --skip-dedup --no-write
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lctrend.modeling.labeling.graph_scoring import main  # noqa: E402

if __name__ == "__main__":
    main()
