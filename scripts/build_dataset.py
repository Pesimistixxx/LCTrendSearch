"""Form the training sample (level 1): python scripts/build_dataset.py

Reads the history export and its subgraphs, adds neighbour aggregates,
labels each active snapshot by the technology's own future, groups
related technologies into families, splits them by first appearance and
selects features on train. Writes artifacts/modeling/<run>/dataset/.

    python scripts/build_dataset.py --run 2026-09-29
    python scripts/build_dataset.py --config my.json --target signal_36m
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lctrend.modeling.config import load_config  # noqa: E402
from lctrend.modeling.dataset.builder import build_dataset  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, help="JSON overriding defaults")
    parser.add_argument("--run", help="Run name (default: config or today)")
    parser.add_argument("--target", help="signal_12m or signal_36m")
    args = parser.parse_args(argv)
    config = load_config(args.config, run=args.run, target=args.target)
    summary = build_dataset(config)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in ("target", "split", "classes", "files")
            }
            | {"features": len(summary["features"])},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
