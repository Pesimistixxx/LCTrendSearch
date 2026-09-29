"""Train, compare and apply the models (level 2): python scripts/train_model.py

Trains CatBoost, HGT and their stack on artifacts/modeling/<run>/dataset/,
writes models and report.json to <run>/models/ and the probability of every
technology to <run>/labeling/scores.csv. Builds the dataset first when it
does not exist yet (or with --rebuild).

    python scripts/train_model.py --run 2026-09-29
    python scripts/train_model.py --config my.json --rebuild
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lctrend.modeling.config import load_config  # noqa: E402
from lctrend.modeling.dataset.builder import build_dataset  # noqa: E402
from lctrend.modeling.storage import RunLayout  # noqa: E402
from lctrend.modeling.training.pipeline import train_pipeline  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, help="JSON overriding defaults")
    parser.add_argument("--run", help="Run name (default: config or today)")
    parser.add_argument("--target", help="signal_12m or signal_36m")
    parser.add_argument(
        "--rebuild", action="store_true", help="Rebuild the dataset first"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = load_config(args.config, run=args.run, target=args.target)
    dataset = RunLayout.at(config.get("run")).dataset / "dataset.json"
    if args.rebuild or not dataset.exists():
        logging.info("Building the dataset into %s", dataset.parent)
        build_dataset(config)
    report = train_pipeline(config)
    summary = {
        "winner": report["winner"],
        "rows": report["rows"],
        "positive_families": report["positive_families"],
        "test": {
            name: {
                key: model["metrics"].get("test", {}).get(key)
                for key in (
                    "pr_auc",
                    "pr_auc_family_ci95",
                    "roc_auc",
                    "positive_rate",
                    "at_valid_f1_threshold",
                )
            }
            for name, model in report["models"].items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
