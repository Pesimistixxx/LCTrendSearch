"""Where every stage of the modeling domain keeps its files.

One run is one directory, ``artifacts/modeling/<run>/``:

- ``dataset/``  level 1: history CSV, subgraphs, neighbour features, labels;
- ``models/``   level 2: trained models and their JSON reports;
- ``labeling/`` level 3: duplicate plans, merge logs, final scores.

Every file a stage writes gets ``<name>.manifest.json`` next to it: the
stage, its parameters, a summary and the SHA-256 of each input. A later
stage, or a person, can then tell which exact files a result came from.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

DEFAULT_BASE = Path("artifacts/modeling")


@dataclass(frozen=True)
class RunLayout:
    root: Path

    @classmethod
    def at(cls, run: Optional[str] = None, base: Path = DEFAULT_BASE):
        return cls(Path(base) / (run or date.today().isoformat()))

    @property
    def dataset(self) -> Path:
        return self.root / "dataset"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def labeling(self) -> Path:
        return self.root / "labeling"


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_path(output: Path) -> Path:
    output = Path(output)
    return output.with_name(output.name + ".manifest.json")


def write_manifest(
    output: Path,
    stage: str,
    inputs: Iterable[Path] = (),
    parameters: Optional[Dict[str, Any]] = None,
    summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    manifest = {
        "stage": stage,
        "output": Path(output).name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": [
            {"path": str(path), "sha256": file_digest(path)} for path in inputs
        ],
        "parameters": parameters or {},
        "summary": summary or {},
    }
    manifest_path(output).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return manifest
