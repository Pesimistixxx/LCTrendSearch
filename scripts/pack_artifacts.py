"""Pack modeling artifacts into git-sized parts.

``artifacts/modeling/`` is not in git (runs are regenerated), but moving a
finished run between machines needs it. This packs a directory into one
gzip tar stream cut into parts below GitHub's 100 MB file limit, and
unpacks the parts back. Nothing else is read or written.

    python scripts/pack_artifacts.py            # modeling -> upload/
    python scripts/pack_artifacts.py --unpack   # upload/ -> modeling
    # skip files whose path contains a word:
    python scripts/pack_artifacts.py --exclude subgraphs
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEM = "modeling.tar.gz"


class PartWriter:
    """A write-only stream that rolls over to a new file every ``size``."""

    def __init__(self, directory: Path, size: int) -> None:
        self.directory, self.size = directory, size
        self.index, self.written, self.stream = -1, 0, None
        self.paths = []
        self._next()

    def _next(self) -> None:
        if self.stream:
            self.stream.close()
        self.index += 1
        path = self.directory / f"{STEM}.{self.index:03d}"
        self.paths.append(path)
        self.stream, self.written = path.open("wb"), 0

    def write(self, data: bytes) -> int:
        view = memoryview(data)
        while view:
            if self.written >= self.size:
                self._next()
            chunk = view[: self.size - self.written]
            self.stream.write(chunk)
            self.written += len(chunk)
            view = view[len(chunk) :]
        return len(data)

    def close(self) -> None:
        self.stream.close()


class PartReader:
    """Reads the parts back as one stream, in name order."""

    def __init__(self, paths) -> None:
        self.paths, self.stream = list(paths), None

    def read(self, size: int = -1) -> bytes:
        out = b""
        while size < 0 or len(out) < size:
            if self.stream is None:
                if not self.paths:
                    break
                self.stream = self.paths.pop(0).open("rb")
            data = self.stream.read(-1 if size < 0 else size - len(out))
            if not data:
                self.stream.close()
                self.stream = None
                continue
            out += data
        return out


def pack(source: Path, output: Path, part_mb: int, exclude) -> list:
    output.mkdir(parents=True, exist_ok=True)
    for old in output.glob(f"{STEM}.*"):
        old.unlink()
    writer = PartWriter(output, part_mb * 1024 * 1024)

    def keep(info: tarfile.TarInfo):
        return None if any(word in info.name for word in exclude) else info

    with tarfile.open(fileobj=writer, mode="w|gz") as archive:
        archive.add(source, arcname=source.name, filter=keep)
    writer.close()
    return writer.paths


def unpack(parts: Path, target: Path) -> Path:
    paths = sorted(parts.glob(f"{STEM}.*"))
    if not paths:
        raise FileNotFoundError(f"No {STEM}.* parts in {parts}")
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=PartReader(paths), mode="r|gz") as archive:
        archive.extractall(target.parent, filter="data")
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    artifacts = ROOT / "artifacts"
    parser.add_argument("--source", type=Path, default=artifacts / "modeling")
    parser.add_argument("--parts", type=Path, default=artifacts / "upload")
    parser.add_argument("--part-mb", type=int, default=90)
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--unpack", action="store_true")
    args = parser.parse_args(argv)
    if args.unpack:
        print(f"Unpacked into {unpack(args.parts, args.source)}")
        return
    paths = pack(args.source, args.parts, args.part_mb, args.exclude)
    total = sum(path.stat().st_size for path in paths) / 1024 / 1024
    print(f"{len(paths)} parts, {total:.0f} MB in {args.parts}")


if __name__ == "__main__":
    sys.exit(main())
